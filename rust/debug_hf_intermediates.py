#!/usr/bin/env python3
"""Dump HF intermediates for Layer 1 DeltaNet for comparison with native."""

import os
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "../models/Qwen3.5-0.8B-HF"
PROMPT = "2+2="
OUT_DIR = "debug_hf_intermediates"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading HF model from local path...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"]

# Hook into layer 1 DeltaNet to dump intermediates
layer1 = model.model.layers[1].linear_attn
print(f"Layer 1 module: {type(layer1).__name__}")
print(f"  conv_dim={layer1.conv_dim}, key_dim={layer1.key_dim}, value_dim={layer1.value_dim}")
print(f"  head_k_dim={layer1.head_k_dim}, head_v_dim={layer1.head_v_dim}")
print(f"  num_k_heads={layer1.num_k_heads}, num_v_heads={layer1.num_v_heads}")
print(f"  conv_kernel_size={layer1.conv_kernel_size}")

intermediates = {}

def save(name, tensor):
    arr = tensor.detach().cpu().float().numpy()
    fname = os.path.join(OUT_DIR, f"hf_l1_{name}.npy")
    np.save(fname, arr)
    print(f"Saved {name}: shape={arr.shape}, mean={arr.mean():.6f}, std={arr.std():.6f}, abs_mean={np.abs(arr).mean():.6f}")

with torch.no_grad():
    # Get embeddings
    hidden = model.model.embed_tokens(input_ids)
    save("embed", hidden)
    
    # Pre-layer norm
    if hasattr(model.model, 'norm') and model.model.norm is not None:
        hidden = model.model.norm(hidden)
        save("post_norm", hidden)
    
    # Input to layer 1
    hidden_states = hidden
    batch_size, seq_len, _ = hidden_states.shape
    
    # Manually compute DeltaNet forward for layer 1
    mixed_qkv = layer1.in_proj_qkv(hidden_states)
    mixed_qkv_t = mixed_qkv.transpose(1, 2)
    save("qkv_proj", mixed_qkv_t)
    
    # Causal conv1d (manual)
    conv_out = torch.nn.functional.conv1d(
        mixed_qkv_t,
        layer1.conv1d.weight,
        bias=None,
        padding=layer1.conv_kernel_size - 1,
        groups=layer1.conv_dim,
    )[:, :, :seq_len]
    save("conv_out", conv_out)
    
    conv_out_silu = torch.nn.functional.silu(conv_out)
    save("conv_out_silu", conv_out_silu)
    
    # Split QKV
    mixed_qkv_t = conv_out_silu.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv_t,
        [layer1.key_dim, layer1.key_dim, layer1.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, layer1.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, layer1.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, layer1.head_v_dim)
    
    # L2 norm (manual, same as kernel)
    def l2norm(x, dim=-1, eps=1e-6):
        return x / (x.pow(2).sum(dim=dim, keepdim=True).sqrt() + eps)
    
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    
    save("q_norm", query)
    save("k_norm", key)
    save("v", value)
    
    # Decay and beta
    a = layer1.in_proj_a(hidden_states)
    b = layer1.in_proj_b(hidden_states)
    
    beta = b.sigmoid()
    g = -layer1.A_log.float().exp() * torch.nn.functional.softplus(a.float() + layer1.dt_bias)
    
    save("decay", g)
    save("beta", beta)
    
    # z gate
    z = layer1.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, layer1.head_v_dim)
    save("z_gate", z)
    
    # Delta rule (manual, using torch_recurrent_gated_delta_rule)
    scale = 1 / (query.shape[-1] ** 0.5)
    query_scaled = query * scale
    
    query_t = query_scaled.transpose(1, 2).contiguous()
    key_t = key.transpose(1, 2).contiguous()
    value_t = value.transpose(1, 2).contiguous()
    beta_t = beta.transpose(1, 2).contiguous()
    g_t = g.transpose(1, 2).contiguous()
    
    core_attn_out = torch.zeros(batch_size, layer1.num_v_heads, seq_len, layer1.head_v_dim, dtype=value.dtype)
    last_recurrent_state = torch.zeros(batch_size, layer1.num_v_heads, layer1.head_k_dim, layer1.head_v_dim, dtype=value.dtype)
    
    for i in range(seq_len):
        q_t = query_t[:, :, i]
        k_t = key_t[:, :, i]
        v_t = value_t[:, :, i]
        gt = g_t[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        bt = beta_t[:, :, i].unsqueeze(-1)
        
        last_recurrent_state = last_recurrent_state * gt
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * bt
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    
    core_attn_out = core_attn_out.transpose(1, 2).contiguous()
    save("core_attn_out", core_attn_out)
    
    # Norm + gate
    core_attn_out_2d = core_attn_out.reshape(-1, layer1.head_v_dim)
    z_2d = z.reshape(-1, layer1.head_v_dim)
    
    # Manual RMSNormGated
    variance = core_attn_out_2d.float().pow(2).mean(-1, keepdim=True)
    normed = core_attn_out_2d.float() * torch.rsqrt(variance + layer1.layer_norm_epsilon)
    normed = layer1.norm.weight * normed
    gated = normed * torch.nn.functional.silu(z_2d.float())
    post_gate = gated.to(core_attn_out.dtype).reshape(batch_size, seq_len, -1)
    save("post_gate", post_gate)
    
    # Output projection
    output = layer1.out_proj(post_gate)
    save("ssm_out", output)
    
    # Full model forward for comparison
    full_out = model(input_ids, output_hidden_states=True)
    full_hidden = full_out.hidden_states[2].detach().cpu().float().numpy()  # After layer 1
    np.save(os.path.join(OUT_DIR, "hf_hidden_02.npy"), full_hidden)
    print(f"\nFull model hidden after layer 1: shape={full_hidden.shape}, mean={full_hidden.mean():.6f}, abs_mean={np.abs(full_hidden).mean():.6f}")
    
    logits = full_out.logits.detach().cpu().float().numpy()
    last_logits = logits[0, -1]
    top_idx = np.argsort(last_logits)[-5:][::-1]
    print(f"HF Logits top-5: {[(int(i), float(last_logits[i])) for i in top_idx]}")
    print(f"HF top token: {top_idx[0]} -> '{tokenizer.decode([top_idx[0]])}'")
