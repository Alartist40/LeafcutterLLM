#!/usr/bin/env python3
"""Dump HF Qwen3.5 layer-0 DeltaNet intermediates for comparison with native."""
import sys
sys.path.insert(0, "/home/xander/Documents/portfolio/LeafcutterLLM/.venv/lib/python3.14/site-packages")

import torch
import numpy as np
from transformers import AutoModelForCausalLM

MODEL_PATH = "/home/xander/Documents/portfolio/LeafcutterLLM/models/Qwen3.5-0.8B-HF"
PROMPT = "2+2="

def main():
    print("Loading HF model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    # Tokenize manually using the model's tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokens = tok(PROMPT, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens["input_ids"]
    print(f"Prompt: '{PROMPT}' -> tokens: {input_ids[0].tolist()}")
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
        )
    
    # Get layer 0 input (after embedding + input_layernorm)
    hidden_states = outputs.hidden_states[0]  # [1, seq_len, hidden_size]
    print(f"Layer 0 input shape: {hidden_states.shape}")
    np.array(hidden_states[0].detach().to(torch.float32).numpy(), dtype=np.float32).tofile("hf_l0_input.bin")
    
    # Extract layer 0 (DeltaNet layer)
    layer = model.model.layers[0]
    assert layer.layer_type == "linear_attention", f"Layer 0 is {layer.layer_type}"
    
    delta = layer.linear_attn
    
    # Manually run DeltaNet forward to extract intermediates
    x = hidden_states
    batch_size, seq_len, hidden_size = x.shape
    
    # 1. in_proj_qkv
    mixed_qkv = delta.in_proj_qkv(x)
    print(f"mixed_qkv shape: {mixed_qkv.shape}")
    np.array(mixed_qkv[0].detach().to(torch.float32).numpy(), dtype=np.float32).tofile("hf_l0_qkv_proj.bin")
    
    # 2. Transpose and conv1d
    mixed_qkv_t = mixed_qkv.transpose(1, 2)
    conv_out = delta.conv1d(mixed_qkv_t)
    conv_out = conv_out[:, :, :mixed_qkv_t.shape[-1]]
    conv_out_silu = torch.nn.functional.silu(conv_out)
    conv_out_silu = conv_out_silu.transpose(1, 2)
    print(f"conv_out shape: {conv_out_silu.shape}")
    np.array(conv_out_silu[0].detach().to(torch.float32).numpy(), dtype=np.float32).tofile("hf_l0_conv_out.bin")
    
    # 3. Split Q, K, V
    query, key, value = torch.split(
        conv_out_silu,
        [delta.key_dim, delta.key_dim, delta.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, delta.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, delta.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, delta.head_v_dim)
    print(f"Q shape: {query.shape}, K shape: {key.shape}, V shape: {value.shape}")
    
    # 4. L2 norm
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
    q_norm = l2norm(query, dim=-1, eps=1e-6)
    k_norm = l2norm(key, dim=-1, eps=1e-6)
    np.array(q_norm[0].detach().to(torch.float32).numpy(), dtype=np.float32).tofile("hf_l0_q_norm.bin")
    np.array(k_norm[0].detach().to(torch.float32).numpy(), dtype=np.float32).tofile("hf_l0_k_norm.bin")
    np.array(value[0].detach().to(torch.float32).numpy(), dtype=np.float32).tofile("hf_l0_v.bin")
    
    # 5. Beta and decay
    b = delta.in_proj_b(x)
    a = delta.in_proj_a(x)
    beta = b.sigmoid()
    g = -delta.A_log.float().exp() * torch.nn.functional.softplus(a.float() + delta.dt_bias)
    print(f"beta shape: {beta.shape}, g shape: {g.shape}")
    np.array(beta[0].detach().to(torch.float32).numpy(), dtype=np.float32).tofile("hf_l0_beta.bin")
    np.array(g[0].detach().to(torch.float32).numpy(), dtype=np.float32).tofile("hf_l0_decay.bin")
    
    # 6. Run chunk_gated_delta_rule (for prefill)
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    core_attn_out, _ = torch_chunk_gated_delta_rule(
        q_norm, k_norm, value,
        g=g, beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    print(f"core_attn_out shape: {core_attn_out.shape}")
    np.array(core_attn_out[0].detach().to(torch.float32).numpy(), dtype=np.float32).tofile("hf_l0_core_attn_out.bin")
    
    # 7. Gated RMSNorm
    z = delta.in_proj_z(x)
    z = z.reshape(batch_size, seq_len, -1, delta.head_v_dim)
    core_flat = core_attn_out.reshape(-1, delta.head_v_dim)
    z_flat = z.reshape(-1, delta.head_v_dim)
    normed = delta.norm(core_flat, z_flat)
    normed = normed.reshape(batch_size, seq_len, -1)
    print(f"post_norm shape: {normed.shape}")
    np.array(normed[0].detach().to(torch.float32).numpy(), dtype=np.float32).tofile("hf_l0_post_norm.bin")
    
    # 8. Output projection
    output = delta.out_proj(normed)
    print(f"ssm_out shape: {output.shape}")
    np.array(output[0].detach().to(torch.float32).numpy(), dtype=np.float32).tofile("hf_l0_ssm_out.bin")
    
    print("\nHF intermediates dumped!")

if __name__ == "__main__":
    main()
