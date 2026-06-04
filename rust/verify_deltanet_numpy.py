#!/usr/bin/env python3
"""
Verify native DeltaNet by reimplementing it in NumPy using the SAME GGUF weights.
"""

import numpy as np
import struct

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

def rms_norm(x, weight, eps=1e-6):
    # x: [seq_len, hidden_size]
    scale = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x * scale * weight

def l2_normalize_per_head(x, eps=1e-6):
    # x: [seq_len, num_heads, head_dim]
    norm = np.sqrt(np.sum(x**2, axis=-1, keepdims=True)) + eps
    return x / norm

def causal_conv1d(x, weight):
    # x: [seq_len, channels]
    # weight: [kernel_size, channels]
    seq_len, channels = x.shape
    kernel_size = weight.shape[0]
    out = np.zeros_like(x)
    for c in range(channels):
        for t in range(seq_len):
            for k in range(min(kernel_size, t + 1)):
                w_idx = kernel_size - 1 - k
                out[t, c] += x[t - k, c] * weight[w_idx, c]
    return out

def deltanet_forward_numpy(hidden, weights, params):
    seq_len, hidden_size = hidden.shape
    num_qk_heads, num_v_heads = params['num_qk_heads'], params['num_v_heads']
    head_k_dim, head_v_dim = params['head_k_dim'], params['head_v_dim']
    conv_dim = params['conv_dim']
    
    # 1. QKV projection
    qkv_proj = hidden @ weights['attn_qkv']  # [seq, hidden] @ [hidden, conv_dim]
    
    # 2. Causal Conv1d + SiLU
    conv_w = weights['ssm_conv1d']  # [kernel, channels]
    conv_out = causal_conv1d(qkv_proj, conv_w)
    conv_out = silu(conv_out)
    
    # 3. Split Q, K, V
    q_total = num_qk_heads * head_k_dim
    k_total = num_qk_heads * head_k_dim
    v_total = num_v_heads * head_v_dim
    
    q = conv_out[:, :q_total].reshape(seq_len, num_qk_heads, head_k_dim)
    k = conv_out[:, q_total:q_total+k_total].reshape(seq_len, num_qk_heads, head_k_dim)
    v = conv_out[:, q_total+k_total:].reshape(seq_len, num_v_heads, head_v_dim)
    
    # 4. L2 normalize Q, K
    q = l2_normalize_per_head(q)
    k = l2_normalize_per_head(k)
    
    # 5. Decay rates
    alpha = hidden @ weights['ssm_alpha']  # [seq, num_v_heads]
    dt_bias = weights['ssm_dt_bias']  # [num_v_heads]
    a = weights['ssm_a']  # [num_v_heads]
    
    decay = np.zeros((seq_len, num_v_heads))
    for s in range(seq_len):
        for h in range(num_v_heads):
            dt = np.log(1 + np.exp(alpha[s, h] + dt_bias[h]))
            decay[s, h] = np.exp(dt * a[h])
    
    # 6. Beta gates
    beta_logits = hidden @ weights['ssm_beta']  # [seq, num_v_heads]
    beta = sigmoid(beta_logits)
    
    # 7. Delta rule
    output_dim = num_v_heads * head_v_dim
    output = np.zeros((seq_len, output_dim))
    state = np.zeros((num_v_heads, head_k_dim, head_v_dim))
    
    scale = 1.0 / np.sqrt(head_k_dim)
    
    for s in range(seq_len):
        for h_qk in range(num_qk_heads):
            q_h = q[s, h_qk]
            k_h = k[s, h_qk]
            h_v = h_qk  # 1:1 mapping for Qwen3.5
            
            decay_h = decay[s, h_v]
            beta_h = beta[s, h_v]
            v_h = v[s, h_v]
            
            # v_pred = state @ k
            v_pred = state[h_v] @ k_h  # [v_dim]
            
            # delta = (v - v_pred) * beta
            delta = (v_h - v_pred) * beta_h
            
            # state = decay * state + outer(k, delta)
            state[h_v] = decay_h * state[h_v] + np.outer(k_h, delta)
            
            # output = state @ q * scale
            out_base = h_v * head_v_dim
            output[s, out_base:out_base+head_v_dim] = state[h_v] @ q_h * scale
    
    # 8. Per-head RMSNorm
    norm_w = weights['ssm_norm']  # [head_v_dim]
    output_reshaped = output.reshape(seq_len, num_v_heads, head_v_dim)
    for s in range(seq_len):
        for h in range(num_v_heads):
            x = output_reshaped[s, h]
            scale_norm = 1.0 / np.sqrt(np.mean(x**2) + params['norm_eps'])
            output_reshaped[s, h] = x * scale_norm * norm_w
    output = output_reshaped.reshape(seq_len, output_dim)
    
    # 9. Gate: SiLU(hidden @ attn_gate)
    gate = silu(hidden @ weights['attn_gate'])
    output = output * gate
    
    # 10. Output projection
    ssm_out = output @ weights['ssm_out']
    
    return ssm_out


def load_native_bin(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def main():
    # Load native dumped pre-norm for layer 0
    native_pre_norm = load_native_bin("native_pre_norm_layer0.bin", (4, 1024))
    
    # Load GGUF weights (already dequantized by Rust)
    weights = {
        'attn_qkv': np.fromfile("gguf_qkv_layer0.bin", dtype=np.float32).reshape(1024, 6144),
        'attn_gate': np.fromfile("gguf_gate_layer0.bin", dtype=np.float32).reshape(1024, 2048),
        'ssm_conv1d': np.fromfile("../models/Qwen3.5-0.8B-Q4_0.gguf", dtype=np.uint8),  # We'll extract this properly
    }
    
    # Actually, we need all the weights. Let's dump them from Rust.
    print("Need to dump all layer 0 weights from GGUF. Use dump_gguf_tensor for:")
    print("  blk.0.ssm_conv1d.weight")
    print("  blk.0.ssm_alpha.weight")
    print("  blk.0.ssm_beta.weight")
    print("  blk.0.ssm_dt.bias")
    print("  blk.0.ssm_a")
    print("  blk.0.ssm_norm.weight")
    print("  blk.0.ssm_out.weight")


if __name__ == "__main__":
    main()
