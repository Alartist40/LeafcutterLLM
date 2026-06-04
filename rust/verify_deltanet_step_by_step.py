#!/usr/bin/env python3
"""
Verify native DeltaNet by recomputing each step in NumPy using the SAME GGUF weights,
then comparing with native dumped intermediates.
"""

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

def l2_normalize_per_head(x, eps=1e-6):
    # x: [seq_len, num_heads, head_dim]
    norm = np.sqrt(np.sum(x**2, axis=-1, keepdims=True)) + eps
    return x / norm

def compare(label, computed, native_path, shape):
    native = np.fromfile(native_path, dtype=np.float32).reshape(shape)
    if computed.shape != native.shape:
        print(f"{label}: SHAPE MISMATCH computed={computed.shape} native={native.shape}")
        return
    
    diff = computed - native
    cos = np.dot(computed.flatten(), native.flatten()) / (np.linalg.norm(computed) * np.linalg.norm(native) + 1e-10)
    print(f"{label:25s} | cos_sim={cos:8.5f} | abs_mean_diff={np.abs(diff).mean():.6f} | max_diff={np.abs(diff).max():.6f} | computed_abs_mean={np.abs(computed).mean():.6f} | native_abs_mean={np.abs(native).mean():.6f}")
    return native

# Load pre-norm hidden state from native
pre_norm = np.fromfile("native_pre_norm_layer0.bin", dtype=np.float32).reshape(4, 1024)
print(f"Pre-norm shape: {pre_norm.shape}, abs_mean: {np.abs(pre_norm).mean():.6f}")

# Load GGUF weights
attn_qkv = np.fromfile("blk_0_attn_qkv_weight.bin", dtype=np.float32).reshape(1024, 6144)
attn_gate = np.fromfile("blk_0_attn_gate_weight.bin", dtype=np.float32).reshape(1024, 2048)
ssm_conv1d = np.fromfile("blk_0_ssm_conv1d_weight.bin", dtype=np.float32).reshape(4, 6144)
ssm_alpha = np.fromfile("blk_0_ssm_alpha_weight.bin", dtype=np.float32).reshape(1024, 16)
ssm_beta = np.fromfile("blk_0_ssm_beta_weight.bin", dtype=np.float32).reshape(1024, 16)
ssm_dt_bias = np.fromfile("blk_0_ssm_dt_bias.bin", dtype=np.float32)
ssm_a = np.fromfile("blk_0_ssm_a.bin", dtype=np.float32)
ssm_norm = np.fromfile("blk_0_ssm_norm_weight.bin", dtype=np.float32)
ssm_out = np.fromfile("blk_0_ssm_out_weight.bin", dtype=np.float32).reshape(2048, 1024)

print(f"\nLoaded weights:")
print(f"  attn_qkv: {attn_qkv.shape}")
print(f"  attn_gate: {attn_gate.shape}")
print(f"  ssm_conv1d: {ssm_conv1d.shape}")
print(f"  ssm_alpha: {ssm_alpha.shape}")
print(f"  ssm_beta: {ssm_beta.shape}")
print(f"  ssm_dt_bias: {ssm_dt_bias.shape}")
print(f"  ssm_a: {ssm_a.shape}")
print(f"  ssm_norm: {ssm_norm.shape}")
print(f"  ssm_out: {ssm_out.shape}")

# Step 1: QKV projection
qkv_proj = pre_norm @ attn_qkv
compare("qkv_proj", qkv_proj, "native_l0_qkv_proj.bin", (4, 6144))

# Step 2: Causal Conv1d + SiLU
def causal_conv1d(x, weight):
    seq_len, channels = x.shape
    kernel_size = weight.shape[0]
    out = np.zeros_like(x)
    for c in range(channels):
        for t in range(seq_len):
            for k in range(min(kernel_size, t + 1)):
                w_idx = kernel_size - 1 - k
                out[t, c] += x[t - k, c] * weight[w_idx, c]
    return out

conv_out = causal_conv1d(qkv_proj, ssm_conv1d)
conv_out = silu(conv_out)
compare("conv_out", conv_out, "native_l0_conv_out.bin", (4, 6144))

# Step 3: Split Q, K, V
seq_len = 4
num_qk_heads = 16
num_v_heads = 16
head_k_dim = 128
head_v_dim = 128
q_total = num_qk_heads * head_k_dim
k_total = num_qk_heads * head_k_dim
v_total = num_v_heads * head_v_dim

q = conv_out[:, :q_total].reshape(seq_len, num_qk_heads, head_k_dim)
k = conv_out[:, q_total:q_total+k_total].reshape(seq_len, num_qk_heads, head_k_dim)
v = conv_out[:, q_total+k_total:].reshape(seq_len, num_v_heads, head_v_dim)

# Step 4: L2 normalize Q, K
q = l2_normalize_per_head(q)
k = l2_normalize_per_head(k)
compare("q_norm", q, "native_l0_q_norm.bin", (4, 16, 128))
compare("k_norm", k, "native_l0_k_norm.bin", (4, 16, 128))
compare("v", v, "native_l0_v.bin", (4, 16, 128))

# Step 5: Decay rates
alpha = pre_norm @ ssm_alpha  # [seq, num_v_heads]
decay = np.zeros((seq_len, num_v_heads))
for s in range(seq_len):
    for h in range(num_v_heads):
        dt = np.log(1 + np.exp(alpha[s, h] + ssm_dt_bias[h]))
        decay[s, h] = np.exp(dt * ssm_a[h])

compare("decay", decay, "native_l0_decay.bin", (4, 16))

# Step 6: Beta gates
beta_logits = pre_norm @ ssm_beta
beta = sigmoid(beta_logits)
compare("beta", beta, "native_l0_beta.bin", (4, 16))

# Step 7: Delta rule
output_dim = num_v_heads * head_v_dim
output = np.zeros((seq_len, output_dim))
state = np.zeros((num_v_heads, head_k_dim, head_v_dim))
scale = 1.0 / np.sqrt(head_k_dim)

for s in range(seq_len):
    for h_qk in range(num_qk_heads):
        q_h = q[s, h_qk]
        k_h = k[s, h_qk]
        h_v = h_qk  # 1:1 mapping
        
        decay_h = decay[s, h_v]
        beta_h = beta[s, h_v]
        v_h = v[s, h_v]
        
        # v_pred = state @ k
        v_pred = state[h_v] @ k_h
        
        # delta = (v - v_pred) * beta
        delta = (v_h - v_pred) * beta_h
        
        # state = decay * state + outer(k, delta)
        state[h_v] = decay_h * state[h_v] + np.outer(k_h, delta)
        
        # output = state @ q * scale
        out_base = h_v * head_v_dim
        output[s, out_base:out_base+head_v_dim] = state[h_v] @ q_h * scale

compare("core_attn_out", output, "native_l0_core_attn_out.bin", (4, 2048))

# Step 8: Per-head RMSNorm
output_reshaped = output.reshape(seq_len, num_v_heads, head_v_dim)
for s in range(seq_len):
    for h in range(num_v_heads):
        x = output_reshaped[s, h]
        scale_norm = 1.0 / np.sqrt(np.mean(x**2) + 1e-5)
        output_reshaped[s, h] = x * scale_norm * ssm_norm
output = output_reshaped.reshape(seq_len, output_dim)
compare("post_norm", output, "native_l0_post_norm.bin", (4, 2048))

# Step 9: Gate
gate = silu(pre_norm @ attn_gate)
output = output * gate
compare("post_gate", output, "native_l0_post_gate.bin", (4, 2048))

# Step 10: Output projection
ssm_out_proj = output @ ssm_out
compare("ssm_out", ssm_out_proj, "native_l0_ssm_out.bin", (4, 1024))
