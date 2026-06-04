#!/usr/bin/env python3
"""Compare HF vs Native DeltaNet intermediates for layer 0."""
import numpy as np

def compare(name, hf_file, native_file, hf_shape, native_shape):
    try:
        hf = np.fromfile(hf_file, dtype=np.float32).reshape(hf_shape)
        native = np.fromfile(native_file, dtype=np.float32).reshape(native_shape)
    except FileNotFoundError as e:
        print(f"  {name}: MISSING {e.filename}")
        return
    
    if hf.shape != native.shape:
        print(f"  {name}: SHAPE MISMATCH hf={hf.shape} native={native.shape}")
        return
    
    mae = np.abs(hf - native).mean()
    max_diff = np.abs(hf - native).max()
    cos_sim = np.dot(hf.flatten(), native.flatten()) / (np.linalg.norm(hf) * np.linalg.norm(native) + 1e-12)
    hf_mean = np.abs(hf).mean()
    native_mean = np.abs(native).mean()
    
    print(f"  {name:20s}: MAE={mae:.6f} max_diff={max_diff:.4f} CosSim={cos_sim:.6f} |HF|={hf_mean:.4f} |Nat|={native_mean:.4f}")

print("=== Layer 0 HF vs Native Comparison ===")
print("\nNote: Native dumps are for seq_len=5 (includes BOS + '2+2=')")
print("HF dumps are for seq_len=4 ('2+2=' tokens only, no BOS)")
print("Comparing last 4 positions of native with HF...\n")

# HF shape: [4, ...] (seq_len=4)
# Native shape: [5, ...] (seq_len=5, includes BOS)
# We compare HF[0:4] with Native[1:5] (skip BOS)

# For 2D tensors: native is [5, 6144], hf is [4, 6144]
# We load native and slice [1:5, :]

native_qkv = np.fromfile("native_l0_qkv_proj.bin", dtype=np.float32).reshape(5, 6144)
hf_qkv = np.fromfile("hf_l0_qkv_proj.bin", dtype=np.float32).reshape(4, 6144)
mae = np.abs(hf_qkv - native_qkv[1:5, :]).mean()
cos = np.dot(hf_qkv.flatten(), native_qkv[1:5, :].flatten()) / (np.linalg.norm(hf_qkv) * np.linalg.norm(native_qkv[1:5, :]) + 1e-12)
print(f"  qkv_proj: MAE={mae:.6f} CosSim={cos:.6f}")

native_conv = np.fromfile("native_l0_conv_out.bin", dtype=np.float32).reshape(5, 6144)
hf_conv = np.fromfile("hf_l0_conv_out.bin", dtype=np.float32).reshape(4, 6144)
mae = np.abs(hf_conv - native_conv[1:5, :]).mean()
cos = np.dot(hf_conv.flatten(), native_conv[1:5, :].flatten()) / (np.linalg.norm(hf_conv) * np.linalg.norm(native_conv[1:5, :]) + 1e-12)
print(f"  conv_out: MAE={mae:.6f} CosSim={cos:.6f}")

# Q, K, V are [seq_len, num_heads, head_dim]
native_q = np.fromfile("native_l0_q_norm.bin", dtype=np.float32).reshape(5, 16, 128)
hf_q = np.fromfile("hf_l0_q_norm.bin", dtype=np.float32).reshape(4, 16, 128)
mae = np.abs(hf_q - native_q[1:5, :, :]).mean()
cos = np.dot(hf_q.flatten(), native_q[1:5, :, :].flatten()) / (np.linalg.norm(hf_q) * np.linalg.norm(native_q[1:5, :, :]) + 1e-12)
print(f"  q_norm:   MAE={mae:.6f} CosSim={cos:.6f}")

native_k = np.fromfile("native_l0_k_norm.bin", dtype=np.float32).reshape(5, 16, 128)
hf_k = np.fromfile("hf_l0_k_norm.bin", dtype=np.float32).reshape(4, 16, 128)
mae = np.abs(hf_k - native_k[1:5, :, :]).mean()
cos = np.dot(hf_k.flatten(), native_k[1:5, :, :].flatten()) / (np.linalg.norm(hf_k) * np.linalg.norm(native_k[1:5, :, :]) + 1e-12)
print(f"  k_norm:   MAE={mae:.6f} CosSim={cos:.6f}")

native_v = np.fromfile("native_l0_v.bin", dtype=np.float32).reshape(5, 16, 128)
hf_v = np.fromfile("hf_l0_v.bin", dtype=np.float32).reshape(4, 16, 128)
mae = np.abs(hf_v - native_v[1:5, :, :]).mean()
cos = np.dot(hf_v.flatten(), native_v[1:5, :, :].flatten()) / (np.linalg.norm(hf_v) * np.linalg.norm(native_v[1:5, :, :]) + 1e-12)
print(f"  v:        MAE={mae:.6f} CosSim={cos:.6f}")

# Decay and beta are [seq_len, num_heads]
native_decay = np.fromfile("native_l0_decay.bin", dtype=np.float32).reshape(5, 16)
hf_decay = np.fromfile("hf_l0_decay.bin", dtype=np.float32).reshape(4, 16)
mae = np.abs(hf_decay - native_decay[1:5, :]).mean()
cos = np.dot(hf_decay.flatten(), native_decay[1:5, :].flatten()) / (np.linalg.norm(hf_decay) * np.linalg.norm(native_decay[1:5, :]) + 1e-12)
print(f"  decay:    MAE={mae:.6f} CosSim={cos:.6f}")

native_beta = np.fromfile("native_l0_beta.bin", dtype=np.float32).reshape(5, 16)
hf_beta = np.fromfile("hf_l0_beta.bin", dtype=np.float32).reshape(4, 16)
mae = np.abs(hf_beta - native_beta[1:5, :]).mean()
cos = np.dot(hf_beta.flatten(), native_beta[1:5, :].flatten()) / (np.linalg.norm(hf_beta) * np.linalg.norm(native_beta[1:5, :]) + 1e-12)
print(f"  beta:     MAE={mae:.6f} CosSim={cos:.6f}")

# core_attn_out is [seq_len, num_heads * head_dim]
native_core = np.fromfile("native_l0_core_attn_out.bin", dtype=np.float32).reshape(5, 2048)
hf_core = np.fromfile("hf_l0_core_attn_out.bin", dtype=np.float32).reshape(4, 2048)
mae = np.abs(hf_core - native_core[1:5, :]).mean()
cos = np.dot(hf_core.flatten(), native_core[1:5, :].flatten()) / (np.linalg.norm(hf_core) * np.linalg.norm(native_core[1:5, :]) + 1e-12)
print(f"  core_attn:MAE={mae:.6f} CosSim={cos:.6f}")

# post_norm is [seq_len, num_heads * head_dim]
native_norm = np.fromfile("native_l0_post_norm.bin", dtype=np.float32).reshape(5, 2048)
hf_norm = np.fromfile("hf_l0_post_norm.bin", dtype=np.float32).reshape(4, 2048)
mae = np.abs(hf_norm - native_norm[1:5, :]).mean()
cos = np.dot(hf_norm.flatten(), native_norm[1:5, :].flatten()) / (np.linalg.norm(hf_norm) * np.linalg.norm(native_norm[1:5, :]) + 1e-12)
print(f"  post_norm:MAE={mae:.6f} CosSim={cos:.6f}")

# ssm_out is [seq_len, hidden_size]
native_out = np.fromfile("native_l0_ssm_out.bin", dtype=np.float32).reshape(5, 1024)
hf_out = np.fromfile("hf_l0_ssm_out.bin", dtype=np.float32).reshape(4, 1024)
mae = np.abs(hf_out - native_out[1:5, :]).mean()
cos = np.dot(hf_out.flatten(), native_out[1:5, :].flatten()) / (np.linalg.norm(hf_out) * np.linalg.norm(native_out[1:5, :]) + 1e-12)
print(f"  ssm_out:  MAE={mae:.6f} CosSim={cos:.6f}")

print("\n=== Individual Token Comparison (last token = position 3, '=') ===")
for pos in range(4):
    native_pos = pos + 1  # skip BOS
    hf_pos = pos
    
    n_core = native_core[native_pos, :]
    h_core = hf_core[hf_pos, :]
    mae = np.abs(h_core - n_core).mean()
    cos = np.dot(h_core, n_core) / (np.linalg.norm(h_core) * np.linalg.norm(n_core) + 1e-12)
    print(f"  pos {pos} core_attn: MAE={mae:.6f} CosSim={cos:.6f}")
