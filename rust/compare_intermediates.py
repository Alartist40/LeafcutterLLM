#!/usr/bin/env python3
"""Compare HF vs Native intermediates for DeltaNet Layer 1."""

import os
import sys
import numpy as np

OUT_DIR = "debug_hf_intermediates"
NATIVE_DIR = "."

# Mapping: HF name -> native name -> shape info
# HF shapes are (batch, seq, ...) or (batch, channels, seq)
# Native shapes are (seq, ...)

comparisons = [
    ("hf_l1_qkv_proj.npy", "native_l1_qkv_proj.bin", [4, 6144], "qkv_proj"),
    ("hf_l1_conv_out.npy", "native_l1_conv_out.bin", [4, 6144], "conv_out"),
    ("hf_l1_conv_out_silu.npy", "native_l1_conv_out.bin", [4, 6144], "conv_out_silu"),  # native already has SiLU
    ("hf_l1_q_norm.npy", "native_l1_q_norm.bin", [4, 16, 128], "q_norm"),
    ("hf_l1_k_norm.npy", "native_l1_k_norm.bin", [4, 16, 128], "k_norm"),
    ("hf_l1_v.npy", "native_l1_v.bin", [4, 16, 128], "v"),
    ("hf_l1_decay.npy", "native_l1_decay.bin", [4, 16], "decay"),
    ("hf_l1_beta.npy", "native_l1_beta.bin", [4, 16], "beta"),
    ("hf_l1_core_attn_out.npy", "native_l1_core_attn_out.bin", [4, 16, 128], "core_attn_out"),
    ("hf_l1_post_gate.npy", "native_l1_post_gate.bin", [4, 2048], "post_gate"),
    ("hf_l1_ssm_out.npy", "native_l1_ssm_out.bin", [4, 1024], "ssm_out"),
]

print(f"{'Tensor':20} {'CosSim':>10} {'MSE':>12} {'HF_abs_mean':>12} {'Nat_abs_mean':>12} {'Ratio':>8}")
print("-" * 80)

for hf_file, native_file, shape, label in comparisons:
    hf_path = os.path.join(OUT_DIR, hf_file)
    native_path = os.path.join(NATIVE_DIR, native_file)
    
    if not os.path.exists(hf_path):
        print(f"{label:20} MISSING HF: {hf_path}")
        continue
    if not os.path.exists(native_path):
        print(f"{label:20} MISSING NATIVE: {native_path}")
        continue
    
    hf = np.load(hf_path).astype(np.float32)
    native = np.fromfile(native_path, dtype=np.float32)
    
    # Reshape to match
    hf_flat = hf.reshape(-1)
    native = native.reshape(hf_flat.shape)
    
    # Cosine similarity
    dot = np.dot(hf_flat, native)
    norm_hf = np.linalg.norm(hf_flat)
    norm_nat = np.linalg.norm(native)
    cossim = dot / (norm_hf * norm_nat + 1e-12)
    
    mse = np.mean((hf_flat - native) ** 2)
    hf_abs_mean = np.abs(hf_flat).mean()
    nat_abs_mean = np.abs(native).mean()
    ratio = nat_abs_mean / (hf_abs_mean + 1e-12)
    
    print(f"{label:20} {cossim:10.6f} {mse:12.6e} {hf_abs_mean:12.6f} {nat_abs_mean:12.6f} {ratio:8.3f}")
