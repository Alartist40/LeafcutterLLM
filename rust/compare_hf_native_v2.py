#!/usr/bin/env python3
"""Compare HF vs Native DeltaNet intermediates for layer 0 (both seq_len=4)."""
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

print("=== Layer 0 HF vs Native Comparison (seq_len=4) ===\n")

# Both HF and native now have seq_len=4
compare("qkv_proj",     "hf_l0_qkv_proj.bin",      "native_l0_qkv_proj_s4.bin",      (4, 6144), (4, 6144))
compare("conv_out",     "hf_l0_conv_out.bin",      "native_l0_conv_out_s4.bin",      (4, 6144), (4, 6144))
compare("q_norm",       "hf_l0_q_norm.bin",        "native_l0_q_norm_s4.bin",        (4, 16, 128), (4, 16, 128))
compare("k_norm",       "hf_l0_k_norm.bin",        "native_l0_k_norm_s4.bin",        (4, 16, 128), (4, 16, 128))
compare("v",            "hf_l0_v.bin",             "native_l0_v_s4.bin",             (4, 16, 128), (4, 16, 128))
compare("decay",        "hf_l0_decay.bin",         "native_l0_decay_s4.bin",         (4, 16), (4, 16))
compare("beta",         "hf_l0_beta.bin",          "native_l0_beta_s4.bin",          (4, 16), (4, 16))
compare("core_attn",    "hf_l0_core_attn_out.bin", "native_l0_core_attn_out_s4.bin", (4, 2048), (4, 2048))
compare("post_norm",    "hf_l0_post_norm.bin",     "native_l0_post_norm_s4.bin",     (4, 2048), (4, 2048))
compare("ssm_out",      "hf_l0_ssm_out.bin",       "native_l0_ssm_out_s4.bin",       (4, 1024), (4, 1024))

print("\n=== Per-token core_attn comparison ===")
hf_core = np.fromfile("hf_l0_core_attn_out.bin", dtype=np.float32).reshape(4, 2048)
native_core = np.fromfile("native_l0_core_attn_out_s4.bin", dtype=np.float32).reshape(4, 2048)
for pos in range(4):
    h = hf_core[pos, :]
    n = native_core[pos, :]
    mae = np.abs(h - n).mean()
    cos = np.dot(h, n) / (np.linalg.norm(h) * np.linalg.norm(n) + 1e-12)
    print(f"  pos {pos}: MAE={mae:.6f} CosSim={cos:.6f}")
