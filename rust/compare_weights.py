#!/usr/bin/env python3
"""Compare HF model weights with native dumped weights."""
import sys
sys.path.insert(0, "/home/xander/Documents/portfolio/LeafcutterLLM/.venv/lib/python3.14/site-packages")

import torch
import numpy as np
from transformers import AutoModelForCausalLM

MODEL_PATH = "/home/xander/Documents/portfolio/LeafcutterLLM/models/Qwen3.5-0.8B-HF"

def main():
    print("Loading HF model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    delta = model.model.layers[0].linear_attn
    
    # Dump HF weights
    weights = {
        "attn_qkv": delta.in_proj_qkv.weight.detach().to(torch.float32).numpy(),
        "attn_gate": delta.in_proj_z.weight.detach().to(torch.float32).numpy(),
        "ssm_conv1d": delta.conv1d.weight.squeeze(1).detach().to(torch.float32).numpy(),
        "ssm_alpha": delta.in_proj_a.weight.detach().to(torch.float32).numpy(),
        "ssm_beta": delta.in_proj_b.weight.detach().to(torch.float32).numpy(),
        "ssm_dt_bias": delta.dt_bias.detach().to(torch.float32).numpy(),
        "ssm_a_log": delta.A_log.detach().to(torch.float32).numpy(),
        "ssm_norm": delta.norm.weight.detach().to(torch.float32).numpy(),
        "ssm_out": delta.out_proj.weight.detach().to(torch.float32).numpy(),
    }
    
    # Load native dumped weights
    native_files = {
        "attn_qkv": "blk_0_attn_qkv_weight.bin",
        "attn_gate": "blk_0_attn_gate_weight.bin",
        "ssm_conv1d": "blk_0_ssm_conv1d_weight.bin",
        "ssm_alpha": "blk_0_ssm_alpha_weight.bin",
        "ssm_beta": "blk_0_ssm_beta_weight.bin",
        "ssm_dt_bias": "blk_0_ssm_dt_bias.bin",
        "ssm_a": "blk_0_ssm_a.bin",
        "ssm_norm": "blk_0_ssm_norm_weight.bin",
        "ssm_out": "blk_0_ssm_out_weight.bin",
    }
    
    native = {}
    for name, fname in native_files.items():
        try:
            data = np.fromfile(fname, dtype=np.float32)
            native[name] = data
        except FileNotFoundError as e:
            print(f"  Native {name} not found: {e.filename}")
    
    print("\n=== Weight Comparison ===")
    for name, hf_w in weights.items():
        nat_name = name if name != "ssm_a_log" else "ssm_a"
        if nat_name not in native:
            continue
        
        nat_data = native[nat_name]
        
        # Try all combinations: reshape to hf_shape or transposed, and compare with hf_w or hf_w.T
        best_mae = float('inf')
        best_cos = -1
        best_desc = ""
        
        candidates = []
        if len(hf_w.shape) == 1:
            candidates.append((nat_data.reshape(hf_w.shape), "same shape"))
        elif len(hf_w.shape) == 2:
            for shape in [hf_w.shape, (hf_w.shape[1], hf_w.shape[0])]:
                if np.prod(shape) == nat_data.size:
                    nat_r = nat_data.reshape(shape)
                    candidates.append((nat_r, f"reshape{shape}"))
                    candidates.append((nat_r.T, f"reshape{shape}.T"))
        
        for nat_r, desc in candidates:
            if nat_r.shape != hf_w.shape:
                continue
            mae = np.abs(hf_w - nat_r).mean()
            cos = np.dot(hf_w.flatten(), nat_r.flatten()) / (np.linalg.norm(hf_w) * np.linalg.norm(nat_r) + 1e-12)
            if cos > best_cos:
                best_cos = cos
                best_mae = mae
                best_desc = desc
        
        print(f"  {name:15s}: HF shape={hf_w.shape} best={best_desc} MAE={best_mae:.6f} CosSim={best_cos:.6f}")
        
        # Special check for ssm_a: compare -exp(A_log) vs native ssm_a
        if name == "ssm_a_log":
            hf_a = -np.exp(hf_w)
            nat_r = native[nat_name].reshape(hf_a.shape)
            mae_a = np.abs(hf_a - nat_r).mean()
            cos_a = np.dot(hf_a.flatten(), nat_r.flatten()) / (np.linalg.norm(hf_a) * np.linalg.norm(nat_r) + 1e-12)
            print(f"    -> -exp(A_log): MAE={mae_a:.6f} CosSim={cos_a:.6f}")

if __name__ == "__main__":
    main()
