#!/usr/bin/env python3
import numpy as np
import sys

def load_native(path, dim=1024):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    # Last token only
    return data[-dim:]

def load_hf(path):
    return np.load(path)

print(f"{'Layer':>6} {'CosSim':>10} {'MSE':>12} {'HF_norm':>10} {'Nat_norm':>10} {'NormRatio':>10}")
print("-" * 70)

for i in range(26):
    hf_path = f"hf_layer_{i:02d}.npy"
    nat_path = f"native_layer_{i:02d}.bin"
    
    try:
        hf = load_hf(hf_path)
        nat = load_native(nat_path)
    except FileNotFoundError as e:
        print(f"{i:6} MISSING: {e}")
        continue
    
    # Cosine similarity
    dot = np.dot(hf, nat)
    hf_norm = np.linalg.norm(hf)
    nat_norm = np.linalg.norm(nat)
    cos_sim = dot / (hf_norm * nat_norm) if hf_norm > 0 and nat_norm > 0 else 0
    
    # MSE
    mse = np.mean((hf - nat) ** 2)
    
    # Norm ratio
    norm_ratio = nat_norm / hf_norm if hf_norm > 0 else 0
    
    status = "OK" if cos_sim > 0.99 else ("WARN" if cos_sim > 0.95 else "DIVERGED")
    if i == 0:
        status = "EMBED"
    
    print(f"{i:6} {cos_sim:10.6f} {mse:12.6f} {hf_norm:10.4f} {nat_norm:10.4f} {norm_ratio:10.4f} {status}")
