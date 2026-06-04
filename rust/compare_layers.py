#!/usr/bin/env python3
import sys

def parse_layer_file(path):
    layers = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("LAYER"):
                continue
            parts = line.split("|")
            idx = int(parts[0].split()[1])
            stats = {}
            for p in parts[1:]:
                k, v = p.strip().split("=")
                stats[k] = float(v)
            layers[idx] = stats
    return layers

hf = parse_layer_file("hf_layers.txt")
native = parse_layer_file("native_layers.txt")

print(f"{'Layer':>6} {'HF_abs_mean':>12} {'Nat_abs_mean':>12} {'Ratio':>8} {'Status'}")
print("-" * 60)

divergence_layer = None
for i in sorted(hf.keys()):
    if i not in native:
        print(f"{i:6} MISSING IN NATIVE")
        continue
    
    hf_am = hf[i]['abs_mean']
    nat_am = native[i]['abs_mean']
    ratio = nat_am / hf_am if hf_am != 0 else 0
    
    status = "OK" if 0.7 < ratio < 1.3 else "DIVERGED"
    if status == "DIVERGED" and divergence_layer is None:
        divergence_layer = i
        status = "DIVERGED <<< FIRST"
    
    print(f"{i:6} {hf_am:12.6f} {nat_am:12.6f} {ratio:8.3f} {status}")

if divergence_layer is not None:
    print(f"\nFirst divergence at layer {divergence_layer}")
else:
    print(f"\nNo significant divergence detected")
