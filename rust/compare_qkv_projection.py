#!/usr/bin/env python3
"""
Compare Qwen3.5 QKV weights and projection between HF and GGUF.
Run: python3 compare_qkv_projection.py /path/to/Qwen3.5-0.8B-Q4_0.gguf
"""
import sys
import struct
import numpy as np
import torch
from transformers import AutoModelForCausalLM

GGUF_PATH = sys.argv[1] if len(sys.argv) > 1 else "../models/Qwen3.5-0.8B-Q4_0.gguf"
MODEL_ID = "../models/Qwen3.5-0.8B-HF"
LAYER = 0

print(f"=== Loading HF model from {MODEL_ID} ===")
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map="cpu", local_files_only=True
)

hf_layer = hf_model.model.layers[LAYER]
print(f"\nHF layer {LAYER} module: {type(hf_layer.linear_attn).__name__}")

# HF DeltaNet uses in_proj_qkv
hf_w = hf_layer.linear_attn.in_proj_qkv.weight.detach().cpu().float().numpy()
print(f"HF in_proj_qkv.weight shape: {hf_w.shape}, mean={hf_w.mean():.6f}, abs_mean={np.abs(hf_w).mean():.6f}")

# Also get the gate projection (z)
hf_z = hf_layer.linear_attn.in_proj_z.weight.detach().cpu().float().numpy()
print(f"HF in_proj_z.weight shape: {hf_z.shape}, mean={hf_z.mean():.6f}, abs_mean={np.abs(hf_z).mean():.6f}")


def read_gguf_tensor(path, tensor_name):
    """Minimal GGUF tensor reader using gguf-dump or direct parsing."""
    # Try using the gguf python package if available
    try:
        from gguf import GGUFReader
        reader = GGUFReader(path)
        for tensor in reader.tensors:
            if tensor.name == tensor_name:
                data = reader.get_tensor(tensor.name)
                return data.astype(np.float32), list(data.shape)
    except ImportError:
        pass
    
    # Fallback: parse gguf-dump hex output (slow but works)
    import subprocess
    result = subprocess.run(
        ["gguf-dump", path, tensor_name],
        capture_output=True, text=True
    )
    print(f"gguf-dump fallback not implemented, skipping tensor {tensor_name}")
    return None, None


print(f"\n=== Loading GGUF {GGUF_PATH} ===")

# Load dequantized weight from Rust dump
# The Rust dump is [1024, 6144] in native GGUF layout
# But Tensor::matmul in Rust expects [k, n] = [1024, 6144] for x @ W
# PyTorch Linear stores [out, in] = [6144, 1024]
gguf_qkv_f32 = np.fromfile("gguf_qkv_layer0.bin", dtype=np.float32).reshape(1024, 6144)
print(f"GGUF attn_qkv.weight shape (raw): {gguf_qkv_f32.shape}")

# The GGUF stores [in=1024, out=6144] which is the transpose of PyTorch's [out=6144, in=1024]
# Our Rust matmul does [seq, hidden] @ [hidden, conv_dim] which matches this layout
# So for comparison with PyTorch's x @ W^T, we need to transpose
gguf_qkv_for_compare = gguf_qkv_f32.T  # Now [6144, 1024] to match PyTorch
print(f"GGUF shape transposed to match PyTorch: {gguf_qkv_for_compare.shape}")

print(f"\n=== Weight Comparison ===")
print(f"HF shape:   {hf_w.shape}")
print(f"GGUF shape: {gguf_qkv_f32.shape}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# GGUF may store transposed
candidates = [("GGUF raw [1024,6144]", gguf_qkv_f32), ("GGUF transposed [6144,1024]", gguf_qkv_for_compare)]

best_match = None
best_cos = -1.0

for label, gguf_w in candidates:
    if hf_w.shape != gguf_w.shape:
        print(f"  {label}: shape mismatch {gguf_w.shape}, skipping")
        continue
    
    diff = hf_w - gguf_w
    cos = cosine_similarity(hf_w.flatten(), gguf_w.flatten())
    abs_mean_diff = np.abs(diff).mean()
    
    print(f"\n  {label}:")
    print(f"    GGUF abs_mean: {np.abs(gguf_w).mean():.6f}")
    print(f"    Weight diff abs_mean: {abs_mean_diff:.6f}")
    print(f"    Weight diff max: {np.abs(diff).max():.6f}")
    print(f"    Weight cos_sim: {cos:.6f}")
    
    if cos > best_cos:
        best_cos = cos
        best_match = (label, gguf_w)

if best_match is None:
    print("\nNo matching shape found!")
    sys.exit(1)

label, gguf_w = best_match

# === Projection Test ===
print(f"\n=== Projection Test (using {label} orientation) ===")
test_input = np.random.randn(1, hf_w.shape[1]).astype(np.float32) * 0.1

# HF projection: PyTorch Linear does x @ W^T where W is [out, in]
# So hf_w shape is [out=6144, in=1024]
hf_proj = test_input @ hf_w.T

# Our Rust matmul does: [seq, hidden] @ [hidden, conv_dim] = [seq, conv_dim]
# If GGUF weight is [hidden, conv_dim], we do test_input @ gguf_w
# If GGUF weight is [conv_dim, hidden] (transposed), we do test_input @ gguf_w.T
if label == "as-is":
    gguf_proj = test_input @ gguf_w
    gguf_proj_T = test_input @ gguf_w.T
else:
    gguf_proj = test_input @ gguf_w
    gguf_proj_T = test_input @ gguf_w.T

print(f"HF proj shape:        {hf_proj.shape}")
print(f"GGUF proj shape:      {gguf_proj.shape}")
print(f"GGUF proj^T shape:    {gguf_proj_T.shape}")

if hf_proj.shape == gguf_proj.shape:
    diff = hf_proj - gguf_proj
    print(f"\nProjection (GGUF as-is):")
    print(f"  diff abs_mean: {np.abs(diff).mean():.6f}")
    print(f"  diff max: {np.abs(diff).max():.6f}")
    print(f"  cos_sim: {cosine_similarity(hf_proj.flatten(), gguf_proj.flatten()):.6f}")

if hf_proj.shape == gguf_proj_T.shape:
    diff = hf_proj - gguf_proj_T
    print(f"\nProjection (GGUF transposed):")
    print(f"  diff abs_mean: {np.abs(diff).mean():.6f}")
    print(f"  diff max: {np.abs(diff).max():.6f}")
    print(f"  cos_sim: {cosine_similarity(hf_proj.flatten(), gguf_proj_T.flatten()):.6f}")

# Also test with the real embedding pre-norm input
print(f"\n=== Real Input Projection ===")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
input_ids = torch.tensor([[17, 10, 17, 28]])
hidden = hf_model.model.embed_tokens(input_ids)
normed = hf_layer.input_layernorm(hidden)

# Use first token's pre-norm hidden state
real_input = normed[0, 0].detach().cpu().numpy().reshape(1, -1)
hf_real_proj = real_input @ hf_w.T

if label == "as-is":
    gguf_real_proj = real_input @ gguf_w
    gguf_real_proj_T = real_input @ gguf_w.T
else:
    gguf_real_proj = real_input @ gguf_w
    gguf_real_proj_T = real_input @ gguf_w.T

print(f"Real input shape: {real_input.shape}")
print(f"HF real proj shape: {hf_real_proj.shape}")

if hf_real_proj.shape == gguf_real_proj.shape:
    print(f"\nReal projection (GGUF as-is):")
    print(f"  HF abs_mean: {np.abs(hf_real_proj).mean():.6f}")
    print(f"  GGUF abs_mean: {np.abs(gguf_real_proj).mean():.6f}")
    print(f"  diff abs_mean: {np.abs(hf_real_proj - gguf_real_proj).mean():.6f}")
    print(f"  cos_sim: {cosine_similarity(hf_real_proj.flatten(), gguf_real_proj.flatten()):.6f}")

if hf_real_proj.shape == gguf_real_proj_T.shape:
    print(f"\nReal projection (GGUF transposed):")
    print(f"  HF abs_mean: {np.abs(hf_real_proj).mean():.6f}")
    print(f"  GGUF abs_mean: {np.abs(gguf_real_proj_T).mean():.6f}")
    print(f"  diff abs_mean: {np.abs(hf_real_proj - gguf_real_proj_T).mean():.6f}")
    print(f"  cos_sim: {cosine_similarity(hf_real_proj.flatten(), gguf_real_proj_T.flatten()):.6f}")
