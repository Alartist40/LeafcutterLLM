#!/usr/bin/env python3
"""Compare HF vs GGUF QKV weights and projection."""

import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load HF model
print("Loading HF model...")
model = AutoModelForCausalLM.from_pretrained(
    '../models/Qwen3.5-0.8B-HF',
    local_files_only=True,
    torch_dtype=torch.float32,
    device_map='cpu',
)
layer0 = model.model.layers[0].linear_attn

# Get HF weight
hf_w = layer0.in_proj_qkv.weight.detach().cpu().float().numpy()
print(f"HF in_proj_qkv weight: shape={hf_w.shape}, mean={hf_w.mean():.6f}, std={hf_w.std():.6f}")

# Load GGUF weight using gguf-dump or direct reading
# Since gguf python package isn't available, let's use the Rust binary or just read directly
# Actually, let's use the leafcutter library if available... 
# Simpler: use numpy memmap or a small Rust program to dump the weight

print("\nTo compare GGUF weight, run the Rust comparison binary.")
print("For now, let's compare the projection output for a random input.")

# Create a random input matching the prompt embedding
hidden = model.model.embed_tokens(torch.tensor([[17, 10, 17, 28]]))
# Apply input_layernorm
normed = model.model.layers[0].input_layernorm(hidden)

# HF projection
hf_qkv = layer0.in_proj_qkv(normed).detach().cpu().float().numpy()
print(f"HF qkv_proj: shape={hf_qkv.shape}, mean={hf_qkv.mean():.6f}, std={hf_qkv.std():.6f}, abs_mean={np.abs(hf_qkv).mean():.6f}")

# Save normed hidden state for native comparison
np.save("hf_pre_norm_layer0.npy", normed.detach().cpu().float().numpy())
print("Saved hf_pre_norm_layer0.npy")
