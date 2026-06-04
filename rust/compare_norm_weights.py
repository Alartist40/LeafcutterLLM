#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/xander/Documents/portfolio/LeafcutterLLM/.venv/lib/python3.14/site-packages")
import torch
import numpy as np
from transformers import AutoModelForCausalLM

MODEL_PATH = "/home/xander/Documents/portfolio/LeafcutterLLM/models/Qwen3.5-0.8B-HF"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)

# Check layer norm weights
for i in range(min(4, len(model.model.layers))):
    layer = model.model.layers[i]
    pre_norm = layer.input_layernorm.weight.detach().to(torch.float32).numpy()
    post_norm = layer.post_attention_layernorm.weight.detach().to(torch.float32).numpy()
    print(f"Layer {i}: pre_norm mean={pre_norm.mean():.4f} std={pre_norm.std():.4f} range=[{pre_norm.min():.4f}, {pre_norm.max():.4f}]")
    print(f"Layer {i}: post_norm mean={post_norm.mean():.4f} std={post_norm.std():.4f} range=[{post_norm.min():.4f}, {post_norm.max():.4f}]")

final_norm = model.model.norm.weight.detach().to(torch.float32).numpy()
print(f"Final norm: mean={final_norm.mean():.4f} std={final_norm.std():.4f} range=[{final_norm.min():.4f}, {final_norm.max():.4f}]")

# DeltaNet norm
delta = model.model.layers[0].linear_attn
ssm_norm = delta.norm.weight.detach().to(torch.float32).numpy()
print(f"DeltaNet ssm_norm: mean={ssm_norm.mean():.4f} std={ssm_norm.std():.4f} range=[{ssm_norm.min():.4f}, {ssm_norm.max():.4f}]")

print("\nIf native uses 'weight' instead of '1.0 + weight' for Qwen3_5RMSNorm:")
print("The relative error for final_norm would be ~{:.1f}%".format(100 * abs(final_norm.mean() - (1.0 + final_norm.mean())) / (1.0 + final_norm.mean())))
