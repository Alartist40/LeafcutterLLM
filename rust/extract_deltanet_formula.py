#!/usr/bin/env python3
import torch, sys, inspect
sys.path.insert(0, "../models/Qwen3.5-0.8B-HF")

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "../models/Qwen3.5-0.8B-HF",
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)

layer = model.model.layers[0]
print("Layer type:", type(layer).__name__)
print("\n=== FORWARD SOURCE ===")
print(inspect.getsource(layer.forward))

print("\n=== DELTANET CONFIG ===")
cfg = model.config
for key in sorted(dir(cfg)):
    val = getattr(cfg, key, 'N/A')
    if any(s in key.lower() for s in ['ssm','delta','gate','conv','head','dim']):
        if not key.startswith('_'):
            print(f"{key}: {val}")
