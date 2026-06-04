#!/usr/bin/env python3
"""Debug autoregressive generation on HF Qwen3.5-0.8B."""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

MODEL_ID = "../models/Qwen3.5-0.8B-HF"
PROMPT = "2+2="
MAX_NEW_TOKENS = 3

print(f"Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    device_map="cpu",
    local_files_only=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True, trust_remote_code=True)

inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"]
prompt_len = input_ids.shape[1]

print(f"Prompt tokens: {input_ids[0].tolist()} (length={prompt_len})")
model.eval()

generated = input_ids.clone()
past_key_values = None

for step in range(MAX_NEW_TOKENS + 1):
    is_prefill = (step == 0)
    with torch.no_grad():
        outputs = model(
            generated if is_prefill else generated[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    past_key_values = outputs.past_key_values
    hidden_states = outputs.hidden_states
    last_pos = -1
    print(f"\n=== STEP {step} ({'prefill' if is_prefill else 'gen'} | pos={prompt_len + step - 1}) ===")
    for layer_idx, hidden in enumerate(hidden_states):
        h = hidden[0, last_pos, :].float().numpy()
        print(f"  Layer {layer_idx:2d}: abs_mean={np.abs(h).mean():.6f} | std={h.std():.6f} | min={h.min():.4f} | max={h.max():.4f}")
    logits = outputs.logits[0, last_pos, :].float().numpy()
    top_idx = int(np.argmax(logits))
    print(f"  LOGITS: top={top_idx} ({tokenizer.decode([top_idx])!r}) | value={logits[top_idx]:.4f} | token_19={logits[19]:.4f}")
    dump = {
        "step": step,
        "position": prompt_len + step - 1,
        "is_prefill": is_prefill,
        "layers": [
            {"layer": i, "abs_mean": float(np.abs(h).mean()), "std": float(h.std()), "min": float(h.min()), "max": float(h.max())}
            for i, h in enumerate([hidden[0, last_pos, :].float().numpy() for hidden in hidden_states])
        ],
        "logits": {"top_token": top_idx, "top_value": float(logits[top_idx]), "token_19": float(logits[19])}
    }
    with open(f"hf_step_{step}.json", "w") as f:
        json.dump(dump, f, indent=2)
    if step < MAX_NEW_TOKENS:
        generated = torch.cat([generated, torch.tensor([[top_idx]])], dim=1)
