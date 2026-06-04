#!/usr/bin/env python3
"""
Dump per-layer hidden states from HuggingFace transformers.
Run: python3 debug_hf_layers.py > hf_layers.txt
"""
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "../models/Qwen3.5-0.8B-HF"
PROMPT = "2+2="

def main():
    print(f"Loading {MODEL_ID}...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    inputs = tokenizer(PROMPT, return_tensors="pt")
    print(f"Prompt tokens: {inputs['input_ids'][0].tolist()}", file=sys.stderr)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        # hidden_states[0] = embedding output (before layer 0)
        # hidden_states[1] = after layer 0
        # ...
        # hidden_states[N] = after final layer (before lm_head)
        
        for i, hidden in enumerate(outputs.hidden_states):
            h = hidden[0, -1, :].to(torch.float32).numpy()  # Last token, all features
            print(f"LAYER {i:2d} | mean={h.mean():.6f} | abs_mean={np.abs(h).mean():.6f} | "
                  f"std={h.std():.6f} | min={h.min():.6f} | max={h.max():.6f}")
            
            # Also dump the full vector for numerical diff
            np.save(f"hf_layer_{i:02d}.npy", h)

        # Logits for last token
        logits = outputs.logits[0, -1, :].to(torch.float32).numpy()
        top_idx = int(np.argmax(logits))
        print(f"\nLOGITS | top_token={top_idx} | top_value={logits[top_idx]:.6f}", file=sys.stderr)
        print(f"TOKEN_19={logits[19]:.6f} TOKEN_248068={logits[248068]:.6f}", file=sys.stderr)

if __name__ == "__main__":
    main()
