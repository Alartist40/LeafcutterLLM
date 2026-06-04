#!/usr/bin/env python3
"""Dump per-layer hidden states from HF Transformers for ground-truth comparison."""

import sys
import numpy as np

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

    print(f"Loading {model_id}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Prompt tokens: {input_ids.shape[1]}", file=sys.stderr)
    print(f"Token IDs: {input_ids[0].tolist()}", file=sys.stderr)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states[0] = embedding output
    # hidden_states[1] = after layer 0
    # ...
    # hidden_states[N] = after layer N-1 (final hidden before lm_head)
    hidden_states = outputs.hidden_states

    print("\n=== Per-layer hidden states ===")
    for i, h in enumerate(hidden_states):
        arr = h.numpy().flatten()
        print(f"Layer {i:2d}: abs_mean={np.abs(arr).mean():.6f} std={arr.std():.6f} min={arr.min():.4f} max={arr.max():.4f}")

    # Logits
    logits = outputs.logits[0, -1, :].numpy()
    top_idx = int(np.argmax(logits))
    print(f"\n=== Logits (last token) ===")
    print(f"Top token: {top_idx} (logit={logits[top_idx]:.4f})")
    top10 = np.argsort(logits)[-10:][::-1]
    for idx in top10:
        tok_str = tokenizer.decode([int(idx)], skip_special_tokens=False)
        print(f"  {int(idx):6d}: logit={logits[int(idx)]:8.4f} -> {tok_str!r}")

    # Save full hidden states for detailed comparison
    np.savez("hf_layers.npz", **{f"layer_{i}": h.numpy() for i, h in enumerate(hidden_states)})
    np.save("hf_logits.npy", logits)
    print("\nSaved hf_layers.npz and hf_logits.npy", file=sys.stderr)

if __name__ == "__main__":
    main()
