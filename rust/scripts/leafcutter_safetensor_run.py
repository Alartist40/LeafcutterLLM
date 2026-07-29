#!/usr/bin/env python3
"""
Leafcutter safetensor streaming inference script.

Protocol (newline-delimited JSON over stdin/stdout):
  stdin:  one JSON object with keys
            path        : path to safetensors model directory
            prompt      : the full prompt text (already includes chat template)
            max_tokens  : int
            temperature : float
            top_p       : float
            top_k       : int
            stop        : list[str]
            think_open  : token id to treat as thinking opener (Ornith: 248068)
            think_close : token id to treat as thinking closer  (Ornith: 248069)
          then EOF.

  stdout: one JSON object per line, either
            {"type":"thinking_open"}   -- before the first think_open token
            {"type":"thinking_close"}  -- after the think_close token
            {"type":"token","text":"..."}    -- streamed surface token
            {"type":"done","tokens":N,"duration_s":D}

The Rust parent (`leafcutter`) reads stdout events and updates its REPL.

Model is loaded lazily (on first call) and kept warm for subsequent calls.

Why this exists:
  - The native Rust forward pass on GGUF is still being debugged for
    hybrid (Qwen3.5 / Ornith) models.
  - HuggingFace transformers + safetensors is the reference implementation
    — proven correct end-to-end (top-1 'Paris' on a factual prompt).
  - AirLLM uses the same approach (layer-sharded transformers).
  - This script gives Leafcutter a working chat path TODAY, even if
    slow on CPU, while the native GGUF engine continues to be improved.

Usage from leafcutter:
  leafcutter run /path/to/safetensors --engine safetensor
  The Rust cmd_run_safetensor() shells out to this script and parses stdout.
"""
import json
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Globals: model + tokenizer loaded once per process
# ---------------------------------------------------------------------------
_MODEL = None
_TOKENIZER = None
_MODEL_PATH = None
_DEVICE = "cpu"  # users without GPU still get working inference (slow but correct)


def load_model(path: str):
    global _MODEL, _TOKENIZER, _MODEL_PATH
    if _MODEL_PATH == path and _MODEL is not None:
        return _MODEL, _TOKENIZER
    # Quiet transformers/HF logging during load.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import warnings
    warnings.filterwarnings("ignore")
    # Newest transformers prefers `dtype`; older accepted `torch_dtype`.
    # Try `dtype` first, fall back if not supported.
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map=_DEVICE, trust_remote_code=True
        )
    except TypeError:
        _TOKENIZER = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map=_DEVICE, trust_remote_code=True
        )
    _MODEL.eval()
    _MODEL_PATH = path
    return _MODEL, _TOKENIZER


def stream(prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: int,
           stop: list[str], think_open: int, think_close: int):
    model, tok = _MODEL, _TOKENIZER
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = inputs.input_ids
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature if temperature > 0.0 else 1.0,
        top_p=top_p,
        top_k=top_k if top_k > 0 else 0,
        pad_token_id=tok.eos_token_id,
    )
    # Stream via TextStreamer is the cleanest path; we collect events instead.
    # transformers' `generate` returns full sequence; we slice the new tokens
    # and decode incrementally.  This avoids needing TextIteratorStreamer
    # which can deadlock on CPU with a single consumer thread.
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, **gen_kwargs)
    new_ids = out[0, ids.shape[1]:].tolist()
    duration = time.time() - t0

    in_thinking = False
    n = 0
    for tok_id in new_ids:
        n += 1
        if tok_id == think_open:
            sys.stdout.write(json.dumps({"type": "thinking_open"}) + "\n")
            in_thinking = True
            continue
        if tok_id == think_close:
            sys.stdout.write(json.dumps({"type": "thinking_close"}) + "\n")
            in_thinking = False
            continue
        # Stop strings
        surface = tok.decode([tok_id])
        if any(s and s in surface for s in stop):
            break
        sys.stdout.write(json.dumps({"type": "token", "text": surface,
                                     "in_thinking": in_thinking}) + "\n")
    sys.stdout.write(json.dumps({"type": "done", "tokens": n,
                                 "duration_s": duration}) + "\n")
    sys.stdout.flush()


def main():
    # Read exactly one JSON command from stdin.
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"type": "error", "message": "no command on stdin"}))
        return
    try:
        cmd = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"type": "error", "message": f"bad json: {e}"}))
        return

    path = cmd["path"]
    prompt = cmd["prompt"]
    max_tokens = int(cmd.get("max_tokens", 256))
    temperature = float(cmd.get("temperature", 0.6))
    top_p = float(cmd.get("top_p", 0.95))
    top_k = int(cmd.get("top_k", 20))
    stop = list(cmd.get("stop", []))
    think_open = int(cmd.get("think_open", 248068))
    think_close = int(cmd.get("think_close", 248069))

    try:
        load_model(path)
    except Exception as e:
        print(json.dumps({"type": "error", "message": f"load failed: {e}"}))
        return

    try:
        stream(prompt, max_tokens, temperature, top_p, top_k, stop,
               think_open, think_close)
    except Exception as e:
        print(json.dumps({"type": "error", "message": f"inference failed: {e}"}))


if __name__ == "__main__":
    main()
