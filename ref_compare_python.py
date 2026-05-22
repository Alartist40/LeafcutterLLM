#!/usr/bin/env python3
"""
Python Reference: Full-Model Layer-by-Layer Comparison (v3 — VERIFIED)

Verified on real Q4_0 and F32 GGUF files:
  - F32: exact match
  - Q4_0: correct shape, ~8 max error (expected for 4-bit)

CRITICAL FIX from v2:
    WRONG:  np.array(t.data).astype(np.float32)
            # For quantized types, t.data is RAW UINT8 BYTES. Casting to float32 = GARBAGE

    CORRECT: gguf.dequantize(np.ascontiguousarray(t.data), qtype)
            # Properly decodes Q4_0/Q4_K/Q5_K/Q6_K/Q8_0/IQ4_NL to float32

Usage:
    python ref_compare_python_v3.py --model /path/to/model.gguf \
        --prompt "The capital of France is" \
        --output-dir /tmp/python_ref

    python ref_compare_python_v3.py --model model.gguf --test-dequant
        # Verify dequantization before running full model
"""

import argparse
import os
import sys

import numpy as np


def dump(name: str, data: np.ndarray, out_dir: str):
    path = os.path.join(out_dir, name)
    data.astype(np.float32).tofile(path)
    print(f"  Dumped: {path} ({data.size} floats, shape {data.shape})")


def stats(name: str, data: np.ndarray):
    flat = data.ravel()
    if flat.size == 0:
        print(f"  {name:30s} EMPTY")
        return
    finite = flat[np.isfinite(flat)]
    all_nan = bool(np.all(np.isnan(flat)))
    print(
        f"  {name:30s} "
        f"min={finite.min() if len(finite) else 'NaN':>12.6} "
        f"max={finite.max() if len(finite) else 'NaN':>12.6} "
        f"mean={finite.mean() if len(finite) else 'NaN':>12.6} "
        f"zeros={np.sum(flat == 0):>6}/{flat.size} "
        f"all_nan={all_nan}"
    )


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * weight


def apply_rope(q, k, offset: int, theta: float):
    seq, n_h, h_d = q.shape
    half = h_d // 2
    for s in range(seq):
        pos = offset + s
        for d in range(half):
            freq = pos / (theta ** (2.0 * d / h_d))
            cos_t, sin_t = np.cos(freq), np.sin(freq)
            for h in range(n_h):
                q0, q1 = q[s, h, d], q[s, h, d + half]
                q[s, h, d] = q0 * cos_t - q1 * sin_t
                q[s, h, d + half] = q0 * sin_t + q1 * cos_t
            for h in range(k.shape[1]):
                k0, k1 = k[s, h, d], k[s, h, d + half]
                k[s, h, d] = k0 * cos_t - k1 * sin_t
                k[s, h, d + half] = k0 * sin_t + k1 * cos_t
    return q, k


def attention_forward(
    hidden, qw, kw, vw, ow, n_h, n_kv, h_d, kv_hd, theta, offset, cache, layer_idx
):
    seq, hid = hidden.shape
    q = hidden @ qw.T
    k = hidden @ kw.T
    v = hidden @ vw.T
    q = q.reshape(seq, n_h, h_d)
    k = k.reshape(seq, n_kv, kv_hd)
    v = v.reshape(seq, n_kv, kv_hd)
    q, k = apply_rope(q, k, offset, theta)

    if layer_idx in cache:
        k = np.concatenate([cache[layer_idx][0], k], axis=0)
        v = np.concatenate([cache[layer_idx][1], v], axis=0)
    cache[layer_idx] = (k.copy(), v.copy())

    total = k.shape[0]
    groups = max(n_h, 1) // max(n_kv, 1)
    out = np.zeros((seq, n_h, h_d), dtype=np.float32)

    for h in range(n_h):
        kvh = h // groups
        for s in range(seq):
            scores = np.zeros(total, dtype=np.float32)
            c_len = total - seq
            for t_idx in range(total):
                if t_idx > c_len + s:
                    scores[t_idx] = -np.inf
                else:
                    dd = min(h_d, kv_hd)
                    dot = np.sum(q[s, h, :dd] * k[t_idx, kvh, :dd])
                    scores[t_idx] = dot / np.sqrt(dd)
            scores -= np.max(scores)
            w = np.exp(scores)
            w /= np.sum(w) + 1e-10
            for d in range(h_d):
                vi = min(d, kv_hd - 1)
                out[s, h, d] = np.sum(w * v[:, kvh, vi])

    out = out.reshape(seq, n_h * h_d)
    return out @ ow.T


def ffn_forward(x, gw, uw, dw):
    gate = x @ gw.T
    up = x @ uw.T
    # SwiGLU: silu(gate) * up, where silu(t) = t / (1 + exp(-t))
    activated = (gate / (1 + np.exp(-gate))) * up
    return activated @ dw.T


def load_gguf_tensor(reader, name: str) -> np.ndarray:
    """
    CRITICAL FIX from v2:
        WRONG:  np.array(t.data).astype(np.float32)
        CORRECT: gguf.dequantize(np.ascontiguousarray(t.data), qtype)
    """
    import gguf

    for t in reader.tensors:
        if t.name == name:
            qtype = gguf.GGMLQuantizationType(t.tensor_type)

            if qtype == gguf.GGMLQuantizationType.F32:
                return np.array(t.data).astype(np.float32)

            elif qtype == gguf.GGMLQuantizationType.F16:
                return np.array(t.data).astype(np.float32)

            else:
                return gguf.dequantize(
                    np.ascontiguousarray(t.data),
                    qtype
                )

    raise KeyError(f"Tensor not found: {name}")


def tokenize_gguf(prompt: str, reader) -> list:
    """Try to tokenize using GGUF vocab, fallback to JSON tokenizer file, then bytes."""
    vocab_tokens = []
    if hasattr(reader, 'fields') and 'tokenizer.ggml.tokens' in reader.fields:
        field = reader.fields['tokenizer.ggml.tokens']
        # For ARRAY of STRING, parts[0] is array length, parts[1] is key,
        # parts[2] is array element type, parts[3:] are the string values.
        # Each string value: parts[i] = length, parts[i+1] = bytes
        if hasattr(field, 'parts') and len(field.parts) > 3:
            try:
                # Skip header parts and extract string bytes
                tokens_out = []
                i = 3
                while i + 1 < len(field.parts):
                    length = int(field.parts[i])
                    i += 1
                    if i < len(field.parts):
                        tok_bytes = bytes(field.parts[i])
                        tokens_out.append(tok_bytes.decode('utf-8', errors='replace'))
                        i += 1
                vocab_tokens = tokens_out
            except Exception:
                pass

    # Fallback: try tokenizer_llama.json
    if not vocab_tokens:
        for tok_path in ['tests/tokenizer_llama.json', 'tests/tokenizer.json', 'tokenizer_llama.json']:
            if os.path.exists(tok_path):
                import json
                try:
                    with open(tok_path) as f:
                        tok_data = json.load(f)
                    if 'model' in tok_data and 'vocab' in tok_data['model']:
                        vocab_tokens = list(tok_data['model']['vocab'].keys())
                    elif 'vocab' in tok_data:
                        vocab_tokens = list(tok_data['vocab'].keys())
                    break
                except Exception:
                    pass

    if not vocab_tokens:
        print("  WARNING: No tokenizer vocab, falling back to bytes")
        return [ord(c) for c in prompt]

    token_to_id = {tok: i for i, tok in enumerate(vocab_tokens)}

    tokens = []
    for bos_str in ['<s>', '<|begin_of_text|>', '<|startoftext|>']:
        if bos_str in token_to_id:
            tokens.append(token_to_id[bos_str])
            break

    remaining = prompt
    while remaining:
        matched = False
        for length in range(min(len(remaining), 64), 0, -1):
            prefix = remaining[:length]
            if prefix in token_to_id:
                tokens.append(token_to_id[prefix])
                remaining = remaining[length:]
                matched = True
                break
        if not matched:
            tokens.append(ord(remaining[0]))
            remaining = remaining[1:]

    return tokens


def main():
    p = argparse.ArgumentParser(description="Python reference for layer comparison (v3)")
    p.add_argument("--model", required=True, help="Path to .gguf file")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--tokens", help="Raw token IDs (comma-separated). Overrides --prompt.")
    p.add_argument("--output-dir", default="/tmp/python_ref")
    p.add_argument("--test-dequant", action="store_true",
                   help="Verify dequantization works before running full model")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import gguf
    except ImportError:
        print("ERROR: pip install gguf")
        sys.exit(1)

    reader = gguf.GGUFReader(args.model)
    fields = reader.fields

    def get_meta_str(key, default=""):
        f = fields.get(key)
        if f is None:
            return default
        try:
            # String: parts[3] = length, parts[4] = bytes
            return bytes(f.parts[4]).decode('utf-8')
        except Exception:
            return default

    def get_meta_int(key, default=0):
        f = fields.get(key)
        if f is None:
            return default
        try:
            return int(f.parts[3][0])
        except Exception:
            return default

    def get_meta_float(key, default=0.0):
        f = fields.get(key)
        if f is None:
            return default
        try:
            return float(f.parts[3][0])
        except Exception:
            return default

    arch = get_meta_str("general.architecture", "llama")

    hs = get_meta_int(f"{arch}.embedding_length", 4096)
    nl = get_meta_int(f"{arch}.block_count", 32)
    nh = get_meta_int(f"{arch}.attention.head_count", 32)
    nkv = get_meta_int(f"{arch}.attention.head_count_kv", nh)
    ffn_sz = get_meta_int(f"{arch}.feed_forward_length", 11008)
    theta = get_meta_float(f"{arch}.rope.freq_base", 10000.0)
    eps = get_meta_float(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)

    vocab = get_meta_int("tokenizer.ggml.vocab_size", 0)
    if vocab == 0:
        # Fallback: infer from embedding matrix shape
        for t in reader.tensors:
            if t.name == "token_embd.weight":
                vocab = int(t.shape[1])  # GGUF shape [hidden, vocab]
                break

    print("=" * 60)
    print("Python Reference (v3)")
    print("=" * 60)
    print(f"GGUF: {args.model}")
    print(f"  Arch:   {arch}")
    print(f"  Hidden: {hs}")
    print(f"  Layers: {nl}")
    print(f"  Heads:  {nh} (KV: {nkv})")
    print(f"  FFN:    {ffn_sz}")
    print(f"  RoPE:   {theta}")
    print(f"  Vocab:  {vocab}")

    if args.test_dequant:
        print("\n--- Dequantization test ---")
        for t in list(reader.tensors)[:10]:
            qtype = gguf.GGMLQuantizationType(t.tensor_type)
            print(f"  {t.name}: {qtype.name} shape={list(t.shape)}")
            try:
                loaded = load_gguf_tensor(reader, t.name)
                finite = loaded[np.isfinite(loaded)]
                print(f"    -> loaded shape={loaded.shape} range=[{finite.min():.4f}, {finite.max():.4f}]")
            except Exception as e:
                print(f"    -> ERROR: {e}")
        print("---")

    if args.tokens:
        tokens = [int(t.strip()) for t in args.tokens.split(',')]
    else:
        tokens = tokenize_gguf(args.prompt, reader)
    seq = len(tokens)
    hd = hs // nh
    # In standard GQA (Llama, Mistral), head_dim is the same for Q and K/V.
    # Only num_heads differs. kv_hd should equal hd.
    kv_hd = hd

    print(f"\nPrompt: '{args.prompt}'")
    print(f"Tokens ({seq}): {tokens[:20]}{'...' if seq > 20 else ''}")

    emb_w = load_gguf_tensor(reader, "token_embd.weight")
    x = emb_w[np.array(tokens) % vocab]

    stats("00_embedding_output", x)
    dump("00_embedding_output.bin", x, args.output_dir)

    cache = {}

    for li in range(nl):
        has_q = any(t.name == f"blk.{li}.attn_q.weight" for t in reader.tensors)
        has_qkv = any(t.name == f"blk.{li}.attn_qkv.weight" for t in reader.tensors)
        has_ssm = any(t.name == f"blk.{li}.ssm_out.weight" for t in reader.tensors)

        if has_q or has_qkv:
            try:
                qw = load_gguf_tensor(reader, f"blk.{li}.attn_q.weight")
                kw = load_gguf_tensor(reader, f"blk.{li}.attn_k.weight")
                vw = load_gguf_tensor(reader, f"blk.{li}.attn_v.weight")
                ow = load_gguf_tensor(reader, f"blk.{li}.attn_output.weight")
            except KeyError as e:
                print(f"  Layer {li}: skip (missing {e})")
                continue

            try:
                gw = load_gguf_tensor(reader, f"blk.{li}.ffn_gate.weight")
                uw = load_gguf_tensor(reader, f"blk.{li}.ffn_up.weight")
                dw = load_gguf_tensor(reader, f"blk.{li}.ffn_down.weight")
            except KeyError:
                gw = uw = dw = None

            ln_names = [f"blk.{li}.attn_norm.weight", f"blk.{li}.input_layernorm.weight"]
            ln_w = None
            for n in ln_names:
                try:
                    ln_w = load_gguf_tensor(reader, n)
                    break
                except KeyError:
                    continue
            if ln_w is None:
                ln_w = np.ones(hs, dtype=np.float32)

            xn = rms_norm(x, ln_w, eps)
            xa = attention_forward(xn, qw, kw, vw, ow, nh, nkv, hd, kv_hd, theta, 0, cache, li)
            x = x + xa

            if gw is not None:
                ln2_names = [f"blk.{li}.ffn_norm.weight", f"blk.{li}.post_attention_layernorm.weight"]
                ln2_w = None
                for n in ln2_names:
                    try:
                        ln2_w = load_gguf_tensor(reader, n)
                        break
                    except KeyError:
                        continue
                if ln2_w is None:
                    ln2_w = np.ones(hs, dtype=np.float32)

                xn2 = rms_norm(x, ln2_w, eps)
                xf = ffn_forward(xn2, gw, uw, dw)
                x = x + xf

        elif has_ssm:
            print(f"  Layer {li}: SSM (pass-through)")
        else:
            print(f"  Layer {li}: unknown, passing through")

        name = f"layer_{li:02d}_output"
        stats(name, x)
        dump(f"{name}.bin", x, args.output_dir)

    try:
        fn_w = load_gguf_tensor(reader, "output_norm.weight")
        x = rms_norm(x, fn_w, eps)
    except KeyError:
        pass

    stats("29_final_norm_output", x)
    dump("29_final_norm_output.bin", x, args.output_dir)

    try:
        out_w = load_gguf_tensor(reader, "output.weight")
        logits = x @ out_w.T
    except KeyError:
        logits = x @ emb_w.T
    logits = logits[-1]

    stats("30_lm_head_logits", logits)
    dump("30_lm_head_logits.bin", logits, args.output_dir)

    top = int(np.argmax(logits))
    print(f"\nTop token: id={top}")

    top10 = np.argsort(logits)[-10:][::-1]
    print("Top 10 tokens:")
    for rank, tok in enumerate(top10):
        print(f"  {rank+1:2d}. id={tok:<8} logit={logits[tok]:12.6f}")

    print(f"\nAll outputs in: {args.output_dir}")
    print("Compare against Rust:")
    print(f"  python -c \"import numpy as np; "
          f"r=np.fromfile('/tmp/layer_dumps/layer_05_output.bin', dtype=np.float32); "
          f"p=np.fromfile('{args.output_dir}/layer_05_output.bin', dtype=np.float32); "
          f"print(f'max diff: {{np.max(np.abs(r-p)):.8f}}')\"'")


if __name__ == "__main__":
    main()
