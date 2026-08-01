# Colibri Port — Architecture Notes

Captured 2026-07-29 from studying `/home/xander/Documents/reference/colibri/`.

These notes are a reference for future Colibri work. They're NOT being
acted on this session — we're building the native Rust safetensors
forward pass instead (Path B).

> **Update 2026-08-01 (project wrap-up):** Still not acted on — the GGUF
> native engine became the shipped path. Colibri lessons already harvested
> (LFRU cache, pread+DONTNEED streaming, expert streaming design) are
> documented in `LEAFCUTTER_STRATEGY.md` and `COLIBRI_ANALYSIS.md`.

## Why Colibri matters

- **Pure C engine**, ~6,744 lines in colibri.c + ~2,498 lines of headers.
- Runs on CPU (no CUDA required). CUDA/Metal backends are OPTIONAL.
- Designed for **massive MoE models** (GLM-5.2 744B) on tiny machines
  (~25 GB RAM, no GPU).
- Speed: 0.05-0.1 tok/s cold on a 12-core laptop, 0.28-0.37 tok/s with
  better NVMe. **Memory-vs-speed tradeoff**, not C-vs-Python tradeoff.
- AirLLM uses transformers (Python) with layer-sharded safetensors.
- Colibri writes its own forward pass from scratch.

## Architecture overview

```
colibri.c (6744 lines)
├── Cfg struct                — model architecture parameters
├── Layer struct              — per-layer weights (attention, MoE, shared)
├── Model struct              — full model state + KV cache + LRU expert cache
├── ESlot struct              — one MoE expert slot (3 quantized matrices)
├── Quantization              — int4/int8/int3 per-row scales, e8 lattice
├── st.h (479 lines)          — safetensors loader (pread + posix_fadvise)
├── tok.h (426 lines)         — BPE tokenizer with cl100k/o200k pre-tokenizer
├── quant.h (1219 lines)      — quantized matmul kernels (SIMD-heavy)
├── backend_loader.c          — Windows CUDA DLL shim (linux links direct)
├── decode_batch.h            — batched decode scheduling
├── uring.h                   — Linux io_uring async I/O for expert streaming
├── sample.h, grammar.h, schema_gbnf.h, telemetry.h, tier.h, compat.h
```

## Forward pass flow

```
forward_all(m, ids, S, pred):
    kv_alloc(m, S)                       # resize KV cache
    x[0..S] = embed_row(m, ids[i])       # token → hidden state
    layers_forward(m, x, S, 0)           # run all layers
    for s in 0..S:
        h = rmsnorm(x[s], final_norm)
        lo = matmul_qt(h, lm_head)
        pred[s] = argmax(lo)
```

```
layers_forward(m, x, S, pos_base):
    for i in 0..n_layers:
        if S >= 8: print "[prefill] layer {i}/{n}"
        layer_forward_rows(m, L[i], i, x, S, pos_base, nrm, tmp)
```

## Key pieces to port (when we come back)

### 1. st.h (479 lines) — DONE ✅
On-demand safetensors reads via `pread` + `posix_fadvise(DONTNEED)`.
Header is parsed once at load; data is read on demand.
**Ported to `rust/src/safetensors_loader.rs`**.

### 2. tok.h (426 lines) — DONE ✅
BPE tokenizer with cl100k/o200k pre-tokenizer, GPT-2 ByteLevel.
**Ported to `rust/src/bpe_tokenizer.rs`**.

### 3. quant.h (1219 lines) — TODO
Quantized matmul kernels for int4/int8/int3/e8.
Not needed for safetensors (which are BF16/F16/F32).
Will need this when porting the GGUF path.

### 4. colibri.c forward pass (6744 lines) — TODO
The main engine. Key functions:
- `rmsnorm(out, x, w, D, eps)` (~10 lines, trivial)
- `embed_row(m, tok, x)` (lookup in embed matrix)
- `attention_rows(m, l, layer, x, S, pos_base, ...)` — MLA with KV-cache
  compression, q/kv-LoRA, RoPE interleaved (370 lines)
- `moe(m, l, layer, x, S, out, with_shared)` — router + expert streaming
  with LRU cache (270 lines)
- `layer_forward_rows(m, l, layer, x, S, ...)` — combines pre-attention
  norm + attention + post-attention norm + MLP + MoE + residual (90 lines)
- `forward_all(m, ids, S, pred)` — top-level (35 lines)
- `generate(m, prompt, np, n_new, out)` — token-by-token with KV cache
  (~50 lines)
- `spec_decode(m, ...)` — speculative decoding with MTP draft (~100 lines)
- `profile_print(m, elapsed)` — timing breakdown
- KV cache management (`kv_alloc`, `kv_persist.h`)

## Quantization design

Colibri uses **per-row scales** with packed int4/int8 weights:
- int8: 1 byte/param + 4 byte scale per row
- int4: 0.5 byte/param + 4 byte scale per row
- int3: 3.5 bits/param + 4 byte scale per 64-element group
- int2: 0.25 byte/param + 4 byte scale per row
- e8 lattice: 3.06 bits/param (IQ3-style)

The resident part (densest ~17B params) is at int4 = 8.7 GB.
That's how GLM-5.2 fits in 15 GB RAM.

## MoE + LRU streaming

The genius move: don't load all experts at once.
- ~744B total params (GLM-5.2), but only ~40B active per token
- Routed experts live on disk (hundreds of GB)
- Per-layer LRU cache: keep only the top-K most recently used
- PILOT thread: prefetches next layer's experts while compute runs
- io_uring for async I/O on Linux (uring.h)
- CACHE_ROUTE: pin "sacred top ranks" (always take, even uncached)

## CUDA / Metal backends

- `backend_cuda.cu` + `backend_cuda.h` (175 + 66 lines)
  - Tensor upload, expert MLP, attention absorb
  - PIPE2 resident stream: layer-by-layer GPU pipeline
- `backend_metal.mm` + `backend_metal.h` (164 lines)
  - Apple GPU equivalent
- Both are OPTIONAL — pure-CPU path is the default

## Profile/chat loop considerations

Colibri does NOT do chat templating — it just runs raw tokens.
The user has to do their own template rendering.
Leafcutter should keep its chat template logic (profiles.rs).

## When to come back to Colibri

After we've shipped a working native safetensors forward pass:
- Add quant.h for GGUF support
- Port the MoE layer forward + router + LRU cache
- Add CUDA/Metal backends (or just CPU if we want simplicity)
- Add speculative decoding with MTP

## Reference commands

```bash
# See the main engine
cd /home/xander/Documents/reference/colibri/c
wc -l *.c *.h

# Look at the data structures
sed -n '82,180p' colibri.c

# See forward pass entry point
grep -n "^static void forward_all\|^static void generate\|^static void layers_forward" colibri.c

# See how safetensors are read
cat st.h

# See how tokenizer works
cat tok.h
```
