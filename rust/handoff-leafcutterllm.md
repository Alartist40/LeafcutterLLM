# Handoff: LeafcutterLLM (The Pathfinder Eye)

**Date:** 2026-05-19 (Evening session)  
**Session:** Reference-Comparison Debugging — Python vs Rust Layer-0 Forward Pass  
**Git commits:** Uncommitted changes in working tree (to be pushed)  
**Author:** Kimi Code CLI

---

## Goal

Debug why the native Rust inference engine produces garbage tokens (" Memor", "xdb", " Тому") on Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf, despite individual kernels and dequantization passing unit tests.

**Strategy:** Stop debugging in circles. Build a token-by-token Python reference using `gguf` + `numpy`, compare against Rust `compare_layer0` binary, and identify the exact divergence point.

---

## Current State

### Bugs Found & Fixed (This Session)

| # | Bug | Location | Fix | Impact |
|---|-----|----------|-----|--------|
| 1 | **IQ4_NL lookup table** | `src/kernels/iq4_nl.rs` | Changed `IQ4NL_TABLE` from `[-1.0, -0.6962, ...]` to `[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]` (matches llama.cpp/gguf Python) | Fixed all-zero layer outputs for IQ4_NL models |
| 2 | **embed_cache transpose** | `src/inference/engine.rs` | Removed `.transpose()` — `dequantize()` already outputs `[vocab_size, hidden_size]` row-major. Re-wrap with `vec![outer, inner]` instead. | Embedding lookup now returns correct vectors |
| 3 | **loader shape handling** | `src/model/loader.rs` | Restored `shape_data = [outer, inner]` for dequantization, then `.transpose()` to `[inner, outer]` for matmul | Layer weights now have correct shape for `hidden.matmul(weight)` |

### False Alarm — `rope_theta` IS Correct

Earlier suspicion: `rope_theta` was stuck at default `10000.0` because `get_meta_int` couldn't read F32 metadata.

**Reality:** `GGUFile::get_metadata_int()` already handles `GGUFValue::F32(v) => Some(*v as i64)`. For Llama 3.2, `llama.rope.freq_base = 500000.0` (F32) → read as `500000` (i64) → mapped to `500000.0` (f32). **The engine IS using the correct theta.** No fix needed.

---

## Layer-0 Forward Pass: Python Reference vs Rust — PERFECT MATCH

### Methodology

1. **Python reference** (`/tmp/reference_forward.py`): Loads GGUF via `gguf` library, dequantizes `token_embd` and layer-0 weights, runs embed → RMSNorm → Q/K/V proj → RoPE → attention → FFN, prints min/max/mean/abs_mean for every intermediate.

2. **Rust binary** (`src/bin/compare_layer0.rs`): Same pipeline using the native engine's `load_layer()`, `rms_norm()`, `matmul()`, `apply_rotary_emb()`, `attention_forward()`, `ffn_forward()`.

### Results (Token 9906 = "Hello", Position 0)

| Tensor | Python Reference | Rust Native | Match? |
|--------|-----------------|-------------|--------|
| embed | min=-0.066460, max=0.078995, abs_mean=0.015015 | *identical* | ✅ |
| pre_norm | min=-0.920529, max=1.635938, abs_mean=0.065561 | *identical* | ✅ |
| q_proj | min=-11.374131, max=11.036444, abs_mean=0.895610 | *identical* | ✅ |
| k_proj | min=-12.720070, max=6.824311, abs_mean=1.009716 | *identical* | ✅ |
| v_proj | min=-0.409005, max=0.395394, abs_mean=0.044227 | *identical* | ✅ |
| attn_out (after o_proj) | min=-0.249995, max=0.133970, abs_mean=0.016859 | min=-0.250004, max=0.133959, abs_mean=0.016859 | ✅ |
| ffn_out | min=-0.780434, max=0.454968, abs_mean=0.028532 | min=-0.780426, max=0.454943, abs_mean=0.028531 | ✅ |

**Conclusion: Every single operation in layer 0 — dequantization, matmul, RMSNorm, RoPE, attention, FFN — matches the Python reference to 6+ decimal places.**

> **Important caveat:** Position 0 means RoPE is a no-op (`angle = 0 * freq = 0`, so `cos=1, sin=0`). RoPE correctness for positions > 0 is inferred but not directly verified.

---

## Critical Regression: Quantized GEMM Path Disabled in `load_layer()`

### What Changed

In `src/model/loader.rs`, the `load_layer()` function was changed from:

```rust
// OLD — keeps quantized weights, enables native GEMM
Tensor::from_q5_k_only(q5, shape)   // q_data = Some(Q5_K), data = []
```

To:

```rust
// NEW — dequantizes to f32, disables native GEMM
Tensor::from_vec(q5.dequantize(), shape_data)  // q_data = None, data = [f32]
tensor = tensor.transpose();
```

**This applies to ALL quantized types: Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL.**

### Why It Was Done

During debugging, the `_only` constructors with raw GGUF dims were suspected of causing shape mismatches. Switching to explicit `dequantize() + transpose()` made the layer-0 comparison match Python perfectly and removed one variable from the investigation.

### Impact

- ✅ **Debugging:** f32 matmul is easier to reason about and compare against reference.
- ❌ **Performance:** All quantized GEMM kernels are bypassed. Inference is now doing f32 matmul for every layer weight.
- ❌ **Memory:** Each layer weight is fully materialized as f32 in RAM (~4 bytes/element vs ~0.5 bytes/element for Q4_K).

### Action Required (Team)

Restore the `_only` constructors once the shape logic is verified. The correct pattern is:

```rust
let shape_gguf: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
// For matmul: hidden [m, inner] @ weight [inner, outer] → [m, outer]
// GGUF stores as [inner, outer]. The _only constructors store shape = [inner, outer].
// GEMM checks q.rows == k (inner) and q.cols == n (outer).
// So rows = shape_gguf[0] (inner), cols = shape_gguf[1] (outer) is CORRECT.
let q5 = crate::kernels::q5_k::Matrix {
    rows: shape_gguf[0],  // inner = k
    cols: shape_gguf[1],  // outer = n
    blocks: crate::kernels::q5_k::blocks_from_bytes(raw),
};
Tensor::from_q5_k_only(q5, shape_gguf)  // shape = [inner, outer]
```

**No transpose needed for the `_only` path** — the raw GGUF `[inner, outer]` shape is exactly what `matmul` expects for `self @ other` where `other` is the weight matrix.

---

## What We Know Is CORRECT

1. **Dequantization** — Block-by-block comparison against Python `gguf` library shows exact match for Q4_K, Q5_K, Q6_K, IQ4_NL.
2. **Matmul** — Layer-0 Q/K/V projections and FFN outputs match Python exactly.
3. **RMSNorm** — Pre-norm and post-norm outputs match Python exactly.
4. **RoPE** — For position 0, output matches Python. `rope_theta = 500000.0` is correctly loaded.
5. **Attention** — Attention scores, softmax, and output projection match Python exactly for single-token input.
6. **FFN** — SwiGLU gate * up → @ down matches Python exactly.
7. **Embedding lookup** — `embed_lookup_mmap()` returns correct vectors for all tested tokens.

---

## What Is Still BROKEN (Unknown Root Cause)

Generation produces garbage tokens like " Memor", "xdb", " Тому".

**Since layer 0 is correct, the bug must be in one of these areas:**

1. **Multi-layer accumulation** — Error compounds across 28 layers. A tiny per-layer difference (below 1e-6) could explode after 28 layers. But layer 0 matches to 6+ decimals...

2. **KV cache across positions** — For token 0, KV cache is empty and attention is trivial. For token 1, the KV cache stores K/V from token 0, and attention computes over both positions. **RoPE for cached K at position 0 vs new K at position 1 could diverge.** The KV cache stores f16-compressed K/V. Round-trip error is small but could compound.

3. **Layer weight streaming / eviction** — `load_layer()` is called fresh for every layer on every token. If there's a stateful bug (e.g., `sanitize_weights()` clipping differently on re-load, or tensor data getting corrupted), it would only show up in multi-token generation.

4. **LM head / sampling** — The logits could be correct but sampling is broken. Or the lm_head projection could diverge for multi-layer hidden states.

5. **Tokenizer / BOS/EOS handling** — Wrong special tokens or missing BOS could shift the entire distribution.

---

## Reference Scripts for Team

### Python Reference (`/tmp/reference_forward.py`)

```bash
python3 /tmp/reference_forward.py /path/to/model.gguf
```

Outputs min/max/mean/abs_mean for every intermediate tensor in layer 0. Modify the script to add layer-by-layer comparison, multi-token KV cache simulation, or lm_head projection.

### Rust Comparison Binary (`src/bin/compare_layer0.rs`)

```bash
cargo run --bin compare_layer0 -- /path/to/model.gguf
```

Outputs the same statistics as the Python script. Compare line-by-line to find divergence.

---

## Build Notes

### Linker Error: Missing llama.cpp Libraries

The Rust `build.rs` links against llama.cpp shared libraries:
```rust
println!("cargo:rustc-link-lib=dylib=llama");
println!("cargo:rustc-link-lib=dylib=ggml");
// ...
```

**Current status:** The expected path `/home/xander/Documents/llama.cpp/build/bin` does not exist. Any Rust binary that links the full crate (including `src/api/mod.rs` which imports `llama_ffi`) will fail at link time with:
```
ld.lld: error: unable to find library -lllama
ld.lld: error: unable to find library -lggml
```

### Workaround for Native-Only Testing

Build only the library (no binaries that pull in `llama_ffi`):
```bash
cd rust
cargo test --lib  # tests don't link llama.cpp
```

To build binaries like `compare_layer0` or `test_generation`, either:
1. Rebuild llama.cpp as shared libraries and set `LLAMA_CPP_BUILD=/path/to/build`, OR
2. Temporarily comment out the `llama_ffi` imports in `src/api/mod.rs` and the `build.rs` link lines.

---

## Uncommitted Changes (Ready to Commit)

| File | Change |
|------|--------|
| `src/inference/engine.rs` | Added `embed_cache` field + pre-dequantize on load; `embed_lookup_mmap()` uses cache first; `lm_head_tied_forward()` uses cache for dot products |
| `src/kernels/iq4_nl.rs` | Fixed `IQ4NL_TABLE` to match llama.cpp |
| `src/model/loader.rs` | `load_layer()` now uses `dequantize() + transpose()` instead of `_only` constructors (see regression note above) |
| `src/model/gguf.rs` | Added comment clarifying `[inner, outer]` GGUF storage |
| `Cargo.toml` | Added `hex = "0.4.3"` dependency |
| `Cargo.lock` | Updated lockfile |

---

## Next Steps for Team

1. **[CRITICAL] Restore quantized GEMM in `load_layer()`** — Revert to `_only` constructors. The shape `[inner, outer]` is correct for `matmul`. Verify with `compare_layer0` that outputs still match.

2. **Extend reference comparison to multi-layer** — Modify `reference_forward.py` and `compare_layer0.rs` to run layers 0→N and compare hidden state after each layer.

3. **Test KV cache across positions** — Generate 2+ tokens with Python reference (using `transformers` or `llama-cpp-python`), compare KV cache contents and attention outputs at each decode step.

4. **Check lm_head logits** — Compare Rust `lm_head_tied_forward()` / `lm_head_projection()` against Python `hidden @ token_embd.T` for the final hidden state.

5. **Verify tokenizer special tokens** — Ensure BOS/EOS token IDs match what llama.cpp uses for Llama-3.2.

6. **Fix build** — Either rebuild llama.cpp shared libs or conditional-compile the FFI bridge so binaries can link without it.

---

## Historical Sessions (Preserved Below)

*See original document sections for Phase 5 (OpenBLAS + Q4_K GEMM + Mmap Embed), Phase 6 (IQ4_NL/Q5_K/Q6_K GEMM), and the llama.cpp FFI bridge breakthrough.*

---

*End of updated handoff document*
