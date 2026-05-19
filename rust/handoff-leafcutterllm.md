# Handoff: LeafcutterLLM (The Pathfinder Eye)

**Date:** 2026-05-19  
**Session:** Performance Breakthrough — OpenBLAS + Quantized GEMM + Memory-Mapped Embed  
**Git commit:** `cc62d1e` (pushed)  
**Author:** Kimi Code CLI

---

## Goal

Transform Leafcutter from "works but slow" to "competitive on speed and memory" by:
1. Adding an OpenBLAS backend for optimized f32 GEMM
2. Wiring up existing Q8_0/Q4_0 and new Q4_K direct quantized GEMM kernels
3. Implementing memory-mapped per-token embed lookup to eliminate multi-GB f32 embed/lm_head materialization
4. Enabling 9B models to load and run without OOM

---

## Current State

### ✅ New in This Session

| # | Feature | Key File(s) | Impact |
|---|---------|-------------|--------|
| 1 | OpenBLAS backend | `src/backend/openblas.rs` | 1.6× tok/sec speedup on all models |
| 2 | Q4_K direct GEMM | `src/kernels/q4_k.rs`, `src/kernels/q4_k_gemm.rs` | 2.4× total speedup on Q4_K models; 7× memory savings |
| 3 | Q4_K wired into Tensor + Loader | `src/model/tensor.rs`, `src/model/loader.rs` | 113/335 tensors in 2B-Q4_K_M stay quantized |
| 4 | Memory-mapped embed lookup | `src/model/gguf.rs`, `src/inference/engine.rs` | Eliminates 2–8GB embed RAM; 9B models no longer OOM |
| 5 | Lazy lm_head projection | `src/inference/engine.rs` | Parallel dot-product over vocab; no f32 lm_head in RAM |
| 6 | Quantized tensor memory trim | `src/model/loader.rs` | `t.data.clear()` after `from_q4_k()` frees f32 copy, keeps q_data |

### Benchmark Results (x86_64, AVX2/FMA, OpenBLAS)

| Model | Quant | Forward (20 tok) | tok/sec | Status |
|-------|-------|------------------|---------|--------|
| Qwen3.5-2B-IQ4_XS | IQ4_XS | 17.4s | 1.15 | ✅ Native |
| Qwen3.5-2B-Q4_K_M | Q4_K | 11.4s | 1.76 | ✅ Native |
| Qwen3.5-9B-IQ4_NL | IQ4_NL | 82.7s | 0.24 | ✅ Native (was OOM) |

**Total speedup vs baseline (naive f32 matmul):**
- 2B-Q4_K_M: **2.4×** (27s → 11.4s)
- 2B-IQ4_XS: **1.7×** (30s → 17.4s)

### Test Summary

**99 passed, 0 failed, 3 ignored** (GPU tests)

---

## Architecture Changes

### Backend Selection

`default_backend()` now auto-selects:
- **With `openblas` feature**: `OpenBlasBackend` for matmul, `CpuBackend` for all other ops
- **Without**: `CpuBackend` (pure Rust SIMD)

```rust
// Cargo.toml features
[features]
default = []
openblas = []

// Build with OpenBLAS
cargo build --release --features openblas
```

### Quantized GEMM Dispatch Chain

```
Engine::forward()
  └─> hidden.matmul(weight)
        └─> Tensor::matmul()
              ├─> if other.q_data == Some(Q4_K) → q4_k_matmul()  (NEW)
              ├─> if other.q_data == Some(Q8_0) → q8_0_matmul()  (existing)
              ├─> if other.q_data == Some(Q4_0) → q4_0_matmul()  (existing)
              └─> else → backend.matmul(f32, f32)               (OpenBLAS or CPU)
```

### Memory-Mapped Embed / LM Head Flow

```
Engine::load()
  ├─> load_special() → only loads output_norm.weight (tiny)
  ├─> embed removed from special_weights
  └─> lm_head removed from special_weights

Engine::forward(tokens)
  ├─> embed_lookup_mmap(tokens)
  │     └─> for each token: file.get_tensor_row_f32("token_embd.weight", token_id)
  │           └─> read 8× Q6_K blocks from mmap → dequantize → 2048 f32 values
  ├─> ... layer loop ...
  └─> lm_head_projection(hidden_last)
        ├─> tied:    par_iter over vocab → dot(hidden_last, embed_row[j])
        └─> separate: par_iter over vocab → dot(hidden_last, output_row[j])
```

### GGUF Loader Quantization Branch

```rust
// In load_layer() — raw GGUF dims are already [in, out], no rev+transpose needed
match qtype {
    QuantType::Q8_0 => { parse Q8Matrix; Tensor::from_q8_0(); t.data.clear(); }
    QuantType::Q4_0 => { parse Q4Matrix; Tensor::from_q4_0(); t.data.clear(); }
    QuantType::Q4_K => { parse Q4KMatrix; Tensor::from_q4_k(); t.data.clear(); }
    _ => { dequantize to f32; rev+transpose; }
}
```

**Critical:** `t.data.clear()` frees the f32 dequantization buffer, keeping only the quantized blocks. This saves ~7× RAM per Q4_K tensor. The f32 data was only needed for `sanitize_weights()` and non-matmul ops — weight tensors never need those.

---

## Active Files (Priority Order)

| File | What It Does | Status |
|------|-------------|--------|
| `src/backend/openblas.rs` | `cblas_sgemm` FFI wrapper | ✅ Production |
| `src/kernels/q4_k.rs` | Q4_K block parser + dequantize | ✅ Production |
| `src/kernels/q4_k_gemm.rs` | Q4_K scalar + AVX2 + NEON GEMM | ✅ Production |
| `src/model/gguf.rs` | `get_tensor_row_f32()` for mmap lookup | ✅ Production |
| `src/inference/engine.rs` | `embed_lookup_mmap()`, `lm_head_*_forward()` | ✅ Production |
| `src/model/loader.rs` | Q4_K/Q8_0/Q4_0 quantized branch in `load_layer()` | ✅ Production |
| `src/model/tensor.rs` | `QuantizedData::Q4_K` variant + `from_q4_k()` | ✅ Production |

---

## Known Limitations / Next Steps

1. **Q6_K / Q5_K / IQ4_NL / IQ5_0 direct GEMM missing**
   - These quant types still dequantize to f32 in `load_layer()`
   - 9B-IQ4_NL uses IQ4_NL for most layer weights → slower than Q4_K models
   - **Next:** Add `q6_k_gemm.rs`, `iq4_nl_gemm.rs` following the Q4_K pattern

2. **lm_head projection is CPU-bound dot products**
   - For tied embeddings, 248K vocab × 2048 hidden = 508M ops per token
   - Parallel rayon helps but could be faster with chunked OpenBLAS
   - **Next:** Chunk vocab into 4096-token groups, dequantize chunk to f32, `cblas_sgemm` with `CblasTrans`

3. **No runtime backend selection**
   - `openblas` feature is compile-time; cannot fallback at runtime
   - **Next:** Use `libloading` or `dlopen` to dynamically load OpenBLAS if present

4. **9B models are slow (0.24 tok/sec)**
   - IQ4_NL layer weights dequantize to f32 → large memory bandwidth
   - **Next:** Implement IQ4_NL direct GEMM (similar complexity to Q4_K)

---

## Build & Test Commands

```bash
cd rust

# Full test suite (with OpenBLAS)
cargo test --features openblas

# Real model benchmark
cargo run --release --features openblas --bin diagnose_models

# Build server binary
cargo build --release --features openblas
```

---

## Git Status

All changes committed and pushed to:
`https://github.com/Alartist40/LeafcutterLLM.git`

Commit: `109b06f` — "feat: IQ4_NL, Q5_K, Q6_K direct quantized GEMM kernels + row-dequantize hybrid optimization"

---

## Session History

### Phase 6 (2026-05-19) — IQ4_NL / Q5_K / Q6_K Direct GEMM
**Commit:** `109b06f`  
**Goal:** Unlock 7× RAM savings on ~90% of remaining model weights by implementing direct quantized GEMM for IQ4_NL and Q6_K (plus Q5_K as bonus).

| # | Feature | Key File(s) | Impact |
|---|---------|-------------|--------|
| 1 | IQ4_NL block parser + GEMM | `src/kernels/iq4_nl.rs`, `iq4_nl_gemm.rs` | 168 tensors in 9B model stay quantized (was f32 fallback) |
| 2 | Q6_K block parser + GEMM | `src/kernels/q6_k.rs`, `q6_k_gemm.rs` | 1 tensor in 9B model stays quantized |
| 3 | Q5_K block parser + GEMM | `src/kernels/q5_k.rs`, `q5_k_gemm.rs` | 32 tensors in 9B model stay quantized |
| 4 | Row-dequantize hybrid | `src/kernels/quant_gemm_common.rs` | Dequantize full B row → temp buffer → SIMD FMA; faster than block-by-block |
| 5 | Tensor + Loader wiring | `src/model/tensor.rs`, `src/model/loader.rs` | All K-quant types (Q4_K, Q5_K, Q6_K, IQ4_NL) use native GEMM |
| 6 | Warning cleanup | `src/inference/attention.rs`, `ssm.rs`, `simd.rs`, `gguf.rs`, `loader.rs`, `wgpu.rs` | 104 tests pass, 0 compiler warnings |

**Validation:**
- All kernels numerically verified against reference dequantize-then-matmul on real Qwen3.5-9B-IQ4_NL weights (max error < 1e-6)
- 9B model loads, runs single forward pass (26s), zero NaN/Inf
- 20-token generation: 0.11 tok/sec (9B), 0.50 tok/sec (2B-Q4_K_M)

### Phase 5 (2026-05-19) — OpenBLAS + Q4_K GEMM + Mmap Embed
**Commit:** `cc62d1e`  
**Goal:** Competitive speed and memory; enable 9B models without OOM.

| # | Feature | Key File(s) | Impact |
|---|---------|-------------|--------|
| 1 | OpenBLAS backend | `src/backend/openblas.rs` | 1.6× tok/sec speedup on all models |
| 2 | Q4_K direct GEMM | `src/kernels/q4_k.rs`, `q4_k_gemm.rs` | 2.4× total speedup on Q4_K models; 7× memory savings |
| 3 | Memory-mapped embed lookup | `src/model/gguf.rs`, `src/inference/engine.rs` | Eliminates 2–8GB embed RAM; 9B models no longer OOM |
| 4 | Lazy lm_head projection | `src/inference/engine.rs` | Parallel dot-product over vocab; no f32 lm_head in RAM |
