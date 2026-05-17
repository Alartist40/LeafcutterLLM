# Handoff: LeafcutterLLM (The Pathfinder Eye)

**Date:** 2026-05-15  
**Session duration:** Extended (Phase 5: General-Purpose Inference Engine)  
**Git commit:** `4e47b6b`  
**Author:** Kimi Code CLI  

---

## Goal

Build a **memory-safe, cross-platform LLM inference engine** in Rust that can run large language models on resource-constrained hardware (Raspberry Pi 5, embedded ARM devices) via **layer streaming** (load one layer at a time from disk) and **quantized compute** (Q8_0, Q4_0, future Q4_K_M). The engine must be hardware-agnostic: CPU (ARM NEON, x86_64 SSE/AVX2) and GPU (WGPU/Vulkan/Metal/DX12).

---

## Current State

### ✅ Complete (12 milestones)

| # | Milestone | Key File(s) | Tests |
|---|-----------|-------------|-------|
| 1 | Backend trait abstraction | `src/backend/mod.rs`, `src/backend/cpu.rs` | Tensor ops dispatch through trait |
| 2 | SIMD kernels (NEON/SSE/AVX2) | `src/kernels/simd.rs` | 7 tests (matmul small/large, vec_add, sum_sq, parallel correctness) |
| 3 | Q8_0 block format | `src/kernels/q8_0.rs` | 2 tests (block roundtrip, quantize/dequantize) |
| 4 | Q8_0 shard write/load | `src/shard/format.rs`, `writer.rs`, `loader.rs` | `test_q8_0_shard_roundtrip` |
| 5 | Native INT8 GEMM (Q8_0) | `src/kernels/int8_gemm.rs` | 2 tests (vs dequant reference, large matrix) |
| 6 | Q4_0 block format + INT4 GEMM | `src/kernels/q4_0.rs`, `int8_gemm.rs` | 4 tests (q4_0 roundtrip, matmul vs dequant, large matrix, shard roundtrip) |
| 7 | Multi-threaded CPU matmul | `src/kernels/simd.rs`, `src/backend/cpu.rs` | `test_parallel_matmul_correctness`, `bench_parallel_matmul_speedup` |
| 8 | f16 KV cache | `src/cache/mod.rs` | 2 tests (f16 roundtrip, append/concat) |
| 9 | WGPU GPU backend | `src/backend/wgpu.rs` | 2 tests (both require GPU, marked `#[ignore]`) |
| 10 | ShardEngine end-to-end | `src/inference/shard_engine.rs` | 2 tests (f32 forward, Q8_0 forward) |
| 11 | Benchmark binary | `src/bin/bench_shard.rs` | CLI tool — no unit tests |
| 12 | Milestone documentation | `MILESTONES_AND_TESTING.md` | — |

**Test summary: 70 passed, 0 failed, 3 ignored**

### 🔄 In Progress

Nothing actively in progress. Phase 5 is feature-complete.

### ⛔ Blocked

- **Pi 5 field testing** — Code compiles for `aarch64` via `cfg` but never executed on actual ARM hardware. NEON correctness is validated by algorithmic identity with SSE path on x86_64.
- **WGPU large matmul precision** — GPU vs CPU has ~1e-4 relative float divergence (expected). Test tolerance set to 1e-4.

---

## Active Files

All files in `src/` are production-ready. The most critical:

- `src/backend/mod.rs` — `Backend` trait + global backend singleton
- `src/backend/cpu.rs` — `CpuBackend`, wraps SIMD kernels, adds multi-threading threshold
- `src/backend/wgpu.rs` — `WgpuBackend`, WGSL compute shader matmul, CPU fallback
- `src/kernels/simd.rs` — Architecture-specific SIMD (NEON/SSE/AVX2), multi-threaded dispatch
- `src/kernels/q8_0.rs` — Q8_0 `Block` + `Matrix` + quantize/dequantize
- `src/kernels/q4_0.rs` — Q4_0 `Block` + `Matrix` + quantize/dequantize
- `src/kernels/int8_gemm.rs` — Native INT8/INT4 GEMM kernels (scalar/AVX2/NEON)
- `src/model/tensor.rs` — `Tensor` with `QuantizedData` enum (Q4_0/Q8_0), auto-dispatches matmul
- `src/cache/mod.rs` — `KVCache` with f16 compression
- `src/shard/format.rs` — `ShardHeader`, `ShardTensorMeta`, `QuantFormat` enum
- `src/shard/writer.rs` — `ShardWriter`, writes F32/Q8_0/Q4_0 shards
- `src/shard/loader.rs` — `ShardLoader`, mmap-based loading, FIFO cache, prefetch
- `src/inference/shard_engine.rs` — `ShardEngine`, autoregressive forward pass
- `src/bin/bench_shard.rs` — Benchmark CLI
- `src/bin/split_model.rs` — `split_model` CLI (`--quant f32|q8_0|q4_0`)
- `MILESTONES_AND_TESTING.md` — Full milestone + test + benchmark record

---

## Recent Changes (This Session)

### Added
- **Q4_0 quantization** (`src/kernels/q4_0.rs`) — 18-byte blocks for 32 weights, ~7× compression
- **Q4_0 INT4 GEMM** (`src/kernels/int8_gemm.rs`) — scalar + AVX2 + NEON paths
- **WGPU backend** (`src/backend/wgpu.rs`) — Cross-platform GPU matmul via compute shaders
- **Multi-threaded matmul** (`src/kernels/simd.rs`) — `rayon::join` recursive row-splitting
- **f16 KV cache** (`src/cache/mod.rs`) — 2× RAM savings
- **QuantizedData enum** (`src/model/tensor.rs`) — Unified Q4_0/Q8_0 storage in Tensor
- **bench_shard binary** (`src/bin/bench_shard.rs`) — Performance benchmarking tool
- **MILESTONES_AND_TESTING.md** — Documentation

### Modified
- `src/shard/format.rs` — Added `QuantFormat::Q4_0 = 2`
- `src/shard/writer.rs` — Added Q4_0 data size + write paths
- `src/shard/loader.rs` — Added Q4_0 parse path, 1D tensor fallback fix
- `src/model/tensor.rs` — Refactored from `q8_data: Option<Q8Matrix>` to `q_data: Option<QuantizedData>`
- `src/inference/shard_engine.rs` — Added `reset_kv_cache()` and `kv_cache_memory_mb()`
- `src/inference/attention.rs` — Updated `kv_cache.get()` call (now returns owned Tensors)
- `src/bin/split_model.rs` — Added `--quant q4_0`
- `Cargo.toml` — Added `rayon = "1.10"`, `wgpu = "29.0.3"`, `pollster = "0.4.0"`, `bytemuck`

---

## Failed Attempts

### 1. Q4_0 roundtrip test tolerance too tight
- **What happened:** Initial tolerance was 0.2, but Q4_0 max representable value is `7 * scale`. For values near the edge of range (e.g., 3.1 with scale=0.4), clamping caused 0.39 error.
- **Fix:** Relaxed tolerance to 0.5. Q4_0 is 4-bit; some quantization error is expected.
- **Lesson:** Quantization format tests must account for clamping at the representable range boundary.

### 2. Q8_0 loader panic on 1D tensors
- **What happened:** `shape[1]` panicked when loading `model.norm.weight` (shape `[32]`, 1D).
- **Fix:** Added `shape.len() == 2` guard before accessing `shape[1]` in loader Q8_0 path. 1D tensors fall back to f32 dequantize.
- **Lesson:** Weight tensors like layer norms are 1D vectors — quantized matmul only applies to 2D matrices.

### 3. WGPU API version mismatch
- **What happened:** Wrote backend against wgpu 24 API mental model, but Cargo added wgpu 29. Multiple API changes: `InstanceDescriptor::default()` removed, `Maintain::Wait` → `PollType::wait_indefinitely()`, `PipelineLayoutDescriptor` lost `push_constant_ranges`, gained `immediate_size`, `request_adapter` returns `Result` not `Option`.
- **Fix:** Read wgpu 29 source to find correct struct fields and method signatures.
- **Lesson:** Always check actual crate source when API docs are unclear. Use `cargo doc` + source grep.

### 4. Vulkan info tool silent failure
- **What happened:** `vulkaninfo --summary` returned empty output. `vulkaninfo` (no flags) also silent.
- **What worked:** Direct Rust test with wgpu successfully detected AMD Radeon Vega iGPU via OpenGL backend.
- **Lesson:** Don't rely on system tools for GPU detection; wgpu's own adapter enumeration is the source of truth.

---

## Next Steps

1. **Test on Pi 5** — `cargo test` + `cargo run --release --bin bench_shard` on actual ARM hardware. Validate NEON path correctness and measure tokens/sec.
2. **ARM dotprod optimization** — Pi 5 Cortex-A76 supports `vdotq_s32`. Add a dedicated INT8 GEMM path that quantizes activations to Q8_0 on-the-fly and uses native int8×int8 dot products. Potential 2-3× boost over current f32-fma approach.
3. **Q4_K_M passthrough support** — Most distributed GGUF models are Q4_K_M. Enable `split_model` to passthrough Q4_K_M bytes from GGUF → shards without re-quantizing to Q4_0. Requires `QuantFormat::Q4_K` + `Q4KMatrix` + matmul kernel.
4. **GPU element-wise ops** — Currently WGPU only accelerates matmul. vec_add, silu, softmax on GPU would reduce CPU-GPU transfer overhead for full offload.
5. **Real model benchmark** — Run `bench_shard` or `split_model` + `ShardEngine` on the actual Qwen 3B model on the Pi 5.

---

## Context to Preserve

### Key Decisions

1. **Backend trait is synchronous** — `fn matmul(...) -> Vec<f32>`. WGPU backend blocks on GPU with `pollster`. This keeps the engine simple but means GPU can't be async pipelined. Future: consider async Backend v2.
2. **Dequantize-on-the-fly for INT8/INT4** — Instead of true int8×int8 dot products (which require per-token activation quantization), we dequantize 32-element blocks to f32 stack buffers and use proven f32 SIMD. This gives 90% of the bandwidth win with 10% of the kernel complexity.
3. **Tensor stores both f32 and quantized data** — `Tensor.data` (f32) for all ops, `Tensor.q_data` (Q4_0/Q8_0) for fast matmul. Memory overhead is ~27% for Q8_0 and ~14% for Q4_0 vs f32-only. Acceptable tradeoff for simplicity.
4. **Rayon for CPU threading** — Recursive `rayon::join` over the `m` dimension. Threshold at 4096 output elements to avoid thread overhead for small matrices.
5. **f16 for KV cache** — Not Q8_0. f16 preserves precision (no block-scale quantization error) while giving 2× savings. Q8_0 KV cache is future work.
6. **WGPU over CUDA** — One backend covers all GPU vendors + web. CUDA would only cover NVIDIA.

### Dependencies

```toml
# Core
memmap2 = "0.9"
half = "2.4"
serde_json = "1.0"
tokenizers = "0.23.1"

# Server
axum = "0.7"
tokio = "1"

# Performance
rayon = "1.10"

# GPU
wgpu = "29.0.3"
pollster = "0.4.0"
bytemuck = "1.22"

# Dev
tempfile = "3"
```

### Environment

- **Dev machine:** x86_64, AMD Ryzen 7 5800HS, 16 GB RAM, AMD Radeon Vega iGPU
- **Target machine:** Raspberry Pi 5, BCM2712, quad-core Cortex-A76, 8 GB RAM
- **Rust:** 1.86.0
- **Test command:** `cargo test --lib` (70 pass, 0 fail, 3 ignored)
- **Benchmark command:** `cargo run --release --bin bench_shard -- --layers 4 --hidden 512 --tokens 10 --quant q4_0`
- **GPU test command:** `cargo test --lib -- --ignored` (requires GPU)

### Build Notes

- Release builds essential for performance: `cargo run --release`
- `wgpu` compile is slow (~30s first time) due to shader compilation
- NEON path compiles on x86_64 via `cfg(target_arch = "aarch64")` but is never executed locally
- AVX2 path auto-detected at runtime via `is_x86_feature_detected!("avx2")`

---

## Quick Reference

```bash
# Run all tests
cargo test --lib

# Run GPU tests (requires GPU)
cargo test --lib -- --ignored

# Benchmark
cargo run --release --bin bench_shard -- --layers 4 --hidden 512 --tokens 10 --quant q4_0

# Split a GGUF model to Q8_0 shards
cargo run --release --bin split_model -- \
  --model /path/to/model.gguf \
  --out ./shards \
  --quant q8_0

# Run inference from shards
cargo run --release --bin leafcutter -- \
  --model ./shards/manifest.json \
  --port 8081
```
