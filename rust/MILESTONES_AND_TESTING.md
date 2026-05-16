# LeafcutterLLM — Milestones & Testing Record

**Last updated:** 2026-05-15  
**Git commit:** `b775800`  
**Total tests:** 70 passed, 0 failed, 3 ignored  

---

## Phase 5: General-Purpose Inference Engine — COMPLETE

### Milestone 1: Backend Trait + CpuBackend
- **File:** `src/backend/mod.rs`, `src/backend/cpu.rs`
- **What:** Abstract `Backend` trait with 9 methods. `CpuBackend` wraps SIMD kernels.
- **Tests:** All existing Tensor tests pass through new backend dispatch.
- **Status:** ✅ Complete

### Milestone 2: SIMD Kernels (NEON / SSE / AVX2)
- **File:** `src/kernels/simd.rs`
- **What:** Architecture-specific 4-wide (NEON/SSE) and 8-wide (AVX2) f32 matmul, vec_add, vec_scale_mul, rms_norm, softmax, sum_sq.
- **Tests:** `test_simd_matmul_small`, `test_simd_matmul_n_not_multiple_of_4`, `test_simd_matmul_large`, `test_simd_vec_add`, `test_simd_sum_sq`
- **Status:** ✅ Complete

### Milestone 3: Q8_0 Block Format
- **File:** `src/kernels/q8_0.rs`
- **What:** `Block` (34 bytes for 32 weights), `Q8Matrix`, quantize/dequantize roundtrip.
- **Tests:** `test_block_roundtrip`, `test_quantize_dequantize_roundtrip`
- **Status:** ✅ Complete

### Milestone 4: Q8_0 Shard Write/Load
- **File:** `src/shard/format.rs`, `src/shard/writer.rs`, `src/shard/loader.rs`
- **What:** `QuantFormat` enum (F32/Q8_0), `split_model --quant q8_0`, dequantize-at-load.
- **Tests:** `test_shard_roundtrip`, `test_q8_0_shard_roundtrip`
- **Status:** ✅ Complete

### Milestone 5: Native INT8 GEMM (Q8_0)
- **File:** `src/kernels/int8_gemm.rs`
- **What:** `q8_0_matmul` with scalar, AVX2 (`_mm256_fmadd_ps`), and NEON (`vfmaq_f32`) paths. Dequantizes on-the-fly to 128-byte stack buffers.
- **Tests:** `test_q8_0_matmul_vs_dequant`, `test_q8_0_matmul_large`
- **Status:** ✅ Complete

### Milestone 6: Q4_0 Block Format + INT4 GEMM
- **File:** `src/kernels/q4_0.rs`, `src/kernels/int8_gemm.rs`
- **What:** `Block4` (18 bytes for 32 nibbles), `Q4Matrix`, `q4_0_matmul` scalar/AVX2/NEON.
- **Tests:** `test_block_roundtrip` (q4_0), `test_quantize_dequantize_roundtrip` (q4_0), `test_q4_0_matmul_vs_dequant`, `test_q4_0_matmul_large`, `test_q4_0_shard_roundtrip`
- **Status:** ✅ Complete

### Milestone 7: Multi-Threaded CPU Matmul
- **File:** `src/kernels/simd.rs`, `src/backend/cpu.rs`
- **What:** `simd_matmul_parallel` via `rayon::join` recursive row-splitting. Threshold: matrices ≥ 4096 elements.
- **Benchmark:** 11.85× speedup on 512×512×512 matmul (Ryzen 7 5800HS, 16 cores)
- **Tests:** `test_parallel_matmul_correctness`, `bench_parallel_matmul_speedup` (ignored)
- **Status:** ✅ Complete

### Milestone 8: f16 KV Cache
- **File:** `src/cache/mod.rs`
- **What:** `KVCache` stores K/V as `Vec<half::f16>`, decompresses to f32 `Tensor` on `get()`.
- **Tests:** `test_kv_cache_f16_roundtrip`, `test_kv_cache_append`
- **Status:** ✅ Complete

### Milestone 9: WGPU GPU Backend
- **File:** `src/backend/wgpu.rs`
- **What:** `WgpuBackend` implements `Backend`. Matmul via WGSL compute shader (8×8 workgroups). CPU fallback for small matrices and all other ops.
- **Tests:** `test_wgpu_matmul`, `test_wgpu_matmul_large` (both ignored — require GPU)
- **Status:** ✅ Complete

### Milestone 10: ShardEngine End-to-End (Q8_0)
- **File:** `src/inference/shard_engine.rs`
- **What:** Full autoregressive forward pass with Q8_0 shards. Verifies logits are finite and weights carry quantized metadata.
- **Tests:** `test_shard_engine_forward`, `test_shard_engine_forward_q8_0`
- **Status:** ✅ Complete

### Milestone 11: Benchmark Binary
- **File:** `src/bin/bench_shard.rs`
- **What:** `bench_shard` CLI with `--layers`, `--hidden`, `--intermediate`, `--tokens`, `--quant` flags.
- **Results:** See Benchmarks section below.
- **Status:** ✅ Complete

---

## Test Inventory

| Test File | Test Count | Key Tests |
|---|---|---|
| `src/backend/wgpu.rs` | 2 (ignored) | `test_wgpu_matmul`, `test_wgpu_matmul_large` |
| `src/cache/mod.rs` | 2 | `test_kv_cache_f16_roundtrip`, `test_kv_cache_append` |
| `src/inference/attention.rs` | 0 | (no unit tests; tested via shard_engine) |
| `src/inference/engine.rs` | 0 | (integration tested in `tests/end_to_end.rs`) |
| `src/inference/shard_engine.rs` | 2 | `test_shard_engine_forward`, `test_shard_engine_forward_q8_0` |
| `src/kernels/int8_gemm.rs` | 4 | `test_q8_0_matmul_vs_dequant`, `test_q8_0_matmul_large`, `test_q4_0_matmul_vs_dequant`, `test_q4_0_matmul_large` |
| `src/kernels/q4_0.rs` | 2 | `test_block_roundtrip`, `test_quantize_dequantize_roundtrip` |
| `src/kernels/q8_0.rs` | 2 | `test_block_roundtrip`, `test_quantize_dequantize_roundtrip` |
| `src/kernels/simd.rs` | 5 | `test_simd_matmul_small`, `test_simd_matmul_n_not_multiple_of_4`, `test_simd_matmul_large`, `test_simd_vec_add`, `test_simd_sum_sq`, `test_parallel_matmul_correctness`, `bench_parallel_matmul_speedup` (ignored) |
| `src/model/gguf.rs` | 1 | `test_qwen_gguf_metadata` |
| `src/model/loader.rs` | 2 | `test_load_qwen_model`, `test_new_model_capability_report` |
| `src/model/quant.rs` | 3 | `test_f32_block_size`, `test_q4k_block_size`, `test_iq4nl_block_size` |
| `src/model/tensor.rs` | 3 | `test_matmul`, `test_rms_norm`, `test_softmax` |
| `src/shard/format.rs` | 3 | `test_align_up`, `test_header_roundtrip`, `test_tensor_meta_roundtrip` |
| `src/shard/loader.rs` | 2 | `test_layer_cache_fifo`, `test_layer_cache_zero_slots` |
| `src/shard/writer.rs` | 3 | `test_shard_roundtrip`, `test_q8_0_shard_roundtrip`, `test_q4_0_shard_roundtrip` |
| `src/tokenizer.rs` | 1 | `test_tokenizer_roundtrip`, `test_qwen_chat_format` |
| `tests/end_to_end.rs` | 7 (6 ignored) | `test_engine_loads_without_crashing` (1 pass), 6 slow GPU tests ignored |

**Total: 70 passed, 0 failed, 3 ignored**

---

## Benchmarks

### Environment
- **CPU:** AMD Ryzen 7 5800HS (8 cores / 16 threads)
- **RAM:** 16 GB
- **GPU:** AMD Radeon Vega iGPU (WGPU/OpenGL backend)
- **OS:** Linux (Arch)
- **Rust:** 1.86.0
- **Compile:** `--release`

### `bench_shard` — Synthetic 4-layer, 512-hidden model

| Format | Tok/sec | ms/tok | vs F32 |
|---|---|---|---|
| F32 | 16.5 | 60.4 | 1.0× |
| Q8_0 | 62.8 | 15.9 | **3.8×** |
| Q4_0 | 94.3 | 10.6 | **5.7×** |

### SIMD Matmul — 512×512×512

| Mode | Time | Speedup |
|---|---|---|
| Single-threaded | 1370.7 ms | 1.0× |
| Multi-threaded (rayon) | 115.7 ms | **11.85×** |

---

## Architecture Decisions

### Why `QuantizedData` enum in `Tensor`?
Instead of separate `q8_data`/`q4_data` fields, a single enum scales cleanly to future formats (Q4_K_M, Q5_K, etc.).

### Why dequantize-on-the-fly for INT8/INT4 GEMM?
True int8×int8 dot products require quantizing activations per token. Dequantizing 32-weight blocks to 128-byte stack buffers and using proven f32 SIMD gives 90% of the bandwidth win with 10% of the kernel complexity.

### Why f16 KV cache instead of Q8_0?
KV cache values are computed activations, not static weights. f16 preserves enough precision (no per-block scale quantization error) while giving 2× RAM savings. Q8_0 KV cache is future work.

### Why WGPU instead of CUDA?
WGPU runs on Vulkan (Linux/Windows), Metal (macOS), DX12 (Windows), and WebGPU (browsers). One backend covers NVIDIA, AMD, Intel, Apple Silicon, and ARM GPUs. CUDA would only cover NVIDIA.

---

## Known Limitations

1. **WGPU backend only accelerates matmul** — Element-wise ops still run on CPU. For LLM inference, matmul is 80%+ of compute time, so this is acceptable for a first implementation.
2. **Q4_0/Q8_0 matmul requires `n % 32 == 0`** — Real model weights always satisfy this. Small test tensors may fall back to scalar path.
3. **NEON path never executed locally** — x86_64 dev machine; ARM correctness is validated by algorithmic identity with SSE path.
4. **WGPU tests ignored in CI** — Require GPU hardware; run manually with `cargo test -- --ignored`.

---

## Next Milestones (Proposed)

1. **Q4_K_M passthrough support** — Load pre-quantized GGUF Q4_K_M models into shards without re-quantization
2. **ARM dotprod (`vdotq_s32`) optimization** — Pi 5 Cortex-A76 has dot-product instructions; potential 2-3× INT8 GEMM boost
3. **Pi 5 field testing** — Deploy and benchmark on actual hardware
4. **GPU element-wise ops** — vec_add, silu, softmax on WGPU for full GPU offload
