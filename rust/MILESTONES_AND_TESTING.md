# LeafcutterLLM — Milestones & Testing Record

**Last updated:** 2026-08-01 (project wrap-up)
**Git commit:** Uncommitted wrap-up work (GPT-2 byte decode fix, Q6_K lm_head block cache, streaming UTF-8 buffer, 3 stale tests fixed)
**Total tests:** **161 passed**, **0 failures**, **3 ignored** (GPU/bench tests) — `cargo test --release --lib`

---

## 2026-08-01 — Project wrap-up (native Ornith chat, decode fix, Q6_K lm_head cache)

- **Native GGUF engine end-to-end** — `leafcutter run ornith` streams the
  full Qwen3.5 hybrid model natively: thinking block as `💭…`, clean answer,
  correct stop tokens. Verified `hey there` → coherent greeting, `2+2` →
  correct arithmetic. Peak chat RAM **~8.1 GB**, 1.2–1.65 tok/s.
- **GPT-2 byte-level decode fixed** — emoji/Latin-1 were printed as `�`
  because multi-byte chars split across byte-level tokens were lossy-decoded
  per token. Added `GgufTokenizer::decode_bytes()` + streaming UTF-8 buffer
  (`emit_complete_utf8`) in engine.rs; multi-byte chars reassemble correctly.
- **lm_head cache: f32 → Q6_K blocks** — `cached_lm_head` now holds native
  Q6_K blocks (~0.8 GB vs 3.79 GiB f32) computed via
  `q6_k_matmul_transposed_b`. Bit-identical logits; ~2× faster; RAM 11.1 → 8.1 GB.
- **3 stale tests fixed** — `test_q4_0_roundtrip` (byte-interleaved Q4_0
  layout), `test_ministral_template_uses_inst` (default system prefix),
  `test_ornith_template_starts_with_thinking` (model emits its own `<think>`).
- **Test count**: `cargo test --release --lib` → **161 passed, 0 failed, 3 ignored**.

---

## 2026-07-24 — Phase 2 async layer prefetch + anti-doom detector

- **5aa6154** Phase 2: `std::thread::scope` wrapping `forward_native`
  layer loop; worker thread runs `load_layer(N+1)` while main does N's
  matmul. Gated `LEAFCUTTER_PREFETCH=1`. Bench (Ministral-3B warm):
  0.74 → 1.24 tok/s (1.68×). Borrow trick documented in commit + skill.

- **aaec49d** Anti-doom: `rust/src/inference/anti_doom.rs` (pure Rust,
  4 unit tests). Two-stage detector — byte-fingerprint repetition
  (port of antidoom `repetition.py`) + token-id n-gram repetition.
  Wired into `generate_native` as a sampler hook that suppresses
  continuation-token logits to -inf before the next sample. Gated
  `LEAFCUTTER_ANTIDOOM=1`. Detection cost: 0.02-0.6 ms per 80-token
  decode (negligible).

- **Test count**: tracked by `cargo test --release --lib --no-default-features`.
  Currently **150 passed**, 1 pre-existing failure (unrelated `test_q4_0_roundtrip`).
  Anti-doom brings 4 of those (`anti_doom::tests::*`); the rest accumulated
  across SIMD lm_head, decoder fixes, and cleanups between Q3 sessions.

---

## Phase 6: Stability & Correctness Audit — 2026-06-16

A targeted review of every module in `rust/src/` for crashes, silent
correctness bugs, performance smells, and unsafe public-facing behaviour.
Ten of eleven findings were fixed without altering the inference math.
Detailed finding/fix table lives in `CHANGELOG.md` v0.9.6 — summary:

| # | Severity | Component | Status |
|---|----------|-----------|--------|
| 1 | CRITICAL | `embed_lookup_mmap` OOB when `vocab_size=0` | ✅ Fixed |
| 2 | CRITICAL | `get_tensor_row_f32[_into]` panic on unsupported quants | ✅ Fixed |
| 3 | HIGH     | API handlers drop `top_p` on the floor | ✅ Fixed |
| 4 | HIGH     | BPE tokenizers destroy whitespace via `split_whitespace` | ✅ Fixed |
| 5 | HIGH     | Speculative decoder stub returns `(0, 0)` and pays draft cost anyway | ✅ Fixed (`SpeculativeStatus::Disabled` enum) |
| 6 | LOW      | Qwen36 `known_extra_suffixes` incomplete | ✅ Fixed |
| 7 | HIGH     | `GGUFile::dequantize` quant-type gap | ✅ Fixed |
| 8 | MEDIUM   | `load_layer` silently skips missing optional tensors | ✅ Fixed |
| 9 | MEDIUM   | `tokenizer_from_model` rebuilt per call | ✅ Fixed (cached) |
| 10 | MEDIUM  | `lm_head_projection` thread-local buffer resized per call | ✅ Fixed (cached capacity) |
| 11 | HIGH    | Ministral-3B `ffn_forward` shape panic | ⏸ Deferred — needs refactor |

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

**Total: 100 passed, 3 failed, 3 ignored**

---

### Milestone 12: Ministral Native Inference (mistral3)
- **File:** `src/model/arch.rs`, `src/model/gguf.rs`, `src/inference/engine.rs`, `src/inference/attention.rs`
- **What:** Ministral-3B and Ministral-8B now run natively with layer streaming.
  - Architecture detection: `"mistral3"` → `ModelArchitecture::Mistral`
  - Metadata correction: `hidden_size` and `num_hidden_layers` corrected from actual tensor shapes (metadata lies: 4096→3072, 32→26 for 3B)
  - Weight name mapping: `output_norm.weight` → `model.norm.weight`, `blk.{i}.attn_norm.weight` → `input_layernorm.weight`, etc.
  - Embedding lookup: handles `embedding_dim != hidden_size` by copying `min(row.len(), hidden_size)` and padding zeros
  - Sliding Window Attention (SWA): `window_size` read from GGUF metadata, masked in attention scoring loop
- **Tests:** `test_generation.rs` — coherent decode verified on both 3B and 8B models
- **Status:** ✅ Complete

---

## Benchmarks

### Environment
- **CPU:** AMD Ryzen 7 5800HS (8 cores / 16 threads)
- **RAM:** 16 GB
- **GPU:** AMD Radeon Vega iGPU (WGPU/OpenGL backend)
- **OS:** Linux (Arch)
- **Rust:** 1.86.0
- **Compile:** `--release`

### Real Model Memory — Layer Streaming + madvise

| Model | Params | File Size | Peak RSS | Status |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct | 3B | 2.0 GB | **534 MB** | ✅ Measured (Q4_K_XL) |
| Ministral-3-3B-Reasoning-2512 | 3B | 2.1 GB | **504 MB** | ✅ Measured (Q4_K_M) |
| Ministral-3-8B-Reasoning-2512 | 8B | 5.2 GB | **739 MB** | ✅ Measured (Q4_K_M) |
| Meta-Llama-3.1-70B-Instruct | 70B | 40.3 GB | **1,145 MB** | ✅ Measured (Q4_K_S) |

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

## Phase 6.5: Generation Quality Bug Hunt (2026-05-19)

### Fixes Applied

| Commit | Fix | Tests |
|--------|-----|-------|
| `567cb44` | SSM state persistence, causal conv1d cache, RoPE position offset | 104 passed |
| `fc3ec67` | Attention layer detection for Qwen3.5 (`attn_q.weight`, `attn_k.weight`, `attn_v.weight`), Q/K per-head RMSNorm | 104 passed |

### Generation Test Results

```
2B-Q4_K_M  "Hello" → top prefill: 'asso' (logit 12.39)
                → generated: '熱çado所提供史یین史史症'
9B-IQ4_NL  "Hello" → top prefill: 98564 (logit 10.19)
                → generated: ' isNew clan_rsa_rsa.Creator�'
```

### Architectural Gap Discovered

Qwen3.5 "SSM" layers are **not standard Mamba**. They use **Gated Delta Net** (linear attention with decay gates), which is fundamentally different from our `selective_scan` implementation:

- Dual input projections (`wqkv` + `wqkv_gate`) vs. our single `attn_qkv.weight`
- L2-normalized Q/K after convolution vs. our raw conv output
- `softplus(alpha + bias) * exp(-A_log)` decay gate vs. our `exp(dt * a_i)`
- Vector state per channel with `build_delta_net` vs. our scalar state
- Gated output normalization (`norm * silu(z)`) vs. our no-gating

Attention layers also differ: Qwen3.5 uses **MRoPE** (multi-section RoPE) and outputs Q+gate from a single projection.

**Status:** Native Rust engine loads Qwen3.5 and produces finite logits, but coherent generation requires full Delta Net implementation. Recommend llama.cpp bridge backend for Qwen3.5 until native support is complete.

---

## Phase 7: IQ4_NL Bug Fix + 70B Validation (2026-05-23)

### Fixes Applied

| # | Fix | File | Tests |
|---|-----|------|-------|
| 1 | **IQ4_NL wrong lookup table** | `src/kernels/mod.rs` | `cargo test --lib iq4` → 5 passed |
| 2 | **check_layer_stats compile fix** | `src/bin/check_layer_stats.rs` | Builds |
| 3 | **check_rms compile fix** | `src/bin/check_rms.rs` | Builds |

### 70B Validation Results

| Model | File Size | Layers | Hidden | Peak RSS | Time/token |
|---|---|---|---|---|---|
| Meta-Llama-3.1-70B-Instruct-Q4_K_S | 40.3 GB | 80 | 8192 | **1,145 MB** | ~142s |

**Claim validated:** 70B loads in 39 MB and runs forward pass in 1,145 MB — well under 4 GB target.

## Known Limitations

1. **WGPU backend only accelerates matmul** — Element-wise ops still run on CPU. For LLM inference, matmul is 80%+ of compute time, so this is acceptable for a first implementation.
2. **Q4_0/Q8_0 matmul requires `n % 32 == 0`** — Real model weights always satisfy this. Small test tensors may fall back to scalar path.
3. **NEON path never executed locally** — x86_64 dev machine; ARM correctness is validated by algorithmic identity with SSE path.
4. **WGPU tests ignored in CI** — Require GPU hardware; run manually with `cargo test -- --ignored`.
5. **Qwen3.5 native support incomplete** — SSM layers implement Mamba selective_scan instead of Gated Delta Net. Use llama.cpp bridge for coherent Qwen3.5 generation.

---

## Phase 8: Auto-FFI Fallback + Dual-Backend Routing (2026-05-19)

### Milestone 8.1: Architecture-Based Backend Routing
- **File:** `src/inference/engine.rs`
- **What:** `detect_arch()` peeks GGUF metadata; qwen3.5/qwen3.6 → `load_ffi()`, others → native.
- **Status:** ✅ Complete

### Milestone 8.2: Auto-FFI Fallback for Unsupported Quants
- **File:** `src/inference/engine.rs`
- **What:** When `capability_report()` reports unsupported quant types (IQ1_M=31, Q2_K, IQ2_XXS, etc.), engine automatically calls `load_ffi()` instead of returning error.
- **Tests:** Llama-3.1-70B-IQ1_M auto-routes to FFI, loads, and prefills successfully.
- **Status:** ✅ Complete

### Milestone 8.3: Native DeltaNet Forward Pass
- **File:** `src/inference/deltanet.rs`
- **What:** Correct delta rule `S_t = decay*S + beta*(v - S^T@k) ⊗ k`, L2-normalized Q/K, softplus decay gates, group norm output gating.
- **Tests:** `test_real_deltanet` produces healthy output magnitudes (~0.2).
- **Status:** ✅ Math correct in isolation; full model coherence WIP.

### Milestone 8.4: Context Lifecycle Fix
- **File:** `src/inference/engine.rs`
- **What:** `generate_ffi()` recreates `LlamaContext` on each call to avoid KV cache position conflicts.
- **Status:** ✅ Complete

---

## Test Results (Latest)

| Model | Backend | Route | tok/sec | Coherent? |
|-------|---------|-------|---------|-----------|
| Llama-3.2-3B Q4_K | Native | Direct | ~0.12 | ✅ Yes |
| Qwen3.5-0.8B Q4_0 | FFI | Explicit | 14.68 | ✅ Yes |
| Qwen3.5-9B IQ4_NL | FFI | Explicit | 2.38 | ✅ Yes |
| Llama-3.1-70B IQ1_M | FFI | Auto-fallback | ~0.03 | ✅ Loads + prefill |

---

## Next Milestones (Proposed)

1. **Native Qwen3.5 coherence** — Debug interaction between DeltaNet layers, attention layers, FFN, and residuals.
2. **SIMD quantized GEMM** — Scalar GEMM is ~142s/token on 70B. Need NEON/AVX2 paths.
3. **Q4_K_M passthrough support** — Load pre-quantized GGUF Q4_K_M models without re-quantization.
4. **ARM dotprod (`vdotq_s32`) optimization** — Pi 5 Cortex-A76 dot-product instructions.
5. **Pi 5 field testing** — Deploy and benchmark on actual hardware.

---

## Milestone M9 — llama.cpp FFI Bridge (2026-05-19)

**Status:** ✅ COMPLETE

### Deliverables
- [x] Hand-written `#[repr(C)]` bindings verified against C header with size/offset checker
- [x] Safe Rust wrappers: `LlamaModel`, `LlamaContext`, `LlamaBatch` with Drop guards
- [x] Autoregressive generation: prefill + greedy/temperature sampling loop
- [x] `llama_tokenize` negative-return bug fixed
- [x] `llama_progress_callback` bool return type fixed
- [x] Context recreation in `generate_ffi()` to avoid KV cache position conflicts
- [x] Unified tokenizer: FFI path uses llama.cpp's built-in tokenizer

### Test Results
- ✅ Llama-3.2-3B-Instruct Q4_K_XL: coherent generation verified
- ✅ Llama-3.2-3B-Instruct IQ4_NL: coherent generation verified  
- ✅ Qwen3.5-9B-Instruct IQ4_NL: coherent generation + reasoning verified
- ✅ Qwen3.5-0.8B Q4_0: 14.68 tok/sec, coherent
- ❌ Qwen3.5-9B-UD-Q8_K_XL (13GB): OOM on 8GB system (expected)
- ✅ Llama-3.1-70B-IQ1_M: auto-fallback to FFI, loads, prefill works

### Performance
- 3B Q4_K_XL on CPU: ~2-3 tok/sec
- 9B IQ4_NL on CPU: ~2.4 tok/sec
- 0.8B Q4_0 on CPU: ~14.7 tok/sec
- Startup: instant (no subprocess spawn)
