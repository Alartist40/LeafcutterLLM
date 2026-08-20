# LeafcutterLLM — Colony Architecture

> **The colony:** one Rust engine that adapts to the model and the hardware it
> runs on. LeafcutterLLM is not a single program with a single strategy — it is
> a **colony of systems**, each tuned for a different regime, dispatched from a
> single binary and sharing one loader.

This document records the current state (2026-08-02) and the design intent for
the colony. It supersedes the tier-specific strategy docs (removed 2026-08-17):
`strategy.md` (safetensors streaming, historical), `LEAFCUTTER_STRATEGY.md`
(GGUF integration, the fast-engine layers), and `COLIBRI_ANALYSIS.md` /
`docs/research/airllm_vs_colibri.txt` (competitive research). This doc is the
top-level view of how they fit together.

---

## 1. The Colony Vision

The mission is to **run large models on small hardware** and to beat the
two reference systems on *architecture*, not on scale:

| | AirLLM | Colibri | LeafcutterLLM (intent) |
|--|--------|---------|------------------------|
| Language | Python (torch/transformers) | C (single file) | Rust (memory-safe, no runtime deps) |
| Regime | Layer-wise, dense 70B–405B | Disk-streamed MoE, hundreds of B | **All regimes, auto-dispatched** |
| Key weakness | Python overhead, dumb one-layer-at-a-time | Slow by design (0.05–1 tok/s), fixed LRU | (must not inherit either) |
| Smart idea | — | per-expert LRU + pread/DONTNEED streaming | **adaptive memory-bounded cache** (cache what fits, stream the rest) |

The core differentiator: **AirLLM streams one layer at a time and Colibri keeps a
fixed-size LRU; Leafcutter's loader measures available RAM and caches exactly as
many layers as fit, evicting oldest-first when the budget shrinks.** The engine
self-tunes — no user knob required to trade speed for memory.

---

## 2. The Three Tiers

```
                    ┌──────────────────────────────────────────────┐
                    │            leafcutter run <model>            │
                    └───────────────┬──────────────────────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   Tier 1: GPU present │──→ Vulkan/ROCm/CUDA offload
                        │   (not yet built)     │    (like Ollama --gpu-layers)
                        └───────────────────────┘
                                    │ no GPU
                        ┌───────────▼───────────┐
                        │  Tier 2: fits in RAM? │──yes→ Fast engine: cache all
                        │  (fast small-model)   │      layers, ~1.65 tok/s
                        └───────────────────────┘
                                    │ no
                        ┌───────────▼───────────┐
                        │ Tier 3: too big       │──→ Adaptive streaming:
                        │  (large-model tier)   │     cache what fits,
                        └───────────────────────┘     stream the rest from mmap
```

| Tier | System | Entry point | Target | Current status |
|------|--------|-------------|--------|----------------|
| **1. GPU** | GPU backend | not built | GPU present | ❌ Phase-6 planned (Vulkan partial offload; AMD Vega iGPU probe) |
| **2. Fast small** | `Engine` (`engine.rs`) + `layer_cache` | `leafcutter run ornith` | ≤9B class, fits RAM | ✅ coherent, ~3.3 GB RAM, 192 tests green |
| **3. Large streaming** | same `Engine`/loader in streaming mode | `validate_70b_memory.rs`, `LEAFCUTTER_NO_CACHE=1` | 70B+ on small RAM | ✅ 70B forward @ **1,145 MB peak**; MoE 35B streams @ **3,963 MB peak** (quantized expert slicing) |

Tiers 2 and 3 are **not two programs** — they are two modes of the same loader
and engine. The single adaptive cache (§4) is the bridge between them.

---

## 3. Dispatch Logic

Routing lives in `src/main.rs` and `Engine::load` (`src/inference/engine.rs`), with
the detection + tier brain in **`src/detect.rs`** (dependency-free, unit-tested).

- `leafcutter run <model>` (default `--engine auto`): model may be a name (fuzzy
  match against `models/`/`Downloads/models`) or a direct path. `src/detect.rs`
  probes the path — a `.gguf` file stays on the native Rust engine; a safetensors
  directory (`config.json` + `*.safetensors`) auto-routes to the reference Python
  backend; anything unrecognised warns but still attempts the native engine.
- The banner reports the **hardware snapshot** (cores, free RAM, GPU kind) and the
  **selected tier** from `detect::choose_tier`. Verified live:
  - 9B Q4_K → `Tier: 2 — fast CPU` (fits in free RAM)
  - 70B Q4_K → `Tier: 3 — streaming CPU` (too big)
  - `ornith safetensor` dir → routes to the Python reference backend
- `list-models` / `list-models --dir` now list **both** formats (GGUF files and
  safetensors directories, the latter marked `[safetensors]`).
- `--engine` still overrides: `native` (GGUF), `safetensor` (Python reference),
  `ollama` (HTTP API).
- **Tier 1 is the missing half:** `detect.rs` probes GPU kind (CUDA via
  `/dev/nvidia*`/`libcuda.so`, ROCm via `/dev/kfd`, Vulkan via
  `/dev/dri/renderD*` + `ldconfig -p`, Metal on macOS) and `LEAFCUTTER_PREFER_GPU=1`
  selects `Tier 1` in the decision — but no offload backend is wired yet. When one
  ships, it will expose `--gpu-layers` (mirroring llama.cpp), with CPU fallback when
  the model cannot fit in VRAM. A Vega-class iGPU (~2–4 GB shared VRAM) cannot hold
  a 5.6 GB Q4_K model, so partial offload of attention/FFN layers is the realistic
  ceiling on this hardware; CPU + AVX2 is expected to stay competitive for that class.
- **NPU is detected but not offloadable:** `detect.rs` also probes the NPU
  (`NpuKind` — Zhouyi AIPU via `/dev/aipu` + `/sys/class/misc/aipu`, reported as
  `npu:zhouyi-aipu` in the banner). `NpuKind::supports_dynamic_offload()` is always
  `false`, because Arm China Zhouyi AIPUs only execute **precompiled** `.aipu.bin`
  graphs — they cannot stream llama.cpp-style ops. Detection is honest reporting +
  future-proofing; it never upgrades the tier and never routes compute to the NPU.

### 3.1 ARM Quantized Kernel Dispatch (sdot / NEON) — 2026-08-20

Quantized decode matmuls live in `src/kernels/q8_k.rs` (block dots) and
`q8_k_gemm.rs` (row dispatch), wired from `q4_k_gemm.rs` / `q6_k_gemm.rs`.

- **Activation path**: the f32 activation row is quantized once to Q8_K
  (`build_aux8`), then every weight block is dotted against it.
- **Q4_K decode → single-column `sdot`**: on `is_aarch64_feature_detected!(
  "dotprod")` the dot uses the `sdot vd.4s, vn.16b, vm.16b` instruction
  (bit-exact vs scalar). A 2-column interleave variant exists but is NOT used:
  it loses to single-column on the CIX Sky1 (register pressure, 4.3–5.5 ms vs
  3.08 ms at n=12288). `LEAFCUTTER_Q8_GEMV=0` falls back to NEON.
- **Q6_K stays NEON**: sdot's asm barriers break memory pipelining on the
  bandwidth-bound lm_head (58 → 32 ms with NEON) and decode shapes. The lm_head
  uses the transposed-b Q6_K GEMV (≈24–26 GB/s, bandwidth-bound).
- **m>1 prefill**: `run_q4_k_q8_gemm` / `run_q6_k_q8_gemm` build each weight
  column's block buffers once and each activation row's Q8 once, then reuse them
  across all rows — prefill at m=77 dropped from 63 s to 7.5 s.
- **SVE2/i8mm (smmla) is NOT used**: the Sky1's SVE vector length is 128 bits
  (same as NEON); `smmla` yields the wrong dot pattern for single-column Q8 dots
  (272 vs 136) and only pays off via llama.cpp's 2-column zip-interleave, which
  loses on this chip. Dead end, documented in `CHANGELOG.md` [2026-08-20].

---

## 4. The Adaptive Loader (Colony Glue)

`src/model/loader.rs` — `GGUFModel` owns the memory policy.

### 4.1 Memory-bounded layer cache

- Budget is derived from `/proc/meminfo` `MemAvailable` via
  `available_memory_mb()`; default budget = **MemAvailable − 1 GiB**.
- `layer_cache_budget_bytes()` exposes the computed budget; `LEAFCUTTER_CACHE_MB`
  overrides it explicitly.
- `LayerCacheInner { map: HashMap<usize, Arc<HashMap<String, Tensor>>>, order: VecDeque<usize> }`
  holds cached layers with oldest-first eviction. `get_layer(idx)` fills from the
  mmap on a miss and evicts the oldest entry if the budget would be exceeded.
  `all_layers_cached()` reports whether the whole model is resident.
- `LEAFCUTTER_NO_CACHE=1` disables caching entirely (pure streaming, the original
  leafcutter behavior). Bit-exact equivalence between cache on/off was verified.

**Why this beats the references:** AirLLM loads exactly one layer at a time;
Colibri keeps a fixed-size LRU of experts. Both pick a memory budget up front and
never change it. Leafcutter measures the live budget and caches exactly what
fits — on a 32 GB machine a 70B model's 33 GB of Q4_K weights become nearly
fully resident; on a 4 GB machine the same model streams at bounded RSS.

### 4.2 lm_head gating

`load_lm_head_cache` (`engine.rs`) materializes `output.weight` for fast decoding.
It is gated on `model.model_fits_available_ram()` and `LEAFCUTTER_CACHE_HEAD=0`
opts out. For 70B this skips the 8.84 GB Q6_K head materialization entirely —
that was the single largest memory bug in the old always-cache path.

### 4.3 Page management

`drop_pages_from_cache` (`MADV_DONTNEED` on the whole mmap) is gated: it only runs
when the layer cache is enabled **and** `all_layers_cached()`, so a streaming
(model-too-big) run never evicts its own disk-backed pages. `LEAFCUTTER_DROP_PAGES=1`
re-enables the old aggressive behavior for experimentation. This was the fix for
the 16× regression where the entire 5.6 GB file was re-read from disk ~32×/token.

### 4.5 MoE expert streaming (no f32 materialization)

3-D MoE expert tensors (`gate_exps`/`up_exps`/`down_exps`, GGUF dims `[d0, d1, d2]`
with `d2` = number of experts, experts outermost) stay **quantized** in the layer
cache. `Tensor::expert_slice(e)` (`tensor.rs`) carves one expert's quantized
sub-matrix on demand (Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/IQ4_NL), and `moe.rs` slices only
the active top-k experts per token (a few MB each, freed each call). This replaced
the old eager per-layer f32 materialization (~1.07 GB × 3 per layer) that OOM-killed
`ornith-1.0-35b` at 6.5–7.1 GB — it now streams at **3,963 MB peak RSS**. `cached_bytes`
reports `resident_bytes()` (quantized blocks + materialized f32), so the layer-cache
budget is enforced against real residency.

### 4.6 Policy summary

| Env var | Meaning | Default |
|---------|---------|---------|
| `LEAFCUTTER_CACHE_MB` | explicit layer-cache budget in MiB | MemAvailable − 1 GiB |
| `LEAFCUTTER_NO_CACHE=1` | disable layer cache (pure streaming) | off |
| `LEAFCUTTER_CACHE_HEAD=0` | skip lm_head materialization | on (gated by RAM check) |
| `LEAFCUTTER_DROP_PAGES=1` | force MADV_DONTNEED after each layer | off (auto) |
| `LEAFCUTTER_DEBUG=1` | noisy loader/engine logs | off |
| `LEAFCUTTER_DEBUG_PROMPT` / `_NORMS` / `_LAYERS` | per-component debug | off |

---

## 5. Engines & Code Paths

`src/inference/`

| Module | Role |
|--------|------|
| `engine.rs` | The fast native GGUF engine. `forward_native`, `generate_native`, `generate_streaming_with_stops` (prefill + KV cache + per-token callback), `load_lm_head_cache`. Tier 2 workhorse; Tier 3 when cache is bounded/off. |
| `shard_engine.rs` | Legacy sharded-format engine (pre-GGUF). Reference/archival. |
| `deltanet.rs`, `ssm.rs`, `attention.rs`, `mla.rs`, `moe.rs`, `gemma.rs` | Architecture-specific forward modules (Qwen3.5 Gated DeltaNet, SSM, MLA, MoE, Gemma). |
| `sampler.rs`, `speculative.rs`, `anti_doom.rs` | Sampling, speculative decoding, context-rot hardening. |

`src/model/`

| Module | Role |
|--------|------|
| `gguf.rs` | GGUF v3 parser (metadata, tensor info, data blob). |
| `loader.rs` | `GGUFModel`: adaptive layer cache (§4), quant-aware loading, K-quant support (Q4_K/Q5_K/Q6_K/Q8_K/Q8_0/Q4_0/Q4_1/IQ4_NL/IQ4_XS). |
| `quant.rs` | `is_supported()` gate; unsupported types fail loudly (Q2_K/Q3_K/Q5_0/Q5_1/Q8_1/IQ-family beyond IQ4). |
| `tensor.rs` | `Tensor::matmul` → `matrixmultiply::sgemm` (BLAS-like) + SIMD small mats. |
| `arch.rs` | Architecture registry / tensor-name mapping. |

`src/tokenizer/` — `gguf_bpe.rs` reads `tokenizer.ggml.tokens` directly from the
GGUF metadata (no external tokenizer.json needed); HF fallback with vocab-mismatch
rejection. Byte-level UTF-8 streaming decode handles multi-byte emoji correctly.

`src/safetensor_backend.rs` — Python subprocess backend (`scripts/leafcutter_safetensor_run.py`).
Kept as the **reference-correct** path for hybrid models and for cross-checking
the native engine; it is intentionally slow (~12 s/tok) and is not a shipping path.

---

## 6. Current Measured State

Hardware: AMD Ryzen 7 5800HS (8C/16T, AVX2, CPU-only), 16 GB RAM.

| Scenario | Result |
|----------|--------|
| Ornith-1.0-9B Q4_K_M, `leafcutter run` | Coherent reasoning chat, **1.65 tok/s**, peak RAM **8.1 GB**, stops at `<\|im_end\|>` |
| Q4_K_M ↔ Q6_K | token-identical output (dequant correctness proven) |
| Ornith vs Ollama reference | tokens 1–2 match; token 3 diverges (residual numeric diff, not structural) |
| 70B forward, streaming (`LEAFCUTTER_NO_CACHE=1`) | **1,145 MB peak** (validated); top logits sane, no NaN |
| 70B Tier 3, adaptive cache | Coherent English (`"The capital of France is a city like no other,"`), peak **11.5 GB**, ~58 s/tok disk-bound |
| 70B lm_head, old path | 8.84 GB materialization (now gated off by §4.2) |
| lm_head fast tier | Q6_K block cache via `q6_k_matmul_transposed_b`, ~87.8 ms/tok, −3 GB vs f32 cache |
| Test suite | 202 passed / 0 failed / 4 ignored (release, lib) |

### 6.1 Orange Pi 6 Plus / CIX Sky1 (aarch64, 12 cores, 14 GiB) — 2026-08-20

| Scenario | Result |
|----------|--------|
| Ornith-1.0-9B Q4_K_M, 60-token generation | **~40 s wall (~1.5–1.7 tok/s under load; ≈2.4 tok/s idle)**, correct output |
| Prefill (m=77) | **7.5 s** (was 63 s) |
| Decode matmuls (Q4/Q6 K, sdot+NEON) | ≈ 320 ms/token |
| lm_head (Q6_K, bandwidth-bound) | ≈ 48 ms/token (24–26 GB/s) |
| delta_rule (NEON vectorized) | ≈ 30 ms/token (was 110 ms) |
| FFN SiLU (NEON fast-exp) | 0.45 ms/call (was 1.20 ms) |
| Ollama `ornith:9b` A/B (same load) | 2.93 tok/s decode — Leafcutter is ~1.95× behind, gap is kernel-level |
| DeepSeek V4 (165B, FP8, 156 GB) | Not runnable on 14 GiB RAM (~82 GB needed at Q4_K_M) — abandoned locally |

---

## 7. Correctness Notes (do-not-regress)

These were hard-won and are verified against llama.cpp / the safetensor reference.
Any engine change must keep them:

1. **A_log convention** — GGUF stores `ssm_a = -exp(A_log)` (pre-transformed at
   conversion); the engine multiplies it directly. The safetensors path stores
   raw `A_log` and applies `-exp()` itself. Do not unify these blindly.
2. **Norm weights bake in `+1`** — GGUF adds 1 to every norm weight at conversion
   (except `linear_attn.norm.weight`); runtime multiplies directly. A second `+1`
   was removed — it corrupted output.
3. **Conv1d layout** — GGUF is channel-major `[kernel_size, conv_dim]` flat
   `c*conv_k + k`, no transpose, tap `w[0]` = oldest.
4. **V-head pairing is interleaved** — `h_v = h_qk * r + v_idx` is wrong; it is
   `h_v % n_qk` (llama.cpp `ggml_repeat_4d`).
5. **Full-attention gate is sigmoid**, not SiLU.
6. **BF16 vs f32 drift** — recurrent models drift from the BF16 reference after
   ~3 layers in f32; expected, not a bug. Token 3 divergence vs Ollama is the
   residual symptom.

---

## 8. Roadmap / Missing Pieces

- [x] **Colony dispatch brain** — `src/detect.rs`: hardware probe (CPU/RAM/GPU),
      model probe (GGUF vs safetensors), tier decision; wired into `run` + `list`.
- [x] **Persistent model sources** — `src/config.rs`: OS-aware config file
      (`~/.config/leafcutter/config.json`, `%APPDATA%`, `~/Library/Application
      Support`), `leafcutter source add|remove|list` CLI + `/source` REPL command;
      `resolve_models_dirs()` now cwd-independent (no more `cd` required).
- [x] **OS/arch detection** — `detect::current_os()`/`current_arch()` shown in the
      run banner; basis for the cross-platform installer/container story.
- [x] **GGUF chat-template preference** — `cmd_run` now uses
      `apply_chat_template_from_gguf()` when the GGUF carries
      `tokenizer.chat_template` (fixes Ministral-2512's `[SYSTEM_PROMPT]` format
      that the hardcoded profile templates got wrong). See `NEXT_STEPS.md`.
- [ ] **RoPE-YaRN support** — Ministral-3-3B-Instruct-2512, Llama-3.x-1M, and
      other long-context models use YaRN (`factor=16`, `beta_fast=32`,
      `beta_slow=1`, `mscale=1`). Engine currently treats them as standard
      RoPE → forward pass produces garbage. Implementation reference:
      llama.cpp `ggml_compute_forward_rope_yarn()` + `mscale` attention scaling.
      **Status:** BLOCKING Ministral-2512 correctness.
- [x] **Tier 3 coherence proof** — Llama-3.3-70B through the adaptive streaming
      path generated coherent English (`"The capital of France is a city like no
      other,"`) at **11.5 GB peak RSS** (bounded cache; the 42 GB model never went
      resident). ~58 s/tok disk-bound on this hardware — the correctness question
      is settled; speed is the remaining work.
- [ ] **Tier 1 GPU offload** — probe exists (`detect::GpuKind`, `LEAFCUTTER_PREFER_GPU`);
      wiring `--gpu-layers` partial offload (Vulkan via the existing `backend/wgpu.rs`,
      ROCm, CUDA) is the remaining work.
- [ ] **Prompt prefill in the streaming chat path** — `chat`/`run` must not throw
      away all but the last prompt token (correctness gap, not just perf).
- [ ] **Fused dequant-GEMM** — dequantize inside the SIMD dot inner loop instead
      of full-column-then-dot.
- [ ] **Top-K preselection for lm_head** — min-heap over the top-K rows; 248K
      dequants/token → ~200 (1000× on the head step).
- [ ] **Zero-copy `load_layer`** — mmap slices instead of parsing per call.
- [ ] **Distributed inference** (post-colony milestone).

---

## 9. Reference Files

- Colony doc: `ARCHITECTURE.md` (this file)
- Competitive research: `docs/research/airllm_vs_colibri.txt`
- Historical (removed 2026-08-17): `strategy.md` (safetensors streaming),
  `LEAFCUTTER_STRATEGY.md` (GGUF integration), `COLIBRI_ANALYSIS.md`,
  `docs/architecture/*`, `docs/research/KIMI_K3_IN_C_ANALYSIS.md` (adopted
  techniques are implemented in `rust/src/`; see CHANGELOG 2026-08-17)
- External references (do not modify): `/home/xander/Documents/portfolio/leafcutter_max/airllm`,
  `/home/xander/Documents/portfolio/leafcutter_max/colibri`,
  `/home/xander/Documents/portfolio/leafcutter_max/llama.cpp`
- Target 70B model: `/home/xander/Downloads/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf`
