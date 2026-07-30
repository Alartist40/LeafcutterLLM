# LeafcutterLLM Strategy — Real Goals, Measured Targets

**Date:** 2026-07-22
**Status:** Strategy rewrite after ground-truth measurement.
**Last updated:** 2026-07-24 (Phase 2 prefetch + anti-doom ship)
**Source path:** `/home/xander/Documents/portfolio/LeafcutterLLM/`

---

## 0. Critical Correction From Previous Version

Earlier draft (now superseded by this file) framed the goal as:
> *"Run 70B on 16 GB RAM with streaming weights."*

That was a **regression from what Leafcutter already does.** Ground-truth
measurement today on this machine:

| Metric | Value | Source |
|---|---|---|
| **70B Llama-3.3 Q4_K_M peak RSS** | **~1.08 GB** | Measured live, 2026-07-22 |
| Previously validated | 1,145 MB peak | `FRONTIER_MODELS_PLAN.md` |
| Model file size on disk | 42.5 GB | `ls /home/xander/Downloads/models/` |
| Layers | 80 | engine log |
| Hidden size | 8192 | engine log |
| Tokens per second (decode) | ~0.01 tok/s (1 token in ~92 s) | measured |
| Output correctness | yes — semantic match ("The capital…") | measured |

We **already run 70B dense at 1.1 GB peak.** AirLLM's documented floor is 4 GB
on the same model class, so we already beat it by ~4×. Colibri's 25 GB floor
is real but it targets an 800× larger model.

**The actual goal is not "70B on 16 GB."** It's:
1. Keep 70B dense at ~1 GB peak while improving throughput
2. Get frontier MoE (Kimi K2.6, GLM-5.2) running at the same efficient-envelope
   scale on Pi 5 8GB (~3 GB peak target)
3. Don't regress the 1.1 GB hard floor on existing 9B dense

---

## 1. The Goal Stack (Real, not made up)

| Priority | Goal | Current State | Target | When |
|---|---|---|---|---|
| **G0** | Don't regress 9B dense | 1.2 GB peak | stay < 1.5 GB | now |
| **G1** | 70B dense | 1.08 GB peak | keep < 2 GB | now (✓) |
| **G2** | 70B faster (CPU throughput) | 0.01 tok/s cold; 1.24 tok/s @ 3B warm with prefetch | 0.5–1 tok/s | 2–4 weeks |
| **G2a** | Smarter inference (anti-doom) | ✅ shipped `LEAFCUTTER_ANTIDOOM=1` (commit aaec49d) | 100% loop suppression on Ministral-3B greedy | DONE |
| **G3** | MoE streaming for Kimi K2.6 / GLM-5.2 | not yet | ~3 GB peak (Pi 5 8GB) | 4–8 weeks |
| **G4** | MLA cached conversation resume | partial | full `.kv` persist | 1–2 weeks |

The MoE frontier work is the main substrate of effort this quarter. Throughput
on existing workloads is the polishing layer on top of it.

---

## 2. Verified Inventory — What We Already Have

| Module | Lines | What it gives us |
|---|---|---|
| `model/gguf.rs` | 836 | `drop_pages_from_cache()` with `madvise(MADV_DONTNEED)` — bound-RSS streaming |
| `shard/loader.rs` | 385 | mmap-based loader with layer cache + prefetch slot |
| `model/loader.rs` | 1068 | layer-by-layer forward wiring |
| `inference/engine.rs` | 1321 | main inference loop |
| `inference/attention.rs` | 414 | standard MHA/GQA |
| `inference/deltanet.rs` | 536 | Qwen3.5 hybrid SSM |
| `inference/ssm.rs` | 446 | legacy Mamba |
| `inference/mla.rs` | **432** | MLA with kv_lat compression |
| `inference/moe.rs` | **360** | MoE forward path |
| `inference/speculative.rs` | 189 | speculative decoding |
| `inference/shard_engine.rs` | 441 | shard-aware engine |
| `inference/sampler.rs` | 70 | sampling |
| `cache/mod.rs` | 180 | KV cache abstraction |
| `cache/deltanet_state.rs` | 62 | SSM state |
| `cache/ssm_state.rs` | 62 | Mamba state |
| `shard/format.rs` | 235 | shard binary format |
| `shard/writer.rs` | 548 | shard writer |

**Things we DO NOT need from Colibri** because we already have them:
- Layer streaming from disk: ✓ (`madvise(MADV_DONTNEED)`)
- MLA compressed KV: ✓ (`mla.rs`, 432 lines)
- MoE forward path: ✓ (`moe.rs`, 360 lines)
- Speculative decoding: ✓ (`speculative.rs`)
- Shard format + cache capacity control: ✓

---

## 3. What We DO Need From Colibri (Lessons, Not Code)

Filtered against actual Leafcutter state. Colibri's lessons ranked by
incremental value, not novelty.

### Lesson A — LFRU cache beats FIFO

**Status:** SHIPPED (2026-07-23). `LfruCache` is ported from Colibri's
`tier.h`, behind the `LEAFCUTTER_CACHE=lfru` env var. Default is still FIFO.

**Measured delta:** Across 3 synthetic benchmarks (sequential, strided,
random access patterns), LFRU averages **+9.1% tok/s** vs FIFO with no
regression in any tested case:

- sequential (8 layers, slots=2): +10.1% (35.3% hit rate)
- random (16 layers, slots=4): +19.1% (30.9% hit rate)
- strided (8 layers, slots=1): -2.0% (0.9% hit rate — algorithm can't
  hold the right window with one slot)

Hits the lesson's sweet spot: non-uniform access patterns like MoE routing
where some layers are warm and others cold — LFRU lets frequency dominate
so a recently-cold layer can't evict a long-term-hot one.

### Lesson B — Expert streaming for MoE (the real frontier work)

**Status:** Not yet implemented (per `FRONTIER_MODELS_PLAN.md`, planned for M12
milestone).

**Value:** Highest. Currently 384-expert MoE models (Kimi K2.6) need all experts
resident per layer. We can stream only the top-k routed experts per layer.

**Math:** For Kimi K2.6 (2048-dim experts):
- Per-expert weights: 2048 × 3 (gate/up/down) × 7168 (hidden) × ~0.5B (Q4) ≈ 22 MB
- Per-layer expert pool: 384 × 22 MB = 8.4 GB (impossible on Pi 5 8GB)
- Top-k = 8 routed per token: 8 × 22 MB = 176 MB per layer (fits!)
- 60 MoE layers resident at once: 60 × 176 MB = 10.5 GB

So top-k-8 per-layer loads match the ~3 GB goal IF we also have layer-level
streaming (only one MoE layer active at a time?). With LRU caching of recent
routing decisions across layers, achievable.

**Effort:** HIGH — full expert-level routing + per-expert I/O + LRU per-layer.
Plan in `FRONTIER_MODELS_PLAN.md` already; the strategy is "do it."

### Lesson C — Pivot the cache between prefill and decode

**Status:** Not implemented. Current engine does cache once for entire
forward pass.

**Value:** During prefill, batching amortizes everything; cache thrashing
costs much more than during decode's single-token loop.

**Effort:** LOW. Add a `prefill_end()` hook that refills the cache with
the most-recently-used layer pair once prefill completes.

### Lesson D — O_DIRECT for direct-to-SSD reads

**Status:** Not tested. Colibri measured +34% decode on some NVMe drives.

**Value:** Drive-dependent. Some drives don't benefit. Could regress.

**Effort:** LOW. Add `O_DIRECT` flag behind env var, measure on real hardware.

### Lesson E — Speculative MTP head (Colibri's biggest win)

**Status:** We have `speculative.rs` (189 lines) but it's not wired for
GLM-5.2's `nextn.*` MTP tensors.

**Value:** For GLM-5.2-style MTP, 2.2–2.8× throughput claimed.

**Effort:** HIGH — MTP head wiring, draft/verify protocol, draft-accept rate
measurement.

### Lesson F — KV persistence across restarts (`.kv` files)

**Status:** We have `kv_persist.h` in Colibri for inspiration; our equivalent
is in the cache module only.

**Value:** Real for long conversations. Compressed MLA KV means storage cost
is ~4.6 KB/token — 100k tokens = 460 MB disk. Decent for power-user cache.

**Effort:** MEDIUM. Persist MLA compressed KV on shutdown, reload on
session start if model hash matches.

### Lesson G — CUDA / Metal backends

**Status:** Out of scope for this hardware (no GPU idle for inference on
this box).

**Value:** N/A right now. Defer until a CUDA-capable setup exists.

---

## 4. Throughput Plan (G2): Make 70B Faster

Currently 70B at 0.01 tok/s. Target 0.5–1 tok/s eventually. Where the time
goes on the current path (1 tok ≈ 92 s):

- mmap + madvise roundtrip per layer: ~1 ms (small, OS page cache)
- Dequantize Q4_K → f32 per layer: ~tens of ms
- AVX2 matmul per layer: ~hundreds of ms
- KV cache attention per layer: tens of ms
- Sampler: negligible

The 92 s for 80 layers = ~1.15 s per layer. Most is the matmul + dequant
over f32 activations. Possible wins (in priority order):

1. **Thread pool sizing** (already shipped at HEAD, capped ≈ 7 threads).
   Verify 70B benefits the same way 9B did. (Warning: on 16 GB laptop, RAM
   pressure is binding; don't crank threads spuriously.)
2. **Async pipeline prefetch**: layer N+1 loads while layer N computes. We
   already have `prefetch: Arc<Mutex<Option<...>>>` single-slot; extend to
   multi-slot or a separate `std::thread` worker.
3. **Persistent KV in RAM v2**: avoid reallocating KV buffer per forward.
4. **Reduced CPU thread count during decode** (1.5 GB working set, 7 threads
   thrash cache): auto-reduce to 2–3 on 70B and see if total throughput goes up.

For G2 specifically: **defer the heavy matmul work** until we know the
streaming-pipeline is solid. A 2× pipeline speedup comes for free from
async prefetch; a 2× matmul speedup is the kind of AVX2 work that we
already exhausted in the prior session.

---

## 5. What We DON'T Need (avoids wasted work this session)

Recall: prior session successfully tried and rejected dequant cache,
Q8_0 scalar, and Q8_0 AVX2 maddubs. Re-list to prevent reruns.

- ❌ Dequant-per-call caching (verified 0% net)
- ❌ Q8_0 activation quantification scalar (70% slower)
- ❌ Q8_0 AVX2 maddubs (36% slower than f32 FMA)

Also: don't copy Colibri's single-file C architecture. We stay modular Rust.

---

## 6. What I Got Wrong (honest pushback on the prior strategy doc)

The prior `LEAFCUTTER_STRATEGY.md` (now superseded) claimed:

| Claimed | Reality |
|---|---|
| "Bottleneck for large models on small hardware is matmul FLOPs" | Bottleneck is I/O and dequant, not matmul |
| "Need pread-based streaming as Phase 1" | We have mmap + MADV_DONTNEED; that already works |
| "Need LRU/LFRU cache as Phase 1" | We have FIFO; helpful but Phase 1 is not "add it" |
| "Need MLA compressed KV as Phase 2" | Already shipped (432 lines) |
| "Need MoE expert streaming as Phase 3" | Frontier-MoE work but not "phase 3" — already partially built (moe.rs) |
| "70B on 16 GB RAM target" | 1.08 GB peak today — wrong framing |
| "Colibri-style pread+posix_fadvise+DONTNEED" | We use mmap+MADV_DONTNEED — different syscall, equivalent effect |
| "Make the engine I/O-bound instead of compute-bound" | False dichotomy. 70B at 92 s/tok is both I/O (load) AND compute (1.15 s/layer matmul) |

### What the corrected strategy says

- **Phase 1 (NOW):** LFRU cache replacement for FIFO in `shard/loader.rs`.
  Port Colibri's tier.h LFRU. Bench. Don't ship if hit-rate improvement <10%.
- **Phase 2A (skip, pre-flight failed 2026-07-23):** io_uring backend
  was the original Phase 2. Pre-flight measurement contradicts the
  premise. On 5800HS / NVMe / 16 GB hardware, mmap+MADV_DONTNEED is
  **1.8× faster per layer** than pread-dontneed. The OS page cache
  shortcircuits the disk read; our I/O fraction of 70B decode time
  is <0.06%. Colibri's io_uring wins on 25 GB RAM with 372 GB models
  where every read is cold from disk; our profile is different. Skip.
  See "Phase 2A finding" in COLIBRI_ANALYSIS.md §5.
- **Phase 2B (parallelizable, 1 week):** O_DIRECT for drives that
  benefit (gated `LEAFCUTTER_IO_DIRECT=1`); bench per-disk before/after.
- **Phase 3 (4–8 weeks):** MoE expert streaming for Kimi K2.6 / GLM-5.2.
  Top-k-of-N expert pager with per-layer LRU.
- **Phase 4 (G4, parallel):** MLA KV persist with hash-keyed cache files.
- **Phase 5 (deferred):** MTP speculative decode. Triggered by real-model
  need, not preemptively.

### What we measure

| Benchmark | Tool | Notes |
|---|---|---|
| Peak RSS at 70B decode | `test_generation` + RSS sampling | baseline 1.08 GB |
| 70B decode tok/s | `time test_generation --tokens N` | baseline ~0.01 tok/s |
| Cache hit-rate (FIFO vs LFRU) | inline counters in cache | target LFRU +10% vs FIFO |
| 9B regression check | re-run E2E 5/5 | must stay green |
| MLA KV persist roundtrip | synthetic test save/load | small test |
| MoE expert hit-rate | inline counters | target 60%+ after warmup |

---

## 7. The One Real Win Standing Out: MoE Streaming

If we do **only one** thing this quarter from this doc, it should be
MoE expert streaming (Phase 3). Why:
- Frontier models (Kimi K2.6, GLM-5.2) go from "can't fit on Pi 5"
  to "runs with ~3 GB peak."
- Required for the 80% of frontier-or-near-frontier open-weight models
  that use MoE.
- Builds on what already exists (`moe.rs`, mmap streaming, MLA module).
- Direct port of Colibri's `expert_load_impl()` and `pilot_prefetch()`
  patterns; less risky than the AVX2 dequant work because the I/O
  subsystem is clearly correct already.

Second priority right behind it: **io_uring backend for the shard
loader** (see COLIBRI_ANALYSIS.md §5.1). Linux-only, gated behind
`LEAFCUTTER_IO=uring`. The "Doubles throughput with little engineering"
claim that previously lived here was wrong: a single-slot prefetch
already overlaps most of the mmap-fault latency, so adding more
slots doesn't 2× anything.

Everything else (LFRU, KV persistence, MTP) is incremental
polish, not new capability.












































Objective
- Fix the streaming native Rust ornith pipeline (streaming_ornith.rs) to produce coherent English instead of garbage, matching the Python safetensor-backend reference that outputs "Paris" for "The capital of France is".
Important Details
- Model: Qwen3.5 (Ornith-1.0-9B): hidden=4096, vocab=248320, 32 layers (24 linear_attention + 8 full_attention), head_dim=256, n_heads=16, n_kv=4, intermediate=12288, linear_num_key_heads=16, linear_num_value_heads=32, head_k_dim=128, head_v_dim=128, conv_kernel=4
- Streaming engine reads one layer at a time from safetensors (~400MB peak vs 18GB full load)
- Token 11751 (" Paris") has logit -0.455; top-1 is token 96284 ("取") at logit 4.723
- Python reference (leafcutter run --engine safetensor) works correctly, confirming weights/tokenizer are fine
- A_log values vary by layer: some negative (Layer 0: mean -3.4), some positive (Layer 1: mean ~1.4) — code uses -exp(A_log) which gives correct negative A for both
- Conv1d weight shape [8192, 1, 4] in safetensors; tap 3 mean_abs=0.029, tap 0 mean_abs=0.007 — both attenuate signal ~20-100x vs raw QKV
- All tensor shapes verified match code expectations; q_proj.weight is [8192, 4096] = [2h, h] (includes output gate)
Work State
Completed
- Fixed borrow checker in forward_one_token, deltanet_forward, attention_forward, mlp_forward by cloning self.cfg fields to locals
- Deleted dead files: ornith_forward.rs, safetensor_tensors.rs, engine_keymap.rs; removed their pub mod lines from lib.rs
- Removed dead code: scale_factor line, final_norm lookup, lm_head_transposed variable
- Chunked lm_head reading (1024-row chunks instead of loading all 4GB)
- Wired --engine native CLI handler in main.rs (uses StreamingOrnith directly)
- Fixed critical residual bug: second residual was adding to residual (original hidden) instead of current hidden
- Added DeltaNetStateCache and KV cache to StreamingOrnith for cross-token state
- Updated deltanet_forward with full delta rule: S = decay*S + beta*(v - S@k)⊗k, output S@q
- Updated attention_forward with proper attention over cached K,V (q_norm, k_norm, output gate, softmax over all positions)
- Added conv1d with proper state caching (tap ordering: safetensors tap conv_k-1-k maps to GGUF weightk = current input)
- Test now processes all 5 prompt tokens sequentially to build state before prediction
- Verified all tensor shapes against safetensors metadata via Python
Active
- Output still wrong despite full state accumulation: top-1 is Chinese "取" (token 96284), " Paris" (token 11751) has logit -0.455
- The conv1d output is ~0.0029 mean_abs before SiLU (vs raw QKV ~0.10), suggesting conv weights are too small or indexing is wrong
- Without conv (raw QKV+SiLU): DeltaNet OUT mean_abs=0.06, hidden mean_abs=0.20; still produces wrong prediction
- The delta rule output chain: QKV(~0.10) → conv/tap(~0.03) → delta_output(6e-5) → per-head-norm(→ 1.0) → z-gate(→ ~0.07) → out_proj(→ 3.3) — but actual observed output is 0.06, suggesting a missing factor in the computation chain
Blocked
- Conv1d weights are very small (mean_abs tap_3=0.029), making conv output 20x smaller than raw QKV; unclear if this is expected or indexing is wrong
- Reference code applies SiLU only after conv, not without conv — our no-conv path always applies SiLU, which may be incorrect
- The delta rule output magnitude is much smaller than expected from theoretical computation, suggesting a bug in the normalization, z-gate, or output projection steps
Next Move
1. Test raw QKV with NO SiLU (matching reference behavior when conv is absent) to see if SiLU alone is the culprit
2. If still wrong, add debug prints for intermediate values (Q after norm, K after norm, qk_dot, beta, v_pred, delta_output per head) to find the missing factor
3. Once correct output is achieved with raw QKV, re-enable conv1d with empirical verification of tap ordering
Relevant Files
- rust/src/streaming_ornith.rs: Main streaming forward pass — all fixes applied here
- rust/src/inference/deltanet.rs (561 lines): Reference DeltaNet implementation (GGUF engine, proven correct)
- rust/src/cache/deltanet_state.rs: DeltaNet matrix state cache + conv state cache
- rust/src/ornith_config.rs: Config parser for Qwen3.5 hybrid model
- rust/src/bin/test_streaming_forward.rs: Test binary processing all prompt tokens sequentially
- /home/xander/Documents/portfolio/LeafcutterLLM/strategy.md: Complete build/debug guide with tensor shapes and known failure modes

