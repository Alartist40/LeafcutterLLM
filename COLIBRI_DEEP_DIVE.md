# Colibrì Deep-Dive — Source-Level Analysis for LeafcutterLLM

**Date:** 2026-08-02
**Source path:** `/home/xander/Documents/portfolio/leafcutter_max/colibri/c/`
**Scope:** Everything in the `c/` tree, read at source level, distilled to
"what Leafcutter can surpass" and "what to adopt". Companion to
`COLIBRI_ANALYSIS.md` (2026-07-22, README-level); this is the code-level pass.

---

## 1. Source Inventory (what each file actually is)

| File | Lines | Role | Read? |
|---|---|---|---|
| `colibri.c` | 7,342 | Main engine: model, KV, MoE, decode, prefetch, server | ✅ full pipeline |
| `quant.h` | ~1,461 | Quant kernels: int8/int4/int3/int2/i3g64/E8, VNNI, dot8 | ✅ kernels |
| `st.h` | ~? | safetensors loader (preda, no mmap) | ⏳ partial |
| `decode_batch.h` | 47 | Server `SUBMIT` protocol framing | ✅ |
| `tier.h` | 60 | LFRU cache replacement (heat + recency) | ✅ |
| `kv_persist.h` | 121 | `.coli_kv` crash-safe KV cache persistence | ✅ |
| `uring.h` | 137 | Minimal Linux io_uring reader | ✅ |
| `backend_loader.c` | ? | Weight-format loader (latent/rope/mla params) | ⏳ partial |
| `backend_cuda.cu` | 1,594 | CUDA backend, PIPE2 layer residency | ⏳ partial |
| `backend_metal.mm` | 1,039 | Metal backend, full-layer compute block | ⏳ partial |
| `rans.h` | 1,151 | rANS entropy coding (state-table arithmetic) | ⏳ not read |
| `fse_coli.h` | ? | FSE (finite-state entropy) | ⏳ not read |
| `inkling.c` | 1,729 | **Separate** engine for Thinking-Machines "Inkling" | ✅ header+arch |
| `sample.h`, `grammar.h`, `schema_gbnf.h` | ? | Sampling / GBNF grammar | ⏳ not read |

**Key finding:** `inkling.c` is NOT a Colibri variant. It is a second, nearly
independent pure-C engine targeting a *different* model family (hybrid
sliding-window/global GQA, learned relative-position bias, depthwise causal
short convs on K/V, sigmoid router + loss-free bias). Colibrì itself is a
GLM-5.2 MoE-DSA engine; `inkling.c` shares only the "dense resident + routed
experts streamed from disk, LRU-cached" staging principle. Both engines
validate against a HF oracle (`ref_*.json`) before scaling.

---

## 2. Quantization Subsystem (`quant.h`)

The engine ships **six distinct weight formats**, selected per-tensor:

| fmt | Bits/weight | Layout | Encode line |
|---|---|---|---|
| f32 (bits=0) | 32 | plain float — bit-exact oracle mode | `matmul` |
| int8 per-row | 8 | `int8_t q[I]` + `scale[O]` per-row | `matmul_q` |
| int4 | 4 | packed 2/byte + `scale[O]` per-row | `matmul_i4` |
| int4 grouped (fmt=4) | 4 | + per-*group* scales | `matmul_i4_grouped` |
| int2 | 2 | packed 4/byte + `scale[O]` | `matmul_i2` |
| **i3-g64 (fmt=5)** | 3 | **one f32 scale per 64-input group**; 16B low plane (2 bits/val) + 8B high plane (1 bit/val) | `matmul_i3` |
| **E8/IQ3 (fmt=6)** | **3.0625** | **98 bytes / 256 weights** E8-lattice container | `e8_expand_sub` |

### 2.1 The E8 lattice — the single most interesting quant trick

```
E8_QK   = 256    // weights per super-block
E8_SUB  = 32     // weights per sign/scale word (sub-block)
E8_BBYTES = 98   // bytes per super-block  →  98/32 = 3.0625 bits/weight
```

Per super-block: a global fp16 scale `d` (`e8_fp16_to_f32`), then 32-weight
sub-blocks expand through a **256-entry 4-vector codebook table**
(`e8_grid[256][4]`) with a 32-bit sign/scale word selecting the lattice
codeword per weight. Decode expands one 32-weight sub-block into a stack
buffer then FMA's it against activations (`e8_expand_sub`), explicitly to
avoid per-weight table lookups dominating. There is an fp16→f32 table
(`e8_fp16_to_f32`) and a fast-Walsh-Hadamard transform (`e8_fwht`) + rotation
path (`e8_rot_rows`) — the IQ3 family's Hadamard-rotated lattice construction.
Verified against a Python codec by `tests/test_e8_kernel.c`.

**Leafcutter relevance:** a 3.06 bit/weight format is a **real, validated
win for our Tier-3 streaming tier** — it cuts disk traffic ~25% vs our Q4.
We don't need the lattice codebook; we could adopt the *container* (98 bytes
per 256 weights, expand-per-sub-block) at GGUF packing time, or simply
note that sub-4-bit formats at this quality exist.

### 2.2 VNNI + dot8 paths

- `dot_i8i8` / `dot_i4i8` / `dot_i2x8` families with NEON (`int32x4_t`
  accumulators, 16-lane loads) and AVX-512 variants.
- `IDOT_KERNEL "avx512-vnni"` / `"avx-vnni"` — the **integer dot-product**
  kernel (int4 weight × int8 activation accumulate in VNNI), not the fp16
  path.
- `qrow_i8` — quantize an activation row to int8 inline (feeds the i8·i8 dot).
- `axpy_i4f_avx512` — int4-dequant-axpy twin for the matmul-accumulate path.

### 2.3 The i3-g64 ablation claim (colibri.c #132)

> int3-g64 **beat per-row int4** on an OLMoE ablation. One f32 scale per
> 64-input group with a 2+1 plane split, packed as `lo` (2 bits/val, int2
> layout) + `hi` (1 bit/val).

This is the same claim our own memory notes flagged. Take with a grain of salt
(one ablation, self-reported) but it anchors why they bothered with a third
sub-4-bit path.

---

## 3. The MoE Pipeline (`moe()`, colibri.c:2838)

The engine is a **GLM-5.2 MoE-DSA** (sparse attention) engine. `moe()` is split
into two phases around the async expert loader:

### FASE A — routing (all `S` positions)
- Gate network selects top-K per position, produces routing weights `ws[]`.
- Optional **Metal pre-routed path** (`g_pre_idx`): the router runs on GPU,
  `moe()` skips FASE A and consumes the GPU's chosen indices.
- Bumps per-expert heat (`eheat`/`elast`/`eusage`) for the LFRU cache, with a
  careful `touched[]` guard so a multi-position batch snapshots recency
  **before** the call, not after an earlier position's bump inside the same
  call (prevents same-call contamination).

### FASE B — union of the batch's experts
- Builds the unique expert set across all `S` positions (`seen[E]` + `uniq[]`).
- **EXPERT_BUDGET** (decode-only, `S<=4`): caps distinct experts per layer to
  bound disk I/O on cold/low-RAM hosts. **Miss-aware**: cache hits are always
  kept (they're free), only misses are dropped, and from the misses it keeps
  the highest **aggregate gate weight** up to the budget. Cites
  MoE-Spec (arXiv 2602.16052): top-32 of 64 capture 93% of routing weight.
- Critical safety: budget is **disabled for prefill** (`S=prompt_len`), where
  union `nu` can be 30–100+ experts; capping there corrupts hidden state →
  wrong KV → repetitive garbage. Documented as bug #292 (woolcoxm).

**Leafcutter relevance:** the "never let a position reach zero routed
experts" invariant + "budget only decode, never prefill" rule is exactly the
kind of guard we'd want if we ever add MoE streaming.

---

## 4. Memory Management (tier.h + expert loader)

### 4.1 LFRU (tier.h:60) — already shipped in Leafcutter

```c
score = (heat << 8) | recency        // recency capped at 255
cold  = min score among pinned      // frequency dominates
hot   = max score among non-resident
swap  iff hot > cold + cold/4 + 4    // 25% + 4 hysteresis
heat  >>= 1 every decay tick         // exponential half-life
```

The **LFRU hybrid** (frequency primary, recency breaks close calls) was the
first thing we adopted — it's the `LEAFCUTTER_CACHE=lfru` option. Their
hysteresis rule (25% + small constant margin) prevents cache ping-pong on tiny
samples. We already replicate this in `src/model/loader.rs`.

### 4.2 The tiered expert residency model

- **Dense part** (attention, shared experts, embeddings): always resident, int4
  (~9.9 GB for GLM-5.2).
- **19,456 routed experts** (~19 MB each int4): on disk, streamed per-expert
  with pread + `posix_fadvise(DONTNEED)` (`st.h` header: "legge con pread
  (niente mmap) + posix_fadvise(DONTNEED)").
- **pin[]** (learned hot-store, persisted via `.coli_usage`) + **ecache[]**
  (LRU per layer) — `expert_load_impl()` classifies hit/miss and does
  async-load with a classify-and-read snapshot.
- **PILOT** cross-layer prefetch: router predicts next layer's experts
  (claimed 71.6% one-layer-ahead) and pre-loads while current layer computes.
  A later "couple" prefetch (`couple_prefetch`) uses cross-layer co-occurrence
  pairs. Guarded by `g_pilot`, an in-flight barrier per layer
  (`g_pilot_inflight[layer]` + condvar) so prefetched loads don't race the
  real forward.
- **Disk-class heuristic** (`dc_needed`, `dc_wall_ns[2]`): busy-wall sampling
  to decide *whether* a load needs the expensive pre-bump recency snapshot.

**Leafcutter relevance:** our adaptive RAM-budget loader (Tier 3, 70B proven
coherent at 11.5 GB peak) is the dense-model analogue. The MoE-specific
inventions we'd only need if/when we stream experts:
1. **prefetch-consume barriers** (PILOT) — the race-free handoff protocol,
2. **miss-aware budget** (FASE B) — bounded disk I/O without dropping hits.

### 4.3 io_uring (uring.h:137)

Hand-rolled Linux io_uring: single-owner thread, batch of positioned reads,
`IOSQE_ASYNC` forced (so cold regular-file reads go to io-wq instead of
serializing on the submitter during `io_uring_enter`), `MAP_POPULATE` on the
rings, acquire/release atomics on head/tail. Bounded worker pool registered
via `IORING_REGISTER_IOWQ_MAX_WORKERS`.

**Leafcutter relevance:** our own Phase-2A measurement (bench_measure_io.rs)
proved mmap+DONTNEED already beats pread on this hardware for *dense* layer
streaming — io_uring would be engineering theater for us. **But** for the
MoE read-once-drop expert pattern (no mmap, each expert read exactly once),
batch-io_uring is the right tool. Stays on the shelf until MoE lands.

---

## 5. KV Persistence (kv_persist.h) — feature we should steal

`.coli_kv` writes the **compressed MLA KV-cache** (kv_lora + qk_rope floats
per token, plus DSA index vectors) to disk **incrementally after every turn**:

```
Header:  MAGIC "COLIKV1\0" + {n_layers, kv_lora, qk_rope, index_hd, nic, vocab, nrec}
Record:  token_id + n_layers*(kv_lora + qk_rope)*4  (+ index_hd*4 if DSA)
```

- Crash-safe: `nrec` (record count) is rewritten **last** after each append +
  fflush, so a torn write is detected on load and the conversation is rolled
  back to the last complete record.
- On restart: `kv_disk_load` verifies MAGIC + model-shape header match (else
  "ignoring .coli_kv from a different model"), reads `nrec`, resumes the
  conversation with **zero re-prefill** ("resumed conversation from disk: %d
  tokens in %.1fs").
- `g_kvsave=0` disables. Context overflow → truncates and starts over.

**Leafcutter relevance:** HIGH value, LOW effort for us. Our Tier-2/Tier-3
engines recompute full prefill every restart. A simple append-only KV
snapshot (token ids + per-token cache state) keyed by model+shape would give
"warm conversation reopen" with zero re-prefill — a genuinely differentiating
feature AirLLM does not have. Our `src/model/loader.rs` already has a KV
cache object we could serialize behind a feature flag.

---

## 6. DSA — DeepSeek-Sparse-Attention indexer

Colibri has `m->Ic[i]` per-layer **index vectors** (`index_hd` dim), a
lightning indexer that selects top-K keys for long-sequence attention
(skips attending over the full key cache). Persisted in `.coli_kv` alongside
the MLA cache; gated by `m->has_dsa`. This is the architecture-side answer to
"1M context" — not just MLA compression (576 floats/token vs 32,768) but also
**not attending to every key**. We don't implement DSA; noted as the model
architecture's feature, not the engine's.

---

## 7. The Full-Layer Forward (`layer_forward_rows`, colibri.c:4244)

Order per layer:
```
rmsnorm(in_ln) → attention_rows (MLA) → residual add → rmsnorm(post_ln)
→ moe (sparse) or dense_mlp → residual add
```
- **Metal path:** attention + shared expert + router run as a full-layer GPU
  compute block; the CPU does expert resolve + disk. `COLI_METAL_GEMM_MIN`
  gates which GEMMs go to GPU (min rows heuristic).
- **CUDA path (PIPE2):** layer *residency* on device — pinned weights stream
  ahead of compute so a layer is GPU-resident when the forward reaches it.
- **Prefetch flags:** `pilot_prefetch` (router lookahead, S<=8) and
  `la_predict` (lookahead router prediction).

---

## 8. What To Surpass vs What To Adopt (the whole point)

### Adopt (actionable, ordered by value)

| # | Feature | Effort | Where in Leafcutter |
|---|---|---|---|
| 1 | **KV persistence / warm reopen** | LOW | `src/model/loader.rs` KV cache → append-only `.leafcutter_kv`; key by model+shape; skip prefill on resume |
| 2 | **Never-drop-hits budget** rule | LOW | future MoE streaming; keep cache hits free, cap only misses, decode-only |
| 3 | **3.06-bit container idea** | MEDIUM | Tier-3 streaming: sub-4-bit format cuts disk traffic ~25%; adopt container not codebook |
| 4 | **PILOT-style prefetch barrier** | MEDIUM | Tier-3 if layer-fetch latency dominates (it doesn't today — 70B is compute-bound) |
| 5 | **Oracle validation harness** | MEDIUM | token-exact vs HF reference on a tiny model before scaling — cheap insurance we skipped |

### Surpass (we already beat, or will)

1. **Generalized weight formats.** Colibri is hardwired to its own quant pack
   format (`fmt=`, E8, i3g64). We consume **GGUF** (any quant) — our format
   surface is broader by construction. We should NOT add a proprietary packer.
2. **Model agnosticism.** `colibri.c` is bound to GLM-5.2 dims; `inkling.c` is
   a separate engine for a different arch. Our one binary dispatches
   GGUF-native + safetensors reference + (future) MoE — the "one tool, all
   models" goal is already structurally ahead.
3. **Real measured 70B dense streaming.** Colibri's laptop floor is
   0.05–0.1 tok/s on a 25 GB floor for a 372 GB MoE. Our 70B dense at 11.5 GB
   peak is compute-bound at ~58 s/tok; the gap to Colibri's cold floor on the
   same RAM class is close and the gap closes as we add MoE.
4. **Adaptive RAM budget.** Our loader takes a *budget* and degrades
   gracefully; Colibri's per-layer LFRU is fixed-capacity. Ours is the
   smarter policy (already in ARCHITECTURE.md §4).
5. **No mmap dependency.** We stream dense layers with mmap+DONTNEED (kernel
   page-cache assisted); Colibri is pread-only by design (they need
   DONTNEED-control for a 370 GB file). Our approach measured faster here
   (bench_measure_io.rs) for dense flows.

### Explicitly do NOT copy

- Single-file C, GLM-specific dims, custom packer format, pread-only I/O.
- "io_uring everywhere" — proven noise on our hardware for dense flows.
- The 25 GB / 0.05 tok/s laptop floor as a goal — that's a "it runs"
   bar, not a chat bar. Leafcutter targets useful interactive rates.

---

## 9. Inkling (inkling.c) — the other engine, briefly

Separate pure-C engine for Thinking Machines' **Inkling** (text-only):
hybrid attention (5 sliding-window : 1 global, window=512, 16/8 KV heads,
GQA, **no RoPE** — learned relative-position bias `r_proj`), depthwise-causal
short convs (kernel 4, fp32, on K/V inside attention + after attention +
after MLP), sigmoid router + loss-free bias for top-k, combine weights =
sigmoids of raw logits jointly normalized over topk-routed **+ n_shared
shared experts**, scaled by `route_scale` and `global_scale`, logits scaled by
`logits_mup_width_multiplier`. Same staging principle (dense f32 resident,
routed experts streamed per-expert from fused `[E,2I,D]`/`[E,D,I]` tensors,
LRU-cached, optional int quant with `bits=0` = bit-exact f32 oracle mode).
Scale target: 975B checkpoint.

---

## 10. Benchmarks: both files are now on disk

| File | Size | Status |
|---|---|---|
| `~/Downloads/models/Qwen2.5-1.5B-Instruct/model.safetensors` | 3,087,467,144 B | ✅ complete |
| `~/Downloads/models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1,117,320,736 B | ✅ complete (no .aria2) |

Next: AirLLM (in `/tmp/opencode/airllm-bench`, torch 2.13.0+cpu, safetensors
only) vs Leafcutter native GGUF on the same Qwen2.5-1.5B — tok/s, peak RSS,
dependency footprint. Colibri itself is not runnable here (GLM-5.2-specific,
needs 25 GB+ and its own model download), so the head-to-head is
AirLLM-vs-Leafcutter with the Colibri source lessons as the design reference.
