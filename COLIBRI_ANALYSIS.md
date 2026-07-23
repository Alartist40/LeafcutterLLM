# Colibrì Analysis — What LeafcutterLLM Can Learn

**Date:** 2026-07-22  
**Author:** Analysis of [JustVugg/colibri](https://github.com/JustVugg/colibri) @ commit 6368e1a  
**Source path:** `/home/xander/Documents/reference/colibri/`

---

## 1. What Colibrì Actually Is

Colibrì is a single-file C inference engine (`c/colibri.c`, 400 KB / 6,744 lines)
that runs **GLM-5.2 — a 744B-parameter Mixture-of-Experts model** — on consumer
hardware with as little as 25 GB of RAM. It has optional CUDA and Metal backends.
The runtime is pure C with zero dependencies; Python is only used for the
one-time model converter and the optional API gateway.

### Headline Claims (verified against source)

| Claim | README | Source Evidence | Verdict |
|---|---|---|---|
| 744B MoE model | "744B-parameter MoE" | `coli` line 4: "GLM-5.2 (744B)"; `download_glm52.py` line 4: "zai-org/GLM-5.2-FP8 (~756 GB)" | **Confirmed** — GLM-5.2 is a real Z.ai model |
| 19,456 experts | "75 MoE layers × 256 + MTP head" | `st.h` line 47: "256 expert x 78 layer x 3 x 2"; `colibri.c` line 686: "19,456 experts" | **Confirmed** |
| ~25 GB RAM floor | "25 GB dev box: 0.05–0.1 tok/s cold" | Resident set is dense part only (~9.9 GB int4); experts stream from disk | **Architecturally sound** |
| MLA compressed KV | "576 floats/token instead of 32,768 (57× smaller)" | `backend_loader.c`: `latent`, `rope` params throughout; MLA = Multi-head Latent Attention (DeepSeek V2) | **Confirmed** — real MLA implementation |
| Streaming from disk | "experts live on disk, streamed on demand" | `st.h` line 3: "legge con pread (niente mmap) + posix_fadvise(DONTNEED)"; `colibri.c`: `expert_load_impl()` | **Confirmed** — real pread-based streaming, no mmap by default |
| LRU + learned cache | "per-layer LRU cache, learned pinned hot-store" | `tier.h`: full LFRU (LFU+LRU hybrid) eviction with heat tracking, 25% hysteresis | **Confirmed** — production-quality cache |
| Speculative decoding (MTP) | "2.2–2.8 tokens/forward" | `colibri.c` line 226: "testa MTP (layer n_layers, stile DeepSeek-V3)"; int8 MTP heads required | **Confirmed** — real MTP implementation |
| Router lookahead prefetch | "71.6% predictable one layer ahead" | `colibri.c`: `g_pilot`, `pilot_prefetch()`, `g_pilot_two`, `couple_prefetch()` | **Confirmed** — real cross-layer prefetch system |
| Token-exact oracle validation | "teacher-forcing 32/32" | `ref_glm.json` in repo; `colibri.c` line 6672: oracle reference | **Confirmed** — test infrastructure exists |
| CUDA + Metal backends | "GPU-resident pipeline, Metal backend" | `backend_cuda.cu` (91 KB / 1,594 lines); `backend_metal.mm` (67 KB / 1,039 lines) | **Confirmed** — real GPU backends |
| Dual-SSD mirror | "two copies, twice the read bandwidth" | `compat.h`: `compat_pread` with per-drive routing; deterministic hash for drive selection | **Confirmed** |

### What Is NOT in the README (honest gaps)

1. **Speed claims are best-case.** The 5.8–6.8 tok/s figure requires 6× RTX 5090
   (~$25k of GPUs). The 25 GB laptop floor is 0.05–0.1 tok/s — that's
   **10–20 seconds per token**. Useful for "it runs" but not for chat.

2. **The "1 trillion parameter" claim the user heard is NOT in the README.**
   The README says 744B consistently. The model file is 372 GB on disk (int4).
   A 1T model would be ~500 GB int4. The user's claim may conflate 744B with
   a different model or a rounded-up figure.

3. **"1 million context windows" is not explicitly claimed either.** The MLA
   compression (576 floats/token) would *enable* very long contexts — at
   4.6 KB/token, 1M tokens = 4.6 GB KV. But the README doesn't claim 1M context;
   it claims "conversations reopen warm with zero re-prefill" via `.coli_kv`.

4. **The engine is GLM-5.2-specific.** Despite the "model-agnostic" roadmap
   promise, the code (`colibri.c`) is hardwired for GLM-5.2 architecture:
   `n_routed_experts=256`, `n_layers=75`, specific MLA dimensions. `olmoe.c`
   is a separate engine for OLMoE. Generalizing to arbitrary MoE models would
   require per-model engine variants.

5. **Quality claims are self-reported.** "Token-exact against transformers
   oracle" is tested against a TINY model (`colibri.c` line 6672: "L'oracolo
   e' del modello TINY"). The 744B model's actual output quality is not
   validated in-repo — only the *computational path* is verified.

---

## 2. Core Architecture — How Colibrì Achieves What It Does

### 2.1 The Fundamental Bet

> Parameters are not resident state to be held — they are **data to be staged**
> across a heterogeneous storage hierarchy (VRAM / RAM / NVMe), exactly when
> the router proves they are needed.

This is the same principle as a JIT compiler: don't compile the whole program,
compile the hot paths. Colibrì applies it to weights: don't load all 744B
params, load the ~40B that the router says this token needs.

### 2.2 The Memory Hierarchy

```
VRAM (fastest, smallest)  →  RAM (medium)  →  NVMe/SSD (slowest, largest)
┌──────────────────────────────────────────────────────────────────┐
│  Hot experts (learned pin set)    │  Warm experts (LRU cache)    │
│  Dense layers (attention, embed)  │  Cold experts (stream disk)  │
└──────────────────────────────────────────────────────────────────┘
```

- **Dense part** (attention, shared experts, embeddings — ~17B params): always
  resident in RAM at int4 (~9.9 GB)
- **19,456 routed experts** (~19 MB each at int4): on disk (~370 GB), streamed
  on demand with pread + posix_fadvise(DONTNEED) to avoid page cache pollution
- **LRU/LFRU cache**: per-layer, bounded, with frequency+recency scoring
- **Learning cache** (`.coli_usage`): persists routing heat across restarts,
  pins hottest experts → "gets faster the more you use it"

### 2.3 The Per-Token Pipeline

```
For each layer L (0..74 + MTP):
  1. Route: gate network selects top-k experts (k=8 of 256)
  2. Union: batch all positions → unique expert set
  3. Place: check cache → hit = RAM, miss = pread from disk
  4. Overlap: async I/O loads missing experts while resident ones compute
  5. Learn: update .coli_usage heat counters
```

### 2.4 Key Engineering Tricks

| Trick | How | Why It Works |
|---|---|---|
| pread + DONTNEED | `posix_fadvise(FADV_DONTNEED)` after read | Frees page cache → doesn't evict useful pages, doesn't OOM on 370GB model |
| LFRU eviction | `tier.h`: frequency×256 + recency (0–255) | Frequency dominates; recency breaks ties. 25% hysteresis prevents ping-pong |
| Expert adjacency | 3 matrices (gate, up, down) stored adjacent in one file → single pread | 1 syscall per expert instead of 3 |
| Batch-union | For batch of N positions, union their top-k expert sets → read each unique expert once | Shared experts across positions save redundant disk reads |
| Router lookahead (PILOT) | Run router for layer L+1 while computing layer L | Routing is "71.6% predictable one layer ahead" → prefetch wins often |
| Coupling prefetch (COUPLE) | `.coli_pairs` — cross-layer expert co-occurrence scoring | If expert X fires at layer L, expert Y often fires at L+1 → prefetch Y |
| O_DIRECT | `DIRECT=1` — bypass page cache entirely | +34% decode on drives with DRAM cache; neutral/negative on QLC |
| Dual-SSD | Second copy of model on second drive → 2× read bandwidth | Experts are read-only, deterministic hash routes to drives |
| MLA compressed KV | 576 floats/token vs 32,768 (57×) | DeepSeek V2 architecture — latent projection of K/V |
| KV persistence | `.coli_kv` — compressed KV state saved to disk | Conversations reopen warm, zero re-prefill |
| Speculative MTP | GLM-5.2 native MTP head drafts tokens, main model verifies | 2.2–2.8 tokens/forward when accepted; int8 head required (int4 collapses) |

---

## 3. Lessons for LeafcutterLLM

### What Colibrì Does That We Already Do

- **Layer-at-a-time weight loading.** Leafcutter already loads one layer's
  weights into RAM at a time. Colibrì does the same for dense layers.
  We're on the right page here.

### What Colibrì Does That We Don't (Ranked by Applicability)

| # | Lesson | Applicability to Leafcutter | Effort | Impact |
|---|---|---|---|---|
| A | **Streaming weight tiers** (disk→RAM→VRAM) | HIGH — this IS the 70B-on-small-RAM play | HIGH | HIGH |
| B | **LFRU cache with heat tracking** | HIGH — needed if we stream from disk | MEDIUM | HIGH |
| C | **MLA compressed KV cache** | MEDIUM — only if we support MLA models | HIGH | HIGH |
| D | **prenuclead + DONTNEED** (avoid page cache pollution) | HIGH — direct port to Rust | LOW | MEDIUM |
| E | **Expert adjacency** (store matrices adjacent, single read) | MEDIUM — for MoE models specifically | LOW | MEDIUM |
| F | **Batch-union expert loading** | MEDIUM — for batched MoE decode | MEDIUM | MEDIUM |
| G | **Router lookahead prefetch** | LOW — only for MoE | MEDIUM | MEDIUM |
| H | **KV persistence across restarts** | HIGH — useful for all models | LOW | MEDIUM |
| I | **Speculative decoding (MTP/draft)** | MEDIUM — model-specific | HIGH | MEDIUM |
| J | **Dual-SSD mirror** | LOW — niche hardware config | LOW | LOW |
| K | **O_DIRECT** | LOW — drive-dependent, marginal | LOW | LOW |

### What We Should NOT Copy

1. **Single-file C architecture.** Colibrì's 400 KB `colibri.c` works for them
   but is the opposite of Leafcutter's modular Rust approach. We stay modular.

2. **GLM-5.2-specific hardcoding.** Their engine is bound to one model
   architecture. Leafcutter's GGUF loader is more general — we keep that.

3. **No BLAS / no external deps.** Colibrì prides itself on zero deps. We can
   use rayon, std SIMD, and potentially BLAS — Rust's ecosystem is a strength,
   not a weakness.

---

## 4. The Real Question: Can We Run 70B on Small RAM?

**Yes — and we already do.** Measured today on this machine, on a 16 GB
laptop with leaf-fanned blades spinning:

| Model | File Size | Peak RSS | Layer stream? | Source |
|---|---|---|---|---|
| Llama 3.3 70B Q4_K_M | 42.5 GB | **~1.08 GB** | yes — via `madvise(MADV_DONTNEED)` | measured 2026-07-22 |
| Llama 3.3 70B Q4_K_M (prev claim) | 42.5 GB | 1,145 MB | yes — pre-existing | `FRONTIER_MODELS_PLAN.md` |
| AirLLM (comparable) | ~40 GB | ~4 GB | yes — pread | airllm docs |

**Leafcutter already runs 70B dense at 1.08 GB peak — beating AirLLM ~4×.**
The right framing isn't "can we run 70B"; it's "how do we extend this
to frontier MoE models that need much more disk-resident state."

The math is the same as Colibri's, just on a different shelf of model:

- **Per-layer weight size (70B dense):** 70B / 80 layers ≈ 875M params/layer ≈ 500 MB Q4
- **Resident working set:** 1-2 layers + KV + activations = ~1.1 GB
- **Total RAM:** ~1.1 GB measured — matches prediction
- **Disk served from:** GGUF mmap + MADV_DONTNEED (kernel handles paging,
  kernel tips keep pages out of RAM after we're done with them)

The CPU work is forced to wait on dequantize + AVX2 matmul per layer (~1.15 s
per layer today). For a 70B dense 1 token = 92 s. For frontier MoE with
8× more total layers and a router on top, it could be 10× worse without
expert streaming.

**The key insight (corrected):** for 70B dense on small RAM, the bottleneck
is **dequantized f32 activation flow + AVX2 matmul**, NOT I/O — the kernel's
own page cache is fast enough for streaming. The mmap+MADV_DONTNEED approach
already works because the kernel doesn't waste pages on us.

For frontier MoE (Kimi K2.6 = 384 experts/layer = 60 MoE layers,
~8 GB resident pool of experts), the bottleneck DOES become I/O — at which
point the Colibri lessons (pread+DONTNEED, LFRU, expert streaming) apply.
