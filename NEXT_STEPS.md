# NEXT_STEPS — LeafcutterLLM Roadmap

> Current date: 2026-08-18. This file is the prioritized work plan after the
> UX phase and the Ministral-2512 investigation. **Goal: fix the remaining
> correctness gaps so every GGUF we ship produces coherent text natively,**
> then pursue performance. Everything below is ordered by (impact × urgency).

---

## 0. State of the World (read this first)

**Works today (verified on this machine):**
- Native engine, GGUF v3, K-quants (Q4_K/Q6_K/Q8_0), layer streaming, Tier 2/3
  dispatch, `LEAFCUTTER_NO_CACHE` low-RAM mode.
- **Ornith 1.0 35B** (Qwen3.6 MoE, 256 experts): **streams** after the quantized
  MoE expert-slicing fix (2026-08-18) — was OOM-killed 3×, now **3,963 MB peak**.
  Router now reads `mlp.gate.weight`, MoE forward verified finite on real weights.
- **Ornith 1.0 9B** (Qwen3.5 hybrid DeltaNet): coherent chat, peak **~3.3 GB RAM**
  (was ~8.1 GB), 25-token coherent reply verified on the OPi 6 Plus, **202/202 lib
  tests green** (0 failed, 4 ignored). Decode now ~40 s/60 tok (~1.5–1.7 tok/s under
  load, ≈2.4 idle) using ARM sdot kernels + NEON delta_rule + NEON SiLU.
- **ARM sdot Quantized Kernels** (2026-08-20): Q4_K/Q6_K × Q8_K dots via the
  `sdot` instruction (bit-exact vs scalar) on ARMv8.2+; Q4_K decode uses the
  single-column layout; Q6_K stays NEON (bandwidth-bound lm_head). Batched m>1
  Q8 GEMM cut prefill from 63 s to 7.5 s.
- **Qwen2.5** (after the Qwen2 attention-bias fix): 4.07 tok/s native vs
  0.188 tok/s AirLLM (21.6×). Coherent.
- **Qwen3.8-27B Q4_K_M**: **Verified coherent chat & reasoning** on 16 GB CIX Sky1 hardware — loads in Tier 3 (streaming CPU), accepts prompt, and emits coherent persona-aware thinking (`💭The user is asking what I am. I should respond as Ornith…`).
- **Seamless Gold & Purple UI**: Upgraded streaming REPL UI to eliminate redundant text labels (`Thinking...` / `Leafcutter >`). Thinking tokens stream subtly in **dimmed purple** (`dim_purple`), transitioning seamlessly into **bright Gold** (`#FFD700`) for the model's answer.
- **ARMv9 Thread Pool Sizing**: Updated `default_thread_count()` in `init.rs` for `aarch64` so multi-threaded GEMV utilizes all 12 physical CPU cores on the CIX Sky1 instead of halving the thread pool to 6.
- **Automatic Layer Prefetching**: Enabled background layer prefetching by default (`use_prefetch`) when system RAM fits the model, overlapping layer $l+1$ parsing with layer $l$ matmul.
- **MoE Top-K Optimization**: Replaced full $O(N \log N)$ sorting in `moe.rs` with $O(N)$ `select_nth_unstable_by` partitioning.
- **Reference Architecture Analysis**: Created `reference_analysis.md` summarizing key techniques from `colibri` (weight JIT & multi-tiering), `kimi-k3-in-c` (native MXFP4 nibble GEMV & ring streaming), and `llama.cpp` (`ggml-alloc` static arena).

**Fixed (2026-08-20):** ARM sdot quantized kernels (Q4_K/Q6_K × Q8_K, bit-exact);
single-column sdot dispatch for Q4_K decode; batched m>1 Q8 GEMM (prefill 63 s → 7.5 s);
NEON delta_rule (6.5 → 2.2 ms/call); NEON fast-exp SiLU (1.20 → 0.45 ms/call);
per-component layer profiling under `LEAFCUTTER_PROFILE=1`; Ollama A/B measured
(2.93 vs 1.50 tok/s decode, gap now kernel-level). SVE2/i8mm proven a dead end on the
Sky1 (SVE is 128-bit); DeepSeek V4 (165B/FP8) deemed unrunnable on 14 GiB RAM.
202/202 lib tests green. Full detail in `CHANGELOG.md` [2026-08-20].

**Fixed (2026-08-19):** Qwen3.8-27B verification & coherent decode; aarch64 12-core thread pool scaling; automatic layer prefetching; seamless dimmed-purple to gold streaming REPL without label clutter; REPL EOF loop fix; `moe.rs` top-k $O(N)$ partitioning; `install.sh` `leaf` shortcut symlink; CIX Sky1 hardware speccing document (§ 0.1).

**Fixed (2026-08-18):** MoE expert streaming without f32 materialization
(`tensor.rs::expert_slice` + `loader.rs` 3-D layout `[d1*d2, d0]` +
on-demand top-k slicing in `moe.rs`); cache budget now counts
`resident_bytes()` (was under-counting materialized f32); MoE router reads
`mlp.gate.weight` (was the 3-D `mlp.expert_gate.weight`) and scores use
`hidden.matmul(gate_inp)` (no `.transpose()` on quantized-only tensors);
`moe_forward_one_token` hidden shape fixed to `[1, hidden_dim]`; truncate_str
UTF-8-boundary panic fixed (`floor_char_boundary`); `/home/xander` hardcoded
paths → `$HOME`; Cargo.toml stale `[[bin]]` entries removed (all-bins builds
green); aarch64 test gates for AVX2 intrinsics. Full detail in `CHANGELOG.md`.

**Fixed (2026-08-05):**
- **Ministral-3-3B-Instruct-2512 YaRN**: `freq_scale = 1/factor = 1/16`,
  `ext_factor = 1.0` (hardcoded for YARN in llama.cpp), beta_fast/slow 32/1,
  `attn_factor` = raw GGUF value (llama.cpp pre-divides by `1+0.1*log(factor)`
  and the kernel multiplies it back, so the effective mscale is the GGUF value).
  Key fix: the loader previously conflated the GGUF's `scaling.factor` (the
  interpolation factor, 16) with `yarn_ext_factor` (the extrapolation factor,
  1.0) — that made the ramp-mix term blow up the rotary angles. Now coherent:
  server `/v1/chat/completions` returns `2+2=4.`. Ornith + Qwen2.5 unaffected
  (`rope_yarn=None` no-op), 183/183 tests green. Commit `997308a`.

**Test signals (how to verify):**
- `leafcutter server -m Ministral-3-3B-… -p 8081` then
  `curl /v1/chat/completions` → coherent text (matches Ollama on the same GGUF).
- `LEAFCUTTER_DEBUG_PROMPT=1 leafcutter run Ministral-3-3B-… --max-tokens 16`
  must produce a coherent greeting (not token soup).

---

## 0.1  Hardware Reality: Orange Pi 6 Plus

> **This section is not a task list — it is a constraint document.**  
> Every perf target and dispatch decision in the sections below must be
> evaluated against the actual silicon in this machine. Read before
> touching thread counts, memory budgets, or offload logic.

### Silicon inventory

| Component | Spec | Notes |
|---|---|---|
| SoC | CIX P1 / CD8180 (CIX Sky1) | aarch64, ARMv9-A |
| CPU | 4× Cortex-A720 @ 2.6 GHz + 4× Cortex-A55/A520 @ 1.8 GHz + 4× efficiency cores | 12 logical CPUs total (`nproc` = 12); NEON + dotprod; no SVE |
| RAM | 16 GB LPDDR5 (shared with GPU/NPU) | ~14 GiB available to userspace; no swap currently |
| Storage | NVMe (PCIe) + eMMC | NVMe present; add swap here for 35B models |
| GPU | Mali-G720 (reported as `mali0` in `/sys/class/misc`) | Vulkan 1.2 capable; Immortalis-class tile-based |
| NPU | Arm China Zhouyi AIPU (`/dev/aipu`, `armchina,zhouyi-*`) | **28.8 TOPS INT8; only runs precompiled `.aipu.bin` graphs** |
| OS | Ubuntu/Debian aarch64 | Kernel exposes `/sys/class/misc/aipu` |

> **`detect.rs` is correct.** The `/dev/aipu` + `sysfs_misc_exists("aipu")` probe
> accurately identifies the Zhouyi AIPU on this board. The earlier session note
> claiming "wrong label" was based on incorrect hardware assumptions — ignore it.
> `npu:zhouyi-aipu` and `supports_dynamic_offload() == false` are both right.

### CPU-first strategy (P0 — always on)

The Mali-G720 may have competitive Vulkan throughput for quantized matmul, but
it shares LPDDR5 bandwidth with the CPU. Until a shader is benchmarked,
**CPU is the primary inference path** — Cortex-A720 has strong NEON/dotprod
throughput and the LPDDR5 bus gives more bandwidth than LPDDR4x boards.

Current bottleneck is **memory bandwidth** (~50–68 GB/s LPDDR5), not
FLOPS. Every optimization should target bandwidth reduction first
(smaller quant, mmap, sequential reads).

**Thread tuning targets (12-core Sky1):**

| Thread count | Expected behaviour |
|---|---|
| 4 (A720 big cores only) | Highest per-core throughput; best thermal baseline |
| 8 (4×A720 + 4×A520) | Likely sweet spot — test bandwidth saturation |
| 10–12 (all cores) | May saturate bus; efficiency cores add latency for sequential GEMV |

Start with `--threads 8` as the baseline. Run `leafcutter run <model> --threads 4`
vs `--threads 8` vs `--threads 12` on Ornith 9B and record tok/s + peak RSS.
The A520 efficiency cores share cache bandwidth — net gain is model-dependent.

### Vulkan / Mali-G720 (P1 — next GPU path)

Mali-G720 is an Immortalis-class GPU with Vulkan 1.2 support. The
current `wgpu` backend in `inference/gpu.rs` is the correct hook.
Unlike Mali-G610, the G720 has better FP16 throughput and is a more
realistic target for quantized GEMV shaders.

**What to do:**
1. Write a `matmul_q4` GLSL/SPIRV compute shader targeting subgroup size 4–16 (Mali default).
2. Benchmark `Q4_K_M` matrix-vector product (single decode token, `hidden_dim × ffn_dim`) vs NEON baseline.
3. Only promote to default dispatch if the shader is **≥ 1.5× faster** in sustained throughput — Mali bandwidth is shared with the display pipeline.
4. Keep CPU fallback. The GPU path must be gated behind `cfg!(feature = "vulkan")`.

Do **not** assume Vulkan is faster without a benchmark. Mali LPDDR5 bandwidth
contention under full GPU load can nullify throughput gains.

### NPU — Zhouyi AIPU (P2 — sub-graph only, not full LLM)

`detect.rs` correctly identifies the Zhouyi AIPU via `/dev/aipu` and reports
`npu:zhouyi-aipu`. `supports_dynamic_offload() == false` is intentional and correct.

**Capability reality:**
- Zhouyi AIPU only executes precompiled `.aipu.bin` graphs via Arm China's offline compiler.
- `supports_dynamic_offload() == false` is correct — keep it.
- Useful targets: fixed-shape sub-graphs (embedding lookup, RMS-norm, output projection).
- **Not useful for**: attention (dynamic sequence length), MoE routing, KV cache ops.
- Toolchain: `aipurun` + Arm China NeuralONE offline compiler. Build on x86 host, copy binary to board.

**Priority ladder for NPU work:**

| Priority | Sub-graph | Approx speedup |
|---|---|---|
| P2-a | Token embedding lookup (INT8 table, fixed vocab) | ~5–10% prefill |
| P2-b | RMS-norm layers (fixed hidden dim) | ~2–3% per layer |
| P2-c | Output projection (final layer, fixed vocab size) | ~8–12% decode |
| P3 | Full attention block — **skip for now** | Dynamic seq len = incompatible |

### Expected performance baseline (Q4_K_M on CPU)

These are targets, not guarantees. Measure and record in `CHANGELOG.md`.

| Model | Size | Expected tok/s (CPU, 8 threads) | RAM peak |
|---|---|---|---|
| Ornith 1.0 9B | Q4_K_M | 12–18 tok/s | ~5.5 GB |
| Qwen2.5 7B | Q4_K_M | 14–22 tok/s | ~4.8 GB |
| Qwen3.8-27B | Q4_K_M MoE | 4–8 tok/s (MoE gating overhead) | ~14–16 GB |
| Ornith 1.0 35B | Q4_K_M MoE | 1–3 tok/s | ~19–21 GB (will swap on 16 GB board) |

> Ornith 35B will page at 16 GB RAM. Enable NVMe swap before benchmarking.
> See § 5 (performance) for the mmap roadmap.

### Action items generated by this section

- [ ] Add `--threads <n>` CLI flag if not present; default to `8` on aarch64
- [ ] Benchmark `--threads 4` vs `8` vs `12` on Ornith 9B — record in CHANGELOG
- [ ] Write Vulkan matmul probe shader (Mali-G720 subgroup 4–16); compare vs NEON before enabling by default
- [ ] Investigate Arm China NeuralONE / `aipurun` toolchain for embedding-layer export (P2-a above)
- [ ] Enable NVMe swap partition + document in README for 35B models

---

## 1. P0 — Implement RoPE-YaRN (unblocks Ministral family) — DONE

> Status: **COMPLETE** (commit `997308a`). The root cause was that the loader
> set `ext_factor` from the GGUF's `scaling.factor` key (=16) instead of
> llama.cpp's hardcoded YARN ext_factor (=1.0). Both `freq_scale = 1/factor`
> and the per-dim ramp mixing now match llama.cpp `ops.cpp:5822-5844`.
> Verified: Ministral-3-3B coherent via server + `generate`; Ornith + Qwen2.5
> no-op regression; 183/183 lib tests pass.

### 1.1 What YaRN is (one paragraph)
YaRN ("Yet another RoPE extensioN") rescales RoPE position encodings so a
model trained on 16K/32K context can be queried at up to 1M. It has two parts
that MUST both be implemented:

1. **Position scaling**: instead of `θ_i = base^(-2i/d)`, use
   `θ_i' = base'^(-2i/d)` with a per-dimension frequency interpolation factor.
2. **Attention logit scaling**: multiply attention scores by `mscale` and a
   temperature factor derived from `factor` and `original_max_position_embeddings`.

### 1.2 Reference implementation (read these first)
- llama.cpp: `ggml/src/ggml.c` → `rope_yarn` (freq factors) and
  `ggml_compute_forward_rope_yarn*` in `ggml-cpu.c`. Look for `beta_fast`,
  `beta_slow`, `ffs`, `n_dims`, `m_yarn = powf(mscale, -0.5)`.
- llama.cpp `llama-model-loader.cpp` / `ggml-cuda/ggml-cuda.cu` for how the
  `n_ctx_orig_yarn`, `yarn_ext_factor`, `yarn_beta_fast/slow` params are read
  from GGUF metadata keys:
  - `llama.rope.yarn_beta_fast`, `llama.rope.yarn_beta_slow`
  - `llama.rope.yarn_orig_ctx`, `llama.rope.yarn_ext_factor`
  - `llama.rope.freq_base`, `llama.rope.freq_scale`, `llama.rope.dimension_count`
- **IMPORTANT**: this Ministral GGUF stores them under `mistral3.*` keys
  (confirmed: `mistral3.attention.key_length=128`, head_count=32, kv=8).
  Our `metadata_prefix()` for Mistral returns `"llama"` — but `extract_config()`
  has hardcoded `"mistral3.*"` fallbacks. Follow the same pattern for the
  new YaRN keys (both `{prefix}.*` and `"mistral3.*"` and `"llama.*"`).

### 1.3 The math (implement exactly)
Let `d = head_dim`, `i` from `0..d/2`, `base = rope_theta` (default 100000),
`factor = yarn_ext_factor` (Ministral: 16), `orig_ctx = yarn_orig_ctx` (16384),
`beta_fast = 32`, `beta_slow = 1`, `mscale` (Ministral: 1.0).

```
# 1. Find the interpolation ratio
ratio = orig_ctx / target_ctx            # llama.cpp uses n_ctx_orig_yarn / n_ctx
# 2. Compute beta positions in the log space
freq_base' = base ^ (beta_fast / 2pi)    # = base^(beta_fast/(2*pi))
freq_base' *= (ratio - 1)                # interpolated
# Actually, follow llama.cpp rope_yarn() exactly:
#   ffs = floorf(n_dims/2)                 (n_dims = dim/2)
#   tmp = powf(freq_base, beta_fast / (2*pi))
#   tmp *= powf(freq_base, (beta_slow - beta_fast) / (2*pi) * ...)
```

**Use the canonical formula from the YaRN paper / llama.cpp** — do NOT
hand-derive it here. The two key outputs per dimension are:
- `freqs[i]` = the rescaled rotation angle for dimension `i` (applied to
  cos/sin just like normal RoPE, but with `theta_i'`).
- `m_yarn` = attention score multiplier =
  `powf(mscale, -0.5)` then scaled by
  `powf(ratio, -0.5)`-ish per the paper's `attention_scale` = `1.0 + log(ratio)`.

### 1.4 Where it goes in OUR code
- **RoPE application**: `rust/src/inference/attention.rs` — find the existing
  `rope_theta` / `apply_rope` / position-frequency code. It currently builds
  `inv_freq` from `theta` linearly. Add a `RoPEYarnParams` struct
  (`{beta_fast, beta_slow, factor, orig_ctx, mscale}`) and a
  `rope_yarn` branch that computes per-dim `inv_freq` with the YaRN formula,
  then the existing cos/sin application is unchanged.
- **Config loading**: `rust/src/model/loader.rs` → `extract_config()` — add
  fields to `ModelConfig`: `rope_yarn: Option<YarnParams>` populated from
  `{prefix}.rope.yarn_*` / `mistral3.rope.yarn_*` / `llama.rope.yarn_*`.
- **Attention scoring**: `rust/src/inference/attention.rs` — where the Q·K
  dot product produces scores, multiply by the YaRN attention scale when
  `rope_yarn` is set. (This is the part that's easy to forget and causes
  subtle garbage even when cos/sin are right.)

### 1.5 Verification (must all pass)
1. **Sanity math test** — unit test: for `beta_fast=32, beta_slow=1,
   factor=16, orig_ctx=16384`, assert the first few `inv_freq` values match
   llama.cpp's `rope_yarn` output (hardcode expected constants from running
   the C code once).
2. **Model test** — `LEAFCUTTER_DEBUG_PROMPT=1 leafcutter run Ministral-3-3B
   --max-tokens 16` produces coherent text matching Ollama's semantics.
3. **Regression** — Ornith + Qwen2.5 still pass (their `rope_yarn` is `None`,
   so the new branch must be a no-op for them).
4. Run `cargo test --release --lib` — must stay 183/183.

### 1.6 Files to edit (summary)
- `rust/src/model/loader.rs` — ModelConfig + extract_config (+3 YaRN keys)
- `rust/src/inference/attention.rs` — YaRN inv_freq + attention-scale branch
- `rust/src/inference/engine.rs` — pass config through (maybe `attention_interval` region)
- `rust/src/kernels/*` — only if RoPE is fused there (check first)

---

## 2. P1 — Lock down "coherent or fail-fast"

Once YaRN lands, add a **post-load self-check** so broken models can't silently
produce garbage for a year:
- After loading a GGUF, run a **2-token forward** on a fixed probe prompt and
  check the top-1 logit is a real vocab token and differs from a
  "structurally broken" sentinel. If the engine lacks an arch/template match,
  print a loud warning: *"Model X is loaded but unvalidated for this arch —
  expect garbage; known-working: qwen2, qwen35/36, llama."*

This prevents the exact "Ministral looked fine in the banner but outputs
gibberish" trap that cost this session.

---

## 3. P1 — Complete the RoPE feature matrix (beyond YaRN)

Models we support natively share RoPE but with different theta / dims. Build a
table test (`#[test]`) that reads GGUF metadata and asserts the engine picks:
- Qwen2.5: `theta=1e6`, partial dim 64? (verify), Qwen2 bias fix already in.
- Ministral/Mistral3: YaRN (P0 above).
- Llama-3.x-1M: YaRN with `factor=8` (different from Ministral's 16!) —
  parse from metadata, don't hardcode.
- DeepSeek/V3 (DeepSeek2 arch): check theta + dims.
Cover with the metadata-inspection bin (`rust/src/bin/check_shapes.rs` or a
new `dump_rope.rs`) to print the resolved `ModelConfig` rope fields.

---

## 4. P2 — Prompt prefill in streaming chat (correctness)

`cmd_run`'s streaming path (`generate_streaming_with_stops`) must not throw
away all but the last prompt token. Long prompts / system messages currently
degrade. Fix: prefill the full prompt once (non-streaming), cache the KV,
then stream continuation tokens. Reference: `engine.generate()` does prefill +
cache; the streaming variant needs the same first step.

---

## 5. P2 — Performance (after correctness)

- **Top-K preselection for lm_head** — min-heap over top-K rows; 248K
  dequants/token → ~200 (1000× on the head step). Biggest single win per token.
- **Fused dequant-GEMM** — dequantize inside the SIMD dot inner loop.
- **Zero-copy `load_layer`** — mmap slices instead of parsing per call.
- **RoPE fused into attention kernel** — fold the cos/sin multiply into the
  QKV GEMM epilogue to avoid a pass over the activations.

---

## 6. P3 — Tier 1 GPU offload

Probe exists (`detect::GpuKind`, `LEAFCUTTER_PREFER_GPU`). Wire
`--gpu-layers N` partial offload via Vulkan (`backend/wgpu.rs`). Not required
for correctness; defer until CPU perf is good.

---

## 7. Housekeeping (do before starting P0)

- **`rust/src/bin/scan_corruption.rs` and `check_meta.rs`** were removed from
  the manifest (their source never existed in git). If you want them back,
  restore from `rust/src/bin_archive/` equivalents (`check_metadata.rs`,
  `check_meta_full.rs`) and re-declare. Otherwise leave them out.
- `prefill_only.rs`, `split_model.rs.tmp` were transient debug artifacts —
  already cleaned.

---

## 9. P1 — Cynapse integration (Leafcutter as Cynapse's model backend)

Cynapse is being rebuilt in Rust at `/home/xander/Documents/portfolio/cynapse-rs/`
(the Go repo at `/home/xander/Documents/portfolio/cynapse/` is legacy). Goal:
Leafcutter is the inference backend, Cynapse supplies tools/agents.

- `crates/cynapse-core/src/llm/leafcutter.rs` already spawns `leafcutter server`
  and speaks OpenAI-compatible HTTP, but sends **plain role/content, no tools**
  ("leafcutter is a raw inference server").
- Next: tool pass-through — Cynapse `tools.rs`/`agent.rs` sends tool schemas via
  the chat API; Leafcutter returns `[TOOL_CALLS]…[ARGS]…` (per the embedded
  Mistral3 template) or OpenAI-style `tool_calls`.
- **Unlimited OCR** (`/home/xander/Downloads/models/unlimited ocr/`,
  `Unlimited-OCR-Q4_K_M.gguf`) as a document-RAG recall tool — Cynapse `ocr.rs`
  currently shells out to Ollama `frob/unlimited-ocr`; native path is to load the
  GGUF in Leafcutter.

---

## 8. Suggested order of execution

1. Read `ggml.c` rope_yarn + `ggml-cpu.c` (30 min).
2. Add `YarnParams` to `ModelConfig` + read the `mistral3.rope.yarn_*` keys.
3. Implement `rope_yarn` inv_freq + attention-scale in `attention.rs`.
4. Unit-test against hardcoded llama.cpp constants.
5. Run the Ministral model — iterate until coherent.
6. Regression: Ornith + Qwen2.5 + `cargo test`.
7. Add the P1 fail-fast self-check.
8. Commit with a `P0 YaRN` message; update README status table.

---

## Reference files for this roadmap

- `rust/src/inference/attention.rs` — RoPE application + attention scores
- `rust/src/model/loader.rs` — `extract_config()` (~line 396), ModelConfig
- `rust/src/model/arch.rs` — `metadata_prefix()` (Mistral → "llama", the
  reason we need the `mistral3.*` fallbacks)
- `rust/src/tokenizer/chat_template.rs` — `TemplateFamily::detect()` +
  `Ministral::render()` (DONE, works)
- llama.cpp reference: `/home/xander/Documents/portfolio/leafcutter_max/llama.cpp/`
  (`ggml/src/ggml.c`, `ggml/src/ggml-cpu.c`)
- Model to test against: `/home/xander/Downloads/models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`
- Reference safetensors (config.json has the exact YaRN params):
  `/home/xander/Downloads/models/ministral 3/config.json`
