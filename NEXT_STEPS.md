# NEXT_STEPS — LeafcutterLLM Roadmap

> Current date: 2026-08-03. This file is the prioritized work plan after the
> UX phase and the Ministral-2512 investigation. **Goal: fix the remaining
> correctness gaps so every GGUF we ship produces coherent text natively,**
> then pursue performance. Everything below is ordered by (impact × urgency).

---

## 0. State of the World (read this first)

**Works today (verified on this machine):**
- Native engine, GGUF v3, K-quants (Q4_K/Q6_K/Q8_0), layer streaming, Tier 2/3
  dispatch, `LEAFCUTTER_NO_CACHE` low-RAM mode.
- **Ornith 1.0 9B** (Qwen3.5 hybrid DeltaNet): coherent chat, 1.2–1.67 tok/s,
  reasoning blocks, 183/183 lib tests green.
- **Qwen2.5** (after the Qwen2 attention-bias fix): 4.07 tok/s native vs
  0.188 tok/s AirLLM (21.6×). Coherent.
- UX: `/source`, `leafcutter source add|remove|list`, persistent config,
  OS/arch detection, container at full native speed.
- Chat templates: `cmd_run` prefers the GGUF's embedded Jinja template.

**Broken today:**
- **Ministral-3-3B-Instruct-2512** (and Ministral-3-8B): forward pass works but
  generation is **garbage**. Chat template is now correct
  (`[SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]…[/INST]`). The remaining cause is
  **RoPE-YaRN unsupported** — the engine uses standard RoPE (`theta=10000`)
  but this model needs YaRN (`factor=16`, `beta_fast=32`, `beta_slow=1`,
  `mscale=1`, original_max_position_embeddings=16384).

**Test signals (how to tell when it's fixed):**
- `LEAFCUTTER_DEBUG_PROMPT=1 leafcutter run Ministral-3-3B-… --max-tokens 16`
  must produce a coherent greeting (not token soup). The prompt itself is
  already correct; only the forward pass is wrong.
- Cross-check with Ollama on the same GGUF (Ollama bundles a working YaRN
  implementation): identical prompt → Ollama gives coherent text.

---

## 1. P0 — Implement RoPE-YaRN (unblocks Ministral family)

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
