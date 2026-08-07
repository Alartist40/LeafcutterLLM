# Ministral-3 Native Engine Support

How leafcutter's pure-Rust engine was fixed to run
`Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` correctly, end to end.

This document is the record of what was wrong, what was changed, how it was
verified, and how to address issues in the future. It is meant for anyone who
downloads leafcutter and tries to run a Ministral-3 (or any `mistral3`-arch)
model.

---

## TL;DR — what a new user needs to know

- Use a **release** build, not a debug build:
  ```sh
  cd rust && cargo build --release --bin leafcutter
  ./target/release/leafcutter generate -m /path/to/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf "What is 2+2?"
  ```
- If you installed it via `/usr/local/bin/leafcutter`, make sure that symlink
  points at a freshly rebuilt `target/release/leafcutter` — a stale release
  binary was the original cause of "garbage output" (see [Stale-binary trap](#stale-binary-trap)).
- There is nothing extra to configure for Ministral-3. The engine detects the
  `mistral3` architecture from the GGUF metadata and applies:

  1. **NORM-style RoPE pairing** (`rope_pair_norm`), the core fix.
  2. **Attention temperature scaling** (scale 0.1, floor 16384).
  3. **YaRN long-context RoPE** (factor 16 → freq scale 1/16, orig_ctx 16384).
  4. **Byte-for-byte tokenizer** for the tekken/newline tokens.

- Debug and release builds differ in output quality **only** because debug
  builds are extremely slow at the attention matmuls — not because of code
  paths. Once a release binary is built, `leafcutter run 1` produces coherent,
  on-topic text.

---

## Root cause

Ministral-3 uses the `mistral3` GGUF architecture. In llama.cpp this selects
`LLAMA_ROPE_TYPE_NORM` (0) instead of the classic `LLAMA_ROPE_TYPE_NEOX` (2).

The two RoPE types pair the two halves of the rotary embedding differently:

- **NEOX** (used by llama2/3, mistral, qwen, etc.): the `i`-th rotation is
  applied to elements `(base + d, base + d + rope_dim/2)` — the two halves are
  `rope_dim/2` apart.
- **NORM** (mistral3): the `i`-th rotation is applied to **consecutive**
  elements `(base + 2*d, base + 2*d + 1)`.

The native engine only implemented NEOX pairing. For Ministral-3 this put the
imaginary-part of every rotation in the wrong slot, so every Q/K vector was
rotated incorrectly. As a result, attention scores were effectively noise, and
generation collapsed into word-salad even though every other layer (embedding,
FFN, norms, logits) was already correct.

### Secondary contributors (all verified, not needed for the fix)

- **Stale release binary.** `/usr/local/bin/leafcutter` → `rust/target/release/leafcutter`
  was built (Jul 26) *before* the fix. Every user-facing run therefore used the
  old, broken engine. Rebuilding release fixed this.
- **CPU oversubscription.** `configure_thread_pool()` in `src/init.rs` existed
  but was never called, so rayon used *all* 16 logical cores (~1585% CPU, fans
  on max) for a single-threaded inference workload. Wired into `main()`.
- **CLI arg collision.** `-m` was claimed by both `--model` and `--max-tokens`,
  so `clap` panicked on start for some invocations. `--max-tokens` now uses `-n`.

---

## What was changed (production code only)

### 1. RoPE pairing mode — the core fix

`rust/src/inference/attention.rs`:

- `AttentionParams` gained `rope_pair_norm: bool` (default `false` = NEOX).
- `apply_rotary_emb(..., pair_norm: bool)` now picks the index pair:

  ```rust
  let (x1_idx, x2_idx) = if pair_norm {
      (base + 2 * d, base + 2 * d + 1)
  } else {
      (base + d, base + d + rope_dim / 2)
  };
  ```

- `attention_forward` passes `params.rope_pair_norm` to both Q and K RoPE.

`rust/src/inference/engine.rs` — `infer_attention_params` reads the raw GGUF
metadata key directly (the `mistral3` → `Mistral` mapping in `model/arch.rs`
loses the original arch name, so the metadata is re-checked):

```rust
let rope_pair_norm = match model.file.metadata.get("general.architecture") {
    Some(crate::model::gguf::GGUFValue::String(s)) => s == "mistral3",
    _ => false,
};
```

`rust/src/inference/gemma.rs` — the Gemma path constructs its own
`AttentionParams`; it sets `rope_pair_norm: false` explicitly (Gemma uses NEOX).

> If another architecture needs NORM pairing, extend the match above — do not
> add more bool plumbing.

### 2. Attention temperature scaling

Ministral-3 / Llama-4 scale Q by a per-position temperature factor after RoPE:

```
scale(pos) = log( floor((pos + offset) / floor_scale) + 1 ) * k + 1
```

with `k = 0.1`, `floor_scale = orig_ctx = 16384`. For positions below
`floor_scale` this is exactly 1.0 (identity), so short prompts are unaffected;
only long contexts are.

- `rust/src/model/loader.rs`: reads `<prefix>.attention.temperature_scale`
  (with `mistral3.`/`llama4.`/`llama.` fallbacks) into
  `ModelConfig.attention_temp_scale`; `attention_temp_floor_scale` is taken
  from the YaRN `orig_ctx`. If a scale exists but no `orig_ctx` is found, temp
  scaling is disabled with a warning.
- `attention.rs`: applies the per-position factor to Q inside `attention_forward`.
- `gemma.rs`: sets `temp_scale: 0.0` (disabled).

### 3. YaRN long-context RoPE

Already implemented previously (`YarnParams`, factor 16 → freq scale 1/16,
beta_fast 32, beta_slow 1, orig_ctx 16384) and now verified against llama.cpp
cosine similarity (see below). No change needed this round.

### 4. Tokenizer fixes

`rust/src/tokenizer/gguf.rs` (from the prior session):

- Consecutive newlines group into a single pre-token (`.\n\n` stays together
  and BPEs into one vocab token instead of splitting into `.\n` + `\n`).
- A newline group is standalone — the following word does not absorb leading
  newlines (GPT-2/tokenizer regex `\s+`).

Together with the earlier byte-for-byte fixes this tokenizes all 244 test
prompts identically to llama.cpp's tokenizer.

### 5. Runtime / CLI fixes

`rust/src/main.rs`:

- `--max-tokens` short flag changed from `-m` to `-n` (fixes clap panic).
- `leafcutter::init::configure_thread_pool(None)` is the first statement of
  `main()`, before any `par_iter()`, capping rayon to `physical cores − 1`.

---

## Verification evidence

Ground truth = vendored llama.cpp at `rust/llama.cpp` (custom commits:
per-tensor mmap release, Ollama-lean footprint, flash-KV rotate). Model =
`Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` (26 layers, hidden 3072, vocab
131072, 236 tensors).

### RoPE — before vs after

Post-RoPE Q/K, layer 0, cosine similarity to llama.cpp:

| Tensor | Before (NEOX pairing) | After (NORM pairing) |
|--------|----------------------|----------------------|
| Q      | 0.934                 | **0.999990**          |
| K      | 0.896                 | **0.999990**          |

Layers 1–2 post-RoPE ≈ **0.9997**. Layer-0 `attn_out` and `ffn_out` cosine ≈
**0.9999+**. Remaining logit differences are llama.cpp f16 KV cache vs
leafcutter f32 — not a logic error.

FFI per-position logit comparison (positions 0,1,2,8,64,96) matches top-1
tokens exactly; the "DIFF" positions (3,4,16,32) are near-tie flips within
±0.3 logit of the winning token.

### End-to-end generation

Release binary, `generate`:

- `--raw "What is 2+2?"` → `The answer is 4. But what if you are asked…`
- Chat template → `Hey! 😊 I'm just a bunch of code and data—no feelings, but
  I'm here and ready to…`
- Poem request → coherent `**The Ocean's Song** …`
- Photosynthesis question → coherent essay-style answer.
- `leafcutter run 1` → good responses at **0.95–2.56 tok/s**.

### CPU

Before: ~1585% CPU (all 16 logical cores, fans max). After: **~330%** CPU,
fans quiet, no throughput loss.

---

## How to reproduce the fix from scratch

1. Get the model and confirm the architecture:
   ```sh
   ./target/release/leafcutter diagnose -m Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
   ```
   Expect `general.architecture == "mistral3"`.
2. Rebuild release, then run the smoke tests in [End-to-end generation](#end-to-end-generation).
3. If output is garbage, check these in order:
   - Is the binary a **fresh release build**? (`cargo build --release`)
   - Is the `mistral3` arch detected? Grep for `rope_pair_norm` wiring in
     `engine.rs::infer_attention_params`.
   - Does the GGUF actually contain `mistral3.attention.temperature_scale`
     (KV key) and `rope_yarn` block? If the file was converted oddly, both
     fall back safely.

---

## Addressing issues in the future (troubleshooting guide)

### Symptom: coherent start, then drift / repeats after a long prompt

Suspect **attention temperature scaling**. Confirm the model metadata contains
`mistral3.attention.temperature_scale`. If your model's `orig_ctx` is *not*
16384, `attention_temp_floor_scale` must match it — long-context behavior
beyond `orig_ctx` is exactly what this knob controls.

### Symptom: immediate word-salad from token one

Suspect **RoPE pairing**. Dump post-RoPE Q/K for layer 0 from both engines and
compare cosine. Expected ≈ 0.99999 with NORM pairing; ≈ 0.93 with NEOX (wrong).
The pairing choice is hardcoded to `arch == "mistral3"` in
`infer_attention_params` — if a new `mistral3`-descendant arch appears, add it
to that match. Do **not** switch the default, it would break every other model.

### Symptom: tokenizer mismatches llama.cpp

Run the full byte-for-byte tokenizer corpus test in the repo and compare token
ids. Look at `tokenizer/gguf.rs` newline-grouping logic; regenerate the
expected-token fixtures if the test data is stale.

### Symptom: whole system sluggish / fans on max during inference

`configure_thread_pool(None)` must run before the first `par_iter()` (it's the
first line of `main()`). Override via `RAYON_NUM_THREADS` or
`LEAFCUTTER_THREADS` env vars.

### Symptom: clap panic "`-m` used multiple times" / similar

`--model` and `--max-tokens` used to collide on `-m`; `--max-tokens` is `-n`
now. If a new short flag is added, make sure it doesn't collide with `-m`.

### Working on the engine directly (not recommended for users)

The fastest check loop is:

```sh
cargo build --release --bin leafcutter
./target/release/leafcutter generate --raw -n 64 -m /path/to/Ministral... "What is 2+2?"
```

Compare `target/release` output against `target/debug` only if you are
debugging code paths — for output-quality issues always use release.

---

## Stale-binary trap

`/usr/local/bin/leafcutter` is a symlink to `rust/target/release/leafcutter`.
`cargo build` (debug) and `cargo build --release` write to **different**
targets. If you build debug and then run `leafcutter`, you still execute the
old release binary. Whenever a fix lands, rebuild release and confirm the
symlink target:

```sh
ls -l /usr/local/bin/leafcutter
cargo build --release --bin leafcutter
```

---

## Model spec reference (for this fix)

| Property | Value |
|----------|-------|
| arch | `mistral3` |
| layers | 26 |
| hidden | 3072 |
| vocab | 131072 |
| tensors | 236 |
| RoPE | NORM pairing, dim 64/head, theta 10000 |
| YaRN | factor 16, freq_scale 1/16, beta_fast 32, beta_slow 1, orig_ctx 16384 |
| attn temp | scale 0.1, floor 16384 |
| tie_word_embeddings | yes |

## Related files

- `rust/src/inference/attention.rs` — `rope_pair_norm`, `temp_scale`,
  `temp_floor_scale`, NORM/NEOX pairing, per-position Q scaling.
- `rust/src/inference/engine.rs` — `infer_attention_params` arch detection +
  wiring into `AttentionParams`.
- `rust/src/inference/gemma.rs` — NEOX defaults for the Gemma path.
- `rust/src/model/loader.rs` — `attention_temp_scale` / `attention_temp_floor_scale`
  GGUF parsing with fallbacks.
- `rust/src/tokenizer/gguf.rs` — tekken/newline pre-token grouping.
- `rust/src/init.rs` — `configure_thread_pool` (physical cores − 1).
- `rust/src/main.rs` — thread-pool wiring + `-n` max-tokens flag.
