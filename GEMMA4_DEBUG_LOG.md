# Gemma 4 12B — Forward-Pass Debug Log

**Started:** 2026-06-30
**Model under test:** `/home/xander/Downloads/models/gemma-4-12b-it-Q4_K_M.gguf` (7.12 GB, `general.architecture = "gemma4"`)
**Reference:** HuggingFace `transformers` 5.9.0 (the source the GGUF weights were converted from — single source of truth).
**Goal:** Make Leafcutter's native `forward_native` path produce coherent output matching the HF reference within fp32 noise (~1e-3 per layer).

> **Status 2026-08-01 (project wrap-up):** Gemma 4 12B runs all 48 layers
> through the native path and emits tokens; multi-token coherence is still
> an open item (see the "Notes / Known issues" entry in CHANGELOG 2026-06-29).
> The flagship verified model is Ornith 1.0 9B (Qwen3.5 hybrid), which
> generates coherent chat natively via `leafcutter run ornith`.

This is a **living document**. Every finding is recorded with `file:line` evidence so the team can verify it independently. Nothing here is opinion — every claim should be reproducible from the evidence given.

---

## Symptom (verified from handoff + code)

- 48-layer forward pass runs end-to-end without panic. ✅
- First-token argmax for "Hello" is recognizable (`id=9259 'Hello'`). ✅
- **But** multi-token greedy decode degenerates ("is is is is is").
- Pre-softmax logits come back 10–20× larger than expected (+20 to +30 range vs expected ±a few). ❌

The magnitude blowup through 48 layers × 4 RMSNorms = **192 norm applications** is the signature of a wrong RMSNorm scaling convention. That's the leading hypothesis and the first thing checked below.

---

## Finding 1 — RMSNorm convention: the code contradicts its own test  ⚠️ CONFIRMED BUG

**Evidence (both in `rust/src/inference/gemma.rs`):**

- **Lines 60–86** — `gemma_rms_norm` applies the weight as `(w + 1.0)`:
  ```rust
  out.push(x.data[base + d] * inv_rms * (w + 1.0));
  ```
  Comment: *"Match HF Gemma3: `y = x * inv_rms * (weight + 1)`"*

- **Lines 407–433** — the unit test `gemma_rms_norm_multiplies_weight_directly` asserts the **opposite**, with expected values computed **without** `+1`:
  ```
  // Reference: llama.cpp applies w directly, no +1 offset.
  //   y = x * inv_rms * w
  ```
  The test's arithmetic (e.g. `y[0] = 1 * 0.6325 * w[0] = 0.31625` for `w[0]=0.5`) only works if the weight is applied directly. With `+1` it would be `1 * 0.6325 * 1.5 = 0.949`.

So **the function and its test cannot both be right.** The test would fail against the current function — meaning either the test isn't being run, or it's being ignored.

**Git history shows it was flip-flopped:**
- `2563155 gemma: align RMSNorm formula with llama.cpp reference` (direct `w`)
- `402c427 gemma: re-apply +1 in RMSNorm` (current: `w + 1`)

**Resolution path:** The on-disk GGUF weights have a *single* correct convention. We determine it empirically by comparing one layer's RMSNorm output to the HF reference. See Finding 3.

**Convention background (to be confirmed against HF source):**
- **HF `Gemma3RMSNorm`**: `y = x * rsqrt(mean(x²) + eps) * (1 + weight)` — weight is centered at 0, applied with `+1`.
- **llama.cpp `ggml_rms_norm`**: applies `weight` directly — meaning GGUF converters for Gemma bake the `+1` into the stored weight, so the file already contains `(1 + γ₀)`.

If both statements are true, they describe the same math from two sides — and the question is purely **which side the Gemma 4 GGUF converter landed on**. That is a factual question with a single answer, resolved by running the reference.

---

## Finding 2 — Reference tooling audit

- ✅ `venv/` has `transformers 5.9.0` + `torch 2.12.0+cu130` (CPU-only execution, no GPU present).
- ❌ No `llama-cli` / `llama-server` binary is built. Only `rust/llama.cpp/build/` libs exist. The attachment's claim that "llama.cpp b9840 emits 'The user is asking for'" is **not reproducible on this machine** without a build step. Treat as unverified until rebuilt.
- Hardware constraint: 15 GB RAM total, ~11 GB free. A 12B model in fp16 needs ~24 GB → **cannot load the full HF model naively**. Strategy: load with `device_map="auto"` + `load_in_8bit`, OR extract layer-0 weights and run a single-layer reference in isolation (preferred — also the cleanest comparison).

**Decision:** Use HF transformers as the reference (fidelity to the source weights > llama.cpp convenience). Run single-layer isolated references to fit in RAM and to localize bugs precisely.

---

## Finding 3 — Pending: empirical RMSNorm convention check

_TODO once the HF reference runs. Procedure:_
1. Extract `blk.0.input_layernorm.weight` from the GGUF.
2. Take a known input vector (e.g. a token embedding scaled by `sqrt(hidden)`).
3. Compute RMSNorm three ways: (a) `w` direct, (b) `w + 1`, (c) HF reference.
4. Whichever of (a)/(b) matches (c) within ~1e-5 is the correct convention. Fix the code AND the test to agree.

---

## Working-tree hygiene note (separate from the bug)

`rust/` contains ~500 stray debug artifacts from an earlier, unrelated DeltaNet/SSM debugging effort (`native_l*_ssm_*`, `dbg_*`, `compare_*.py`, `hf_layer_*.bin`, etc.). These are not source. They make the directory unreadable and pollute `git status`. Cleanup is queued as low-priority but recommended — it's part of why the codebase has been hard to reason about.

---

## Repro commands (for the team)

```bash
# Run the (currently failing) RMSNorm test to confirm Finding 1:
cd rust && cargo test --lib gemma_rms_norm_multiplies_weight_directly -- --nocapture

# Dump per-layer L2 norms from Leafcutter:
cd rust && LEAFCUTTER_DEBUG_NORMS=1 ./target/release/leafcutter generate \
    --model /home/xander/Downloads/models/gemma-4-12b-it-Q4_K_M.gguf \
    --prompt "Hello" --max-tokens 4

# (TODO) HF reference single-layer script: rust/scripts/ref_gemma4_layer0.py
```
