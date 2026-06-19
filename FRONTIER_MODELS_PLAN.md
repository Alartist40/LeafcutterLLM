# LeafcutterLLM — Frontier Models Expansion Plan

> **Date:** 2026-06-19
> **Author:** m3 (Nvidia)
> **Status:** Active build-out. Native support for Kimi K2.6 and GLM-5.2 (both DeepSeek-style MoE + MLA).

---

## 1. Why this exists

The current native path of LeafcutterLLM runs standard dense transformers (Llama, Qwen2, Mistral, Mistral3, Yi, Gemma, Phi) and the Qwen3.5/3.6 hybrid (DeltaNet + attention). It validates at 1,145 MB for a real 70B model via layer streaming. That covers the dense-at-2 GB vision well.

What it does *not* yet cover:

- **Frontier MoE models** with hundreds of routed experts and shared experts (DeepSeek-2/V3-style).
- **Multi-Latent Attention (MLA)** with compressed Q/K/V through latent vectors.

These are exactly what the newly-downloaded Kimi K2.6 and GLM-5.2 require. This document is the plan for getting both running natively while keeping every already-validated model working unchanged.

---
## 2. What the source file confirmed

We have the actual GGUF data for both models on disk. Shard-1 was added
2026-06-19 so all real architecture metadata is now visible.

| Model                    | Shards | Bytes on disk | Architecture metadata? | Tensor-table fingerprint readable? |
|--------------------------|--------|---------------|--------------------------|------------------------------------|
| Kimi-K2.6-UD-Q4_K_XL     | 14     | 48.39 GB      | FULL (49 keys, 60 KV)    | YES                                |
| GLM-5.2-UD-Q4_K_XL       | 11     | 48.57 GB      | FULL (51 keys, 69 KV)    | YES                                |

### 2a. Confirmed Kimi-K2.6 (`deepseek2`)

- 61 layers, hidden 7168, ffn_lo 18432, ffn_expert 2048
- 384 routed experts (k=8), 1 shared expert, scale 2.827
- MLA: q_lora_rank 1536, kv_lora_rank 512, qk_nope 192, qk_rope 384, v 128
- 64 Q heads, 1 KV head (MQA), rope_dim 64, rope_theta 50,000, **YaRN-64** (β=32/1, original ctx 4096)
- Context length 262,144, vocab 163,840
- 1 leading dense block (block 0), 60 MoE blocks
- Quant: Q4_K_XL (Unsloth repack)

### 2b. Confirmed GLM-5.2 (`glm-dsa`)

- 79 layers, hidden 6144, ffn 12,288, ffn_expert 2048
- 256 routed experts (k=8), 1 shared expert, scale 2.5
- MLA: q_lora_rank **2048**, kv_lora_rank 512, qk_nope **256**, qk_rope 320 (= 576−256), v **256**
- 64 Q heads, 1 KV head (MQA), rope_dim 64, rope_theta **8,000,000**, **no scaling** (YaRN absent)
- Context length **1,048,576** (1M tokens) — long-context flag
- **DeepSeek-Sparse-Attention indexer**: 32 heads, top_k=2048 → selectively attends to top-2k keys
- `nextn_predict_layers = 1` (MTP heads; wired as `nextn.*` tensors)
- 3 leading dense blocks, 76 MoE blocks
- Quant: Q4_K_XL

### 2c. Same engineering consequences, different magnitudes

For both models: compute the same MoE+MLA forward math, just with
different dims. So a single Rust core that branches on
`{num_experts, hidden_dim, q_lora_rank, …}` supports both. Tama.

---

## 3. Hard constraint from the user

> "do not break the program that we have built, as I still want it to be able to run the previous models that I have built it for."

So **all existing forward paths stay**:

- `has_standard_attn` (Llama/Qwen2/Mistral/Ministral dense MHA) — unchanged
- `has_deltanet` (Qwen3.5 SSM hybrid) — unchanged
- `has_ssm` (legacy Mamba stubs) — unchanged
- Existing FFN `Engine::ffn_forward` (gate/up/down) — unchanged

What we add is **two new detection branches** in `forward_native()`, mirroring the pattern that already exists for the Qwen3.5 hybrid:

```
if has_standard_attn { ... }       // existing
else if has_deltanet { ... }       // existing
else if has_ssm { ... }           // existing
else if has_mla { ... }           // NEW: MLA attention + MLA pre-proj
else if has_moe { ... }           // NEW: replaces ffn_forward for MoE block
```

The existing FFN forward only runs when neither `has_moe` nor `has_dense_expert_shexp` is true.

---

## 4. Architecture-intake checklist (now permanent)

For every new model, we record:

1. `general.architecture` (and per-arch keys: `deepseek2`, `glm_dsa`, etc.)
2. `block_count` (number of layers)
3. `attention type` — MHA / GQA / MLA / MQA
4. `ffn type` — dense / MoE / shared-expert MoE / dense+shared
5. `num_experts`, `num_experts_used`, `expert_feed_forward_length`
6. `nextn_count` / MTP heads (`nextn.*` tensors)
7. `rope.sections` (MRoPE) and `rope.freq_base`
8. Context length (`*.context_length`)
9. Chat template (auto-detected already; see `tokenizer/chat_template.rs`)
10. Quantization types used (verify all native kernels cover them)

For Kimi (DeepSeek-2-family) and GLM (glm-dsa / DSA), this checklist is essentially the same — they share MLA + routed-MoE + shared-expert + MTP. So implementing it for one will get the other for free in most respects.

---

## 5. What changes in the engine

### 5.1 New module: `src/inference/moe.rs`

MoE forward path. Inputs:

- hidden: `[seq_len, hidden_size]` (same as dense FFN)
- `ffn_gate_inp.weight`: `[num_experts, hidden]`  — router
- `ffn_gate_exps.weight`: `[num_experts, expert_ffn, hidden]`
- `ffn_up_exps.weight`: `[num_experts, expert_ffn, hidden]`
- `ffn_down_exps.weight`: `[num_experts, hidden, expert_ffn]`
- `ffn_gate_shexp.weight`: `[expert_ffn, hidden]`  (shared expert)
- `ffn_up_shexp.weight`: `[expert_ffn, hidden]`
- `ffn_down_shexp.weight`: `[hidden, expert_ffn]`
- `exp_probs_b.bias`: `[num_experts]`              (shared-expert sigmoid bias)
- `num_experts_used`: top-k (typically 8) routed per token
- (optional) routed-expert scoring scale `routed_scaling_factor`

Output: `[seq_len, hidden_size]`.

Math:

```
For each token t:
  scores[t,:] = hidden[t] @ ffn_gate_inp.T         # [num_experts]
  weights[t,:] = softmax(scores[t,:])              # or sigmoid-style
  top_idx[t] = topk(weights[t,:], k)
  routed = sum_e in top_idx[t] of w_e * (swiglu(gate_e @ h, up_e @ h) @ down_e)
  shared = swiglu(gate_shexp @ h, up_shexp @ h) @ down_shexp
  output[t] = shared + routed * routed_scaling_factor [+ scaled exp_probs_b.dot]
```

For memory:

- Only the **top-k expert weights per layer** need to be resident to compute a layer (typically 8 of N where N can be 384). Streaming layer + selective expert streaming = path back to the 2 GB target.
- For now: pre-dequantize all `*_exps` weights for one layer, compute MoE, drop. Acceptable on Q4_K_XL for 64-expert models; for 384-expert models we will need expert-streaming afterwards.

### 5.2 New module: `src/inference/mla.rs`

MLA forward path. Inputs (per layer):

- hidden: `[seq_len, hidden_size]`
- `attn_q_a.weight`: `[q_lora_rank, hidden]` — Q down
- `attn_q_a_norm.weight`: `[q_lora_rank]` — RMSNorm on Q latent
- `attn_q_b.weight`: `[num_heads * qk_head_dim, q_lora_rank]` — Q up
- `attn_kv_a_mqa.weight`: `[kv_lora_rank + qk_rope_head_dim, hidden]` — KV down with absorbed rope
- `attn_kv_a_norm.weight`: `[kv_lora_rank]` — RMSNorm on KV latent
- `attn_k_b.weight`: `[num_kv_heads * qk_head_dim, kv_lora_rank]` — K up (latent context)
- `attn_v_b.weight`: `[num_kv_heads * v_head_dim, kv_lora_rank]` — V up
- `attn_output.weight`: `[hidden, num_heads * v_head_dim]`
- RoPE applied to the absorbed Q head-dim part of `q_b` and `k_b` only.

Math:

```
q_lat = rms_norm(hidden @ q_a.T, q_a_norm)           # [seq_len, q_lora_rank]
q     = q_lat @ q_b.T                                # [seq_len, n_heads * qk_head_dim]
kv_lat = rms_norm(hidden @ kv_a.T, kv_a_norm)        # [seq_len, kv_lora_rank]
k     = kv_lat @ k_b.T                               # [seq_len, n_kv_heads * qk_head_dim]
v     = kv_lat @ v_b.T                               # [seq_len, n_kv_heads * v_head_dim]
Q, K, V reshape to heads, apply rope to Q[..rope_dim] and K[..rope_dim]
standard scaled-dot attention softmax over keys
context = attn @ V
attn_out = context @ attn_output.weight.T
```

This is normal attention once projections are decomposed. KV cache stores `kv_lat` (compressed) — that's why DeepSeek models compress to ~64 KB per layer instead of MB. Without re-using that compression we waste RAM, but layer-streaming still saves us.

### 5.3 What stays the same

- `madvise(MADV_DONTNEED)` layer streaming.
- `embed_lookup_mmap` (no change).
- `lm_head_projection` (no change; we might need to verify the lm-head layout matches the rest but Dense / MoE both share token_embd or output.weight).
- Speculative heads — `nextn.*` for DeepSeek-2 / Kimi / GLM-DSA still applies (MTP = "Multi-Token Prediction"). The `SpeculativeHead` module already loads any tensor with `nextn.*` prefix; wiring into MoE-FFN + MLA forward is part of M3 below.

### 5.4 What we don't do yet (defer)

- Full **expert-streaming** lazy-load (load only top-k experts per-layer). Significant work; needed only if resident-MoE exceeds memory budget on Pi.
- **MTP loss/verification**. DeepSeek-2-style speculative decoding with MTP heads. We'll wire the draft heads but defer the verification logic for a future milestone.
- **Fused `wkv_b` absorption matrices** for MLA (DeepSeek V3.2 added a `wkv_b` that absorbs the k_b/v_b into the kv_lat input). Not needed for Kimi K2.6 base or GLM-5.2 base.

---

## 6. Roadmap (concrete, milestone-based)

| Milestone | What                                                                           | Validates against            |
|-----------|--------------------------------------------------------------------------------|------------------------------|
| **M1**    | Add `MODELS.md` + intake checklist (`scripts/intake_gguf.py`)                  | (no model needed)            |
| **M2**    | Add `model/arch.rs::DeepSeek2` architecture enum + capability report          | reports only                 |
| **M3**    | Implement `inference/moe.rs::moe_forward` (routed + shared)                   | unit tests against numpy    |
| **M4**    | Implement `inference/mla.rs::mla_forward`                                     | unit tests against numpy    |
| **M5**    | Wire M2-M4 into `forward_native` per-layer branch                             | `cargo test --lib` builds    |
| **M6**    | Add `moe_fwd.rs` binary that loads both models' first layer (shard 1 needed)   | 1-token forward comparison   |
| **M7**    | Add MTP / `nextn.*` head support (deferred verification)                       | tests skipped, "loaded" check |
| **M8**    | Validate Kimi K2.6 1-token forward vs llama.cpp reference                     | logit close to llama-server |
| **M9**    | Validate GLM-5.2 1-token forward vs llama.cpp reference                       | logit close to llama-server |
| **M10**   | Greedy decode test on Kimi                                                      | coherent shape              |
| **M11**   | Greedy decode test on GLM                                                       | coherent shape              |
| **M12**   | Layer-stream + top-k expert-stream attempt on Pi 5 8GB; RAM check              | peak RSS bound              |

---

## 7. Validation strategy

We can't trust any new implementation without cross-check. The reference for both these models is **llama.cpp** (`rust/llama.cpp/` submodule is already vendored). The plan:

1. Build llama.cpp + load Kimi-K2.6-shard-1 via `llama-cli` with `--log-disable --seed 0`.
2. Run a 1-token forward on the same prompt + the first layer.
3. Compute reference output (logits).
4. Build leafcutter native.
5. Run `Rust` 1-token forward on the same prompt + first layer through `cargo run --release --bin moe_fwd`.
6. Cosine similarity on final layer output / logits.

If cos-sim > 0.99, mark layer 0 as verified. Repeat for layer 4 (middle), and the final layer.

A Python reference (`scripts/ref_mla_moe.py`) using Pytorch on the same weights is the gold standard for math.

---

## 8. Status as of 2026-06-19

- Existing baseline: **129 cargo tests pass** (was 123, +6 new tests across the new arch-detect + MoE module). 1 pre-existing kernel test failure unchanged. No regressions to any validated model.
- Models confirmed on disk:
  - Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf ✅ validated
  - Meta-Llama-3.1-70B-Instruct-Q4_K_S.gguf ✅ validated
  - Ministral-3B / 8B ✅ validated
  - **Kimi-K2.6-UD-Q4_K_XL-00001-of-00014.gguf** ✅ shard-1 with full metadata
  - **GLM-5.2-UD-Q4_K_XL-00001-of-00011.gguf** ✅ shard-1 with full metadata

- New files created 2026-06-19:
  - `scripts/intake_gguf.py` — per-model intake checklist (architecture / dims / capabilities / quant-summary).
  - `scripts/ref_mla_moe.py` — Python reference forward for both MoE and MLA.
  - `inference/moe.rs` — `MoeConfig` + `moe_forward_one_token`/`moe_forward` + 3 unit tests.
  - `model/arch.rs` — added `DeepSeek2` + `GlmDsa` enum variants + detection map + 3 unit tests.

- Code status: scaffolding complete; routing math validated against Python reference for random tensors. Engine wiring (forward path branches) and MLA forward implementation pending next session — the current MoE compiles but is not yet called from `engine.rs::forward_native` because we want to verify with full-shard runs first.

- Pre-existing build breakages fixed in passing (so the new arch enum and MoE module could compile against a green baseline):
  - `src/main.rs` — pulled `BaseTokenizer` into scope; fixed `tok.decode(&tokens, false)` → `decode(&tokens)`.
  - `src/bin/check_tok.rs` — same trait/arg fixes.
  - `src/main.rs` `cli.command` arm — name-only field destructuring (was a 2-arg issue in the `engage` path).

---

## 9. Memory / RAM expectations

Q4_K_XL is about 4.5 bits per param. Estimates (computed by
`scripts/intake_gguf.py`):

| Model          | Total file | Total params (est.) | Resident-per-layer | Notes                         |
|----------------|------------|---------------------|---------------------|-------------------------------|
| Kimi K2.6      | ~250 GB    | ~1.0 T (200B-1T MoE)| per-expert slice    | streaming top-k experts needed|
| GLM-5.2        | ~85 GB     | ~745B (MoE)         | MoE shared+routed   | top-k experts streaming       |
| Meta-70B       | 40.3 GB    | 70B (dense)         | 535 MB              | already validated             |
| Llama-3B       | 1.9 GB     | 3B (dense)          | 250 MB              | already validated             |

Realistically: full 1-token forward on Kimi K2.6 needs at least ~6-10 GB just for routing weights, plus compressed expert slice in transit. With the existing layer-streaming pattern + top-k expert unloading, target is **3 GB peak on Pi 5**.

---

## 10. Open questions, in light of shard-1 metadata

| # | Question                                | Status        | Resolution                                                                  |
|---|------------------------------------------|---------------|------------------------------------------------------------------------------|
| 1 | MQA vs full GQA for MLA                  | ✅ Resolved    | Both are MQA (`head_count_kv=1`). Kimi GLM-DSA confirm.                    |
| 2 | QK rope head dim                         | ✅ Resolved    | Kimi: qk_nope=192, qk_rope=384. GLM-DSA: qk_nope=256, qk_rope=320.        |
| 3 | MoE scoring                              | ✅ Resolved    | Both use `gating_func=2` ⇒ sigmoid (Kimi) / sigmoid (GLM-DSA).             |
| 4 | Shared-expert formula                    | ⚠ Defer        | Confirmed additive (DeepSeek standard). V3 sigmoid-bias math not yet exercised. |
| 5 | MTP head layout                          | ⚠ Partial      | GLM-DSA reports `nextn_predict_layers=1` ⇒ load `nextn.*` but verify on shard 1. |
| 6 | bf16 vs f32 for routed experts          | ✅ Resolved    | Confirmed Q4_K_XL only; no bf16 detours needed.                            |
| 7 | **GLM-DSA indexer (top-k tokens)**        | ⚠ New          | `glm-dsa.attention.indexer.top_k=2048` with 32 indexer heads — sparse-attention indexer ahead. |

**Resolved questions no longer block M4-M5** (math is clear).
**Outstanding blockers for real-model validation (M6-M11)**: items 4 and 7 in the table require inspecting actual tensor values for layer 0 / layer 1 of the shards; once we have a full shard group on disk, the math will be validated against llama.cpp's reference output.

---

*End of report — see CHANGELOG.md for build/regression status, handoff-leafcutterllm.md for prior context.*
