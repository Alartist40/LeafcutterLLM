# Path B — Native Rust Safetensors Forward Pass

**Date:** 2026-07-29 (checkpoint after first session)
**Goal:** Native Rust inference engine that reads `.safetensors` files
directly, no Python dependency. 5-10x faster than the Python subprocess
backend, fewer dependencies than AirLLM, smaller binary than Colibri.

> **Update 2026-08-01 (project wrap-up):** This safetensors path is
> superseded by the native **GGUF** engine (`gguf_provider.rs` +
> `streaming_ornith.rs`), which ships in `leafcutter run ornith`. The
> engine now reads `.gguf` files directly and produces coherent chat with
> no Python. Historical record retained below.

## Status

### ✅ Done (verified working)

| Component | File | Status |
|-----------|------|--------|
| Safetensors loader | `rust/src/safetensors_loader.rs` | 760 Ornith tensors loaded, BF16/F16/F32 dequant matches Python exactly |
| BPE tokenizer | `rust/src/bpe_tokenizer.rs` | Round-trip exact match for "The capital of France is" |
| Config parser | `rust/src/ornith_config.rs` | Parses 32-layer hybrid Ornith config |
| Math kernels | `rust/src/ornith_kernels.rs` | rmsnorm, matmul, swiglu, silu, softmax, rope; 5 unit tests pass |
| First forward step | `rust/src/bin/test_first_forward.rs` | Embed → RMSNorm → QKV matmul produces sensible values |

### 🔜 Next (this checkpoint's plan)

The hardest remaining pieces:

1. **Linear attention (DeltaNet)** — `in_proj_qkv`, `conv1d`,
   `ssm_a`, `ssm_dt.bias`, `ssm_alpha`, `ssm_beta`, `attn_gate`,
   `ssm_norm`, `ssm_out`. The conv1d + delta-rule recurrence.

2. **Full attention** — `q_proj`, `k_proj`, `v_proj`, `o_proj`,
   RoPE, GQA. Standard attention with 4 KV heads.

3. **MLP (SwiGLU)** — `gate_proj`, `up_proj`, `down_proj`.

4. **Layer loop** — input_layernorm, attention, residual, post_attention_layernorm, MLP, residual.

5. **Generation loop** — token-by-token with state caching (KV cache for attention, state matrix for DeltaNet).

6. **Wire as `--engine native`** in `leafcutter run`.

7. **End-to-end test** — `leafcutter run <safetensor-dir> --engine native --prompt "The capital of France is"` → top-1 = " Paris".

## Architecture decisions

### Use leafcutter's existing Tensor type

The existing `model::tensor::Tensor` (Vec<f32> + Vec<usize>) is what
the existing DeltaNet/attention/MLP code expects. We'll wrap
safetensors data in Tensor objects, not invent a new type.

### Wrap safetensors as Tensor

Build a thin `SafetensorModel` struct that:
- Holds a `Shards` instance (our safetensors loader)
- Lazily reads tensors into Tensor objects
- Caches tensors that have been read (LRU or unbounded for small models)

### Reuse existing engine code where possible

The existing `inference/deltanet.rs`, `inference/attention.rs`, and
`inference/mlp.rs` already work for Qwen3.5/Ornith on GGUF. If we can
wrap safetensors as Tensor with the right keys, they should work
without modification.

### Keep the Python safetensor backend as fallback

The Python subprocess backend (`--engine safetensor`) is working
end-to-end. Keep it as a safety net while we build the native path.

## Ornith tensor names (from safetensors)

| Tensor name | Shape | Purpose |
|-------------|-------|---------|
| `model.language_model.embed_tokens.weight` | [248320, 4096] | Token embedding |
| `model.language_model.layers.{N}.input_layernorm.weight` | [4096] | Pre-attention norm |
| `model.language_model.layers.{N}.post_attention_layernorm.weight` | [4096] | Pre-MLP norm |
| For `linear_attention` layers: | | |
| `linear_attn.in_proj_qkv.weight` | [8192, 4096] | QKV projection |
| `linear_attn.conv1d.weight` | [4, 8192] | Causal conv1d |
| `linear_attn.A_log` | [32] | SSM decay (log space) |
| `linear_attn.dt_bias` | [32] | SSM time-step bias |
| `linear_attn.in_proj_a.weight` | [16, 4096] | SSM alpha (decay) projection |
| `linear_attn.in_proj_b.weight` | [16, 4096] | SSM beta (update gate) projection |
| `linear_attn.in_proj_z.weight` | [4096, 4096] | SSM z (gate) projection |
| `linear_attn.norm.weight` | [128] | Per-head RMSNorm for SSM output |
| `linear_attn.out_proj.weight` | [4096, 4096] | SSM output projection |
| For `full_attention` layers: | | |
| `self_attn.q_proj.weight` | [4096, 4096] | Q projection |
| `self_attn.k_proj.weight` | [1024, 4096] | K projection (GQA: 4 heads × 256) |
| `self_attn.v_proj.weight` | [1024, 4096] | V projection |
| `self_attn.o_proj.weight` | [4096, 4096] | Output projection |
| `self_attn.q_norm.weight` | [256] | Per-head Q RMSNorm |
| `self_attn.k_norm.weight` | [256] | Per-head K RMSNorm |
| For MLP (both layer types): | | |
| `mlp.gate_proj.weight` | [12288, 4096] | Gate projection |
| `mlp.up_proj.weight` | [12288, 4096] | Up projection |
| `mlp.down_proj.weight` | [4096, 12288] | Down projection |
| Final: | | |
| `model.language_model.norm.weight` | [4096] | Final norm |
| `lm_head.weight` | [248320, 4096] | LM head (un-tied) |

## Roadmap

### Phase 1 — Single-token forward (1 session)
- Wrap safetensors as Tensor
- Implement linear_attention forward for layer 0
- Implement full_attention forward for layer 3 (first full_attention layer)
- Implement MLP forward
- Implement layer loop (1 token through all 32 layers)
- Compute logits, sample top-1

### Phase 2 — Multi-token forward
- Run full prefill (multiple tokens)
- Generation loop: sample token, run 1-token forward, repeat
- KV cache for attention layers
- State cache for DeltaNet layers

### Phase 3 — Polish
- Wire as `--engine native` in `cmd_run`
- Sampling (greedy, top-p, top-k, temperature)
- Stop tokens
- REPL integration (multi-turn)
- Speed benchmarks vs Python backend

### Phase 4 — Optimization (future)
- SIMD kernels for matmul (AVX2/NEON)
- Quantize safetensors → int8/int4 on load (cut RAM 2-4x)
- Multi-threaded layer forward (parallel attention + MLP)
- KV cache quantization
