# LeafcutterLLM Strategy — Streaming Native Rust Forward Pass

> **Date:** 2026-07-30
> **Goal:** Beat AirLLM in speed. Run large models on small hardware. Pure Rust, minimal deps.
> **Workflow:** Hermes writes step-by-step plans. You build. Hermes reviews.

---

## Current State (what's done and working)

### ✅ Working — Python subprocess backend
`leafcutter run <dir> --engine safetensor` streams coherent English at ~12s/tok.
This is our reference for correctness — it produces "Paris" for "The capital of France is".

### ✅ Working — Rust infrastructure
- `src/safetensors_loader.rs` — reads safetensors from disk, has `read_tensor_f32` + `read_tensor_slice_f32` (reads only N elements from a tensor, not the whole thing)
- `src/bpe_tokenizer.rs` — BPE tokenizer, round-trip verified
- `src/ornith_config.rs` — parses config.json
- `src/model/tensor.rs` — `Tensor::matmul` uses `matrixmultiply::sgemm` (BLAS-like speed)
- `src/backend/cpu.rs` — SIMD + BLAS-like kernels (already exist, already fast)

### ✅ Working — Streaming pipeline validated end-to-end
`src/streaming_ornith.rs` streams ALL 32 layers (137s, ~4.3s/layer):
- Embedding reads ONE row (8KB) not the whole 2GB table
- Each layer loads from disk, computes, discards (~400MB peak RAM)
- Uses `Tensor::matmul` for BLAS-like compute
- Produces 248,320 logits
- Output is "2" (garbage) because DeltaNet is currently a placeholder

### 🔧 In progress — Real DeltaNet forward
I just wrote the real DeltaNet into `streaming_ornith.rs` but it's NOT compiled or tested yet.
You need to compile it, fix any errors, and run it.

---

## What needs to happen (in order)

### STEP 1: Compile and test the real DeltaNet

**File:** `rust/src/streaming_ornith.rs`

The `deltanet_forward` method has been rewritten with the real DeltaNet operations:
1. Input RMSNorm
2. QKV projection (conv_dim = 8192)
3. Causal Conv1d (kernel=4) + SiLU — for pos=0, uses last kernel tap only
4. Split into Q (2048), K (2048), V (4096)
5. L2-normalize Q and K per-head
6. Scale Q by 1/sqrt(d_k)
7. Decay rates: exp(softplus(alpha + dt_bias) * A) where A = -exp(A_log)
8. Beta gates: sigmoid(hidden @ in_proj_b)
9. Delta rule: for pos=0, state=0, so output = beta * (q·k) * v
10. Per-head RMSNorm with norm.weight
11. Z-gate: output *= silu(hidden @ in_proj_z)
12. Output projection

**What you do:**
```bash
cd /home/xander/Documents/portfolio/LeafcutterLLM/rust
cargo build --release --no-default-features --bin test_streaming_forward 2>&1 | grep "^error" | head -10
```

Fix any compile errors. Common issues:
- Missing `scale_factor` variable (I left a dead line in, remove it)
- Borrow checker (clone `self.cfg` fields to locals at top of methods)
- Missing `softplus` or `sigmoid` (should be at bottom of file now)

Then run:
```bash
timeout 300 ./target/release/test_streaming_forward 2>&1 | tail -30
```

**What to look for:**
- Does it compile?
- Does layer 0 print deltanet OUT values?
- Does it complete all 32 layers?
- What's the top-5 predictions? (hopefully "Paris" or at least something English-like)

### STEP 2: Fix lm_head to stream in chunks (avoid loading 2GB)

**File:** `rust/src/streaming_ornith.rs`, in `forward_one_token`, section 4 (lm_head)

**Current problem:** The code reads the ENTIRE lm_head tensor (248320 × 4096 × 2 = 2GB BF16 → 4GB f32) in one shot. This defeats the streaming architecture.

**Replace the lm_head section with:**
```rust
// 4. LM head: stream in chunks to avoid loading 2GB at once.
// lm_head.weight is [vocab, hidden]. Read 1024 rows at a time.
let lm_head_name = "lm_head.weight";
let mut logits = vec![0.0f32; vocab_size];
let chunk_size = 1024;
for chunk_start in (0..vocab_size).step_by(chunk_size) {
    let chunk_end = (chunk_start + chunk_size).min(vocab_size);
    let n_rows = chunk_end - chunk_start;
    let chunk = self.shards.read_tensor_slice_f32(
        lm_head_name,
        chunk_start * h,   // offset in elements
        n_rows * h,        // count in elements
    )?;
    let chunk_t = Tensor::from_vec(chunk, vec![n_rows, h]);
    let hidden_t = Tensor::from_vec(hidden.clone(), vec![1, h]);
    let logits_chunk = hidden_t.matmul(&chunk_t.transpose());
    logits[chunk_start..chunk_end].copy_from_slice(&logits_chunk.data);
}
```

**Verify:** `cargo build` compiles, and `test_streaming_forward` runs with lower peak RAM.

### STEP 3: Remove dead code

Delete these files (they're the old wrong architecture, no longer used):
- `rust/src/ornith_forward.rs`
- `rust/src/safetensor_tensors.rs`
- `rust/src/engine_keymap.rs`

Remove their `pub mod` lines from `rust/src/lib.rs`.

Also remove the dead `scale_factor` line in `deltanet_forward` if still present.

### STEP 4: Validate correctness

Run the streaming forward and compare with Python:

```bash
# Native Rust
cd /home/xander/Documents/portfolio/LeafcutterLLM/rust
timeout 300 ./target/release/test_streaming_forward 2>&1 | tail -20
```

**Success criteria:**
- Top-1 prediction is "Paris" or a real word (not "2" or garbage)
- All 32 layers complete without crash
- Time < 300s for one token

If output is still garbage, the most likely issues are:
1. Conv1d weight layout is wrong (might be [conv_dim, conv_k] not [conv_k, conv_dim])
2. A_log convention differs (might already be A, not log of A)
3. QKV split order might be different (Q+K+V vs interleaved)
4. Beta/decay might not apply for pos=0 in the real model

Report the top-5 output and I'll diagnose which issue it is.

### STEP 5: Wire as `--engine native` in the leafcutter CLI

In `rust/src/bin/leafcutter.rs` (or wherever `cmd_run` is), add:
- If `--engine native` and the model dir has safetensors, use `StreamingOrnith`
- Add a `generate` method that loops: `forward_one_token` → argmax → decode → print

### STEP 6: Multi-token generation

Add a generation loop to `StreamingOrnith`:
```rust
pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, String> {
    let mut ids = self.tok.encode(prompt, 1024);
    for i in 0..max_tokens {
        let last = *ids.last().unwrap() as i32;
        let logits = self.forward_one_token(last, ids.len() - 1)?;
        let next = Self::argmax(&logits);
        ids.push(next as i32);
        let text = self.tok.decode(&[next as i32]);
        print!("{text}");
        std::io::Write::flush(&mut std::io::stdout()).ok()?;
        if text.contains("<|end|>") { break; }
    }
    Ok(self.tok.decode(&ids))
}
```

Note: Multi-token won't produce coherent text yet because:
- No KV cache for attention layers (each token reprocessed)
- No DeltaNet state carried between tokens
- No RoPE for position encoding

But it will prove the pipeline works end-to-end.

### STEP 7: Measure and compare

```bash
# Python backend (reference)
time ./target/release/leafcutter run "/home/xander/Downloads/models/ornith safetensor" \
    --engine safetensor --prompt "The capital of France is" --max-tokens 5

# Native Rust
time ./target/release/leafcutter run "/home/xander/Downloads/models/ornith safetensor" \
    --engine native --prompt "The capital of France is" --max-tokens 5
```

**Targets:**
- Time per token: <12s (beat Python's ~12s/tok)
- Peak RSS: <500MB (vs Python's ~4GB, vs AirLLM's ~2GB)
- Correctness: produces "Paris"

---

## Ornith-1.0-9B Architecture Reference

### Config values
- hidden_size: 4096
- num_hidden_layers: 32 (24 linear_attention + 8 full_attention)
- num_attention_heads: 16, num_key_value_heads: 4, head_dim: 256
- linear_num_key_heads: 16, linear_num_value_heads: 32
- linear_key_head_dim: 128, linear_value_head_dim: 128
- linear_conv_kernel_dim: 4
- intermediate_size: 12288
- vocab_size: 248320
- rms_norm_eps: 1e-6

### Linear attention (DeltaNet) layer weights
```
input_layernorm.weight              [4096]
linear_attn.in_proj_qkv.weight      [8192, 4096]  → Q(2048) + K(2048) + V(4096)
linear_attn.in_proj_a.weight         [32, 4096]    → decay projection
linear_attn.in_proj_b.weight         [32, 4096]    → beta gate
linear_attn.in_proj_z.weight         [4096, 4096]  → z-gate
linear_attn.conv1d.weight            [4, 8192]     → short conv (kernel=4)
linear_attn.A_log                    [32]          → log of decay diagonal
linear_attn.dt_bias                  [32]          → bias for decay
linear_attn.norm.weight              [128]         → per-head norm
linear_attn.out_proj.weight          [4096, 4096]  → output projection
post_attention_layernorm.weight      [4096]
mlp.gate_proj.weight                 [12288, 4096]
mlp.up_proj.weight                   [12288, 4096]
mlp.down_proj.weight                 [4096, 12288]
```

### Full attention layer weights
```
input_layernorm.weight              [4096]
self_attn.q_proj.weight             [4096, 4096]
self_attn.k_proj.weight             [1024, 4096]  (4 heads × 256 dim)
self_attn.v_proj.weight             [1024, 4096]
self_attn.o_proj.weight             [4096, 4096]
post_attention_layernorm.weight     [4096]
mlp.gate_proj.weight                [12288, 4096]
mlp.up_proj.weight                  [12288, 4096]
mlp.down_proj.weight                [4096, 12288]
```

### DeltaNet forward (single token, pos=0)
1. normed = rmsnorm(hidden, input_layernorm)
2. qkv = normed @ in_proj_qkv^T  → [8192]
3. conv: for pos=0, out[c] = conv_w[3*8192 + c] * qkv[c], then SiLU
4. Split: Q[0:2048], K[2048:4096], V[4096:8192]
5. L2-normalize Q and K per-head (128 dims each, 16 heads)
6. Q *= 1/sqrt(128)
7. alpha = hidden @ in_proj_a^T  → [32]
8. A = -exp(A_log)  → [32]
9. decay = exp(softplus(alpha + dt_bias) * A)  → [32]
10. beta = sigmoid(hidden @ in_proj_b^T)  → [32]
11. For each (qk_head, v_head):
    - qk_dot = sum(Q[head] * K[head])
    - output[v_head] = beta[v_head] * qk_dot * V[v_head]
    (This is the pos=0 simplification: state=0, so S = beta * V⊗K, out = S@q = beta*(q·k)*v)
12. Per-head RMSNorm with norm.weight
13. z = hidden @ in_proj_z^T → [4096]
14. output *= silu(z)
15. result = output @ out_proj^T → [4096]

### Full attention forward (single token, pos=0)
1. normed = rmsnorm(hidden, input_layernorm)
2. Q = normed @ q_proj^T  → [4096] (16 heads × 256 dim)
3. K = normed @ k_proj^T  → [1024] (4 heads × 256 dim)
4. V = normed @ v_proj^T  → [1024]
5. For pos=0: attention to itself, softmax of 1 element = 1.0
6. Output = V (broadcast across heads via GQA)
7. result = output @ o_proj^T → [4096]

### MLP forward (SwiGLU)
1. gate = hidden @ gate_proj^T  → [12288]
2. up = hidden @ up_proj^T      → [12288]
3. inter = silu(gate) * up
4. result = inter @ down_proj^T → [4096]

---

## File inventory

| File | Status | Action |
|------|--------|--------|
| `src/safetensors_loader.rs` | ✅ Working | Has slice reads. Done. |
| `src/bpe_tokenizer.rs` | ✅ Working | Done. |
| `src/ornith_config.rs` | ✅ Working | Done. |
| `src/model/tensor.rs` | ✅ Working | Uses BLAS. Done. |
| `src/streaming_ornith.rs` | 🔧 Needs compile+test | Real DeltaNet just written, not tested |
| `src/bin/test_streaming_forward.rs` | ✅ Working | Test binary exists |
| `src/ornith_forward.rs` | ❌ Delete | Old architecture |
| `src/safetensor_tensors.rs` | ❌ Delete | Old cache, wrong |
| `src/engine_keymap.rs` | ❌ Delete | Not needed |

---

## What to do RIGHT NOW

1. `cd /home/xander/Documents/portfolio/LeafcutterLLM/rust`
2. `cargo build --release --no-default-features --bin test_streaming_forward 2>&1 | grep "^error" | head -10`
3. Fix any compile errors (check for dead `scale_factor` line, borrow issues)
4. `timeout 300 ./target/release/test_streaming_forward 2>&1 | tail -30`
5. Report: does it compile? Does it run? What's the top-5 output?

Then I'll diagnose and give you the next step.
