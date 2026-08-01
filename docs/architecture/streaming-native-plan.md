

---

## I. Streaming Native Rust Forward Pass — Implementation Plan

> **Date:** 2026-07-30
> **Status:** COMPLETE / SUPERSEDED — see `LEAFCUTTER_STRATEGY.md` (2026-08-01).
> This safetensors streaming plan was validated but then superseded by the native
> **GGUF** engine, which is what `leafcutter run ornith` uses today (coherent chat,
> ~8.1 GB peak RAM, 1.2–1.65 tok/s). Historical record retained below.
> **Goal:** Beat AirLLM in speed. Run large models on small hardware. Pure Rust, minimal deps.

### Progress (2026-07-30 end of day)

| Milestone | Status |
|-----------|--------|
| Architecture validated (32 layers process, 137s) | ✅ |
| Layer 0 token 0 hidden state matches Python | ✅ (Rust=0.0278 vs Python=0.0276) |
| Real DeltaNet forward (vs placeholder) | ✅ |
| Bug #1 — State update order (decay→predict→update) | ✅ Fixed |
| Bug #2 — Qwen3_5MoeRMSNorm `(1 + weight)` | ✅ Fixed (line 750 + 572/582) |
| Bug #3 — Sigmoid attention gate (not silu) | ✅ Fixed (line 663) |
| Bug #4 — GLM-style split RoPE | ✅ Fixed (lines 571-599) |
| Bug #5 — Conv1d buffer shift | ✅ Fixed |
| Bug #6 — Decay computation (A_log convention) | 🔧 Identified, not yet applied |
| " Paris" logit gap: 16.71 → 16.10 (from +0.150 to reference 16.25) | 🔧 In progress |

See `strategy.md` for the full debugging plan and architecture reference.

### Background

We have a working Python subprocess backend (`leafcutter run <dir> --engine safetensor`) that streams at ~12s/tok. But:

- AirLLM already does this in Python. We'd just be "another AirLLM."
- The native Rust forward pass attempt (`ornith_forward.rs`) FAILED — naive triple-loop matmul was 100x SLOWER than Python's BLAS-backed matmul. We also loaded the ENTIRE model into RAM (18GB), the opposite of streaming.
- The bottleneck wasn't the matmul itself — it was using the WRONG matmul (naive triple loop instead of the existing BLAS-like `Tensor::matmul` which uses `matrixmultiply::sgemm`).
- And the bottleneck wasn't compute — it was loading 18GB into RAM instead of streaming layer-by-layer (~400MB peak).

### The Fix: Streaming + Existing Backend

**Two key changes:**

1. **Streaming I/O:** Load ONE layer's weights from disk at a time. Read embedding as a single row (8KB). Discard weights after computing each layer. Peak RAM = one layer (~400MB for 9B model).
2. **Use `Tensor::matmul`:** Already exists in `src/model/tensor.rs`. Dispatches to `matrixmultiply::sgemm` (BLAS-like) for matrices ≥256 elements, SIMD for small. Near-BLAS performance. No new matmul code needed.

### File Inventory (what exists now)

| File | Status | Notes |
|------|--------|-------|
| `src/safetensors_loader.rs` | ✅ Working | Has `read_tensor_f32` + NEW `read_tensor_slice_f32` |
| `src/bpe_tokenizer.rs` | ✅ Working | Round-trip verified |
| `src/ornith_config.rs` | ✅ Working | Parses config.json |
| `src/model/tensor.rs` | ✅ Working | `Tensor::matmul` uses `matrixmultiply::sgemm` |
| `src/backend/cpu.rs` | ✅ Working | SIMD + BLAS-like kernels |
| `src/streaming_ornith.rs` | 🔄 Drafted | Has borrow-checker error to fix |
| `src/ornith_forward.rs` | ❌ Wrong arch | To be deleted (loads whole model) |
| `src/safetensor_tensors.rs` | ❌ Wrong arch | To be deleted (clone-on-every-access) |
| `src/engine_keymap.rs` | ❌ Not needed | For old architecture |

### Step-by-Step Plan

#### STEP 1: Fix the borrow checker error in `streaming_ornith.rs`

**Problem:** `forward_one_token` borrows `self` mutably (to read from `self.shards`), but also borrows `self` immutably (to read `self.cfg`). Fix: clone the config values we need at the start, or split the borrow.

**File:** `rust/src/streaming_ornith.rs`

**What to do:**

At the top of `forward_one_token`, clone the config values needed:
```rust
let h = self.cfg.hidden_size;
let num_layers = self.cfg.num_hidden_layers;
let layer_types: Vec<String> = self.cfg.layer_types.clone();
let rms_eps = self.cfg.rms_norm_eps;
let vocab_size = self.cfg.vocab_size;
// ... other config fields as needed
```

Then use these local variables instead of `self.cfg.*` throughout. The `self.shards` mutable borrow won't conflict.

**Verify:** `cargo build --release --no-default-features --bin test_streaming_forward` compiles with zero errors.

#### STEP 2: Remove stale files from the old architecture

**Files to delete:**
- `rust/src/ornith_forward.rs`
- `rust/src/safetensor_tensors.rs`
- `rust/src/engine_keymap.rs`

**File to edit:** `rust/src/lib.rs` — remove the `pub mod` lines for those three files.

**Verify:** `cargo build --release --no-default-features --bin leafcutter` still compiles.

#### STEP 3: Fix the DeltaNet forward (linear_attention layers)

**Current state:** The `deltanet_forward` method in `streaming_ornith.rs` is a PLACEHOLDER. It does `out[i] = q[i] * v[i]` which is WRONG. This will produce garbage output for the 24 linear_attention layers.

**What's needed:** The real DeltaNet forward pass for Ornith's linear_attention layers. This is the hardest part. The weights available are:
- `linear_attn.in_proj_qkv.weight` — [3*hidden, hidden] (or split)
- `linear_attn.in_proj_a.weight` — [hidden, hidden]
- `linear_attn.in_proj_b.weight` — [hidden, hidden]
- `linear_attn.in_proj_z.weight` — [hidden, hidden]
- `linear_attn.conv1d.weight` — [hidden, 1, conv_kernel_dim]
- `linear_attn.A_log` — [hidden] (log of the diagonal)
- `linear_attn.dt_bias` — [hidden]
- `linear_attn.norm.weight` — [hidden]
- `linear_attn.out_proj.weight` — [hidden, hidden]

**Reference:** Look at the existing `rust/src/inference/deltanet.rs` — it has DeltaNet code already, but wired for GGUF tensor names. The math is the same; just the weight names differ.

**Key DeltaNet operations (simplified for single-token):**
1. QKV = in_proj_qkv @ x  → split into Q, K, V (each [hidden])
2. A = exp(A_log) — the diagonal decay rates
3. Conv1d: if pos > 0, add conv state (short convolution)
4. State update: S = S * diag(A) + V ⊗ K  (outer product)
5. Output = Q @ S  (state retrieval)
6. Gate with in_proj_z: z = in_proj_z @ x; out = out * silu(z)
7. Norm: out = out * norm.weight
8. Project: out = out_proj @ out

**For first token (pos=0):** Conv1d and state update simplify — no previous state. State starts at zero. After update, S = V ⊗ K. Output = Q @ (V ⊗ K) = (Q · K) * V — which is just scaled attention for one token.

**Verify:** After implementing, the test should run without crashing. Output may still be wrong (placeholder attention) but won't hang.

#### STEP 4: Fix the full attention forward

**Current state:** The `attention_forward` method is SIMPLIFIED. For a single token at pos=0, it sets output = V (since attention to itself is trivially 1.0). This is actually CORRECT for the first token — no KV cache, single token, softmax of one element = 1.0, so output = V.

**What's needed for multi-token:** KV cache + RoPE. For now (single-token test), the current code is correct. Skip this for Step 4 and come back when doing multi-token generation.

**Verify:** Current code compiles and runs for single token.

#### STEP 5: Fix the lm_head (avoid loading 2GB)

**Current state:** `forward_one_token` reads the ENTIRE `lm_head.weight` (248320 × 4096 × 2 = 2GB BF16 → 4GB f32) into memory. This defeats the streaming architecture.

**Fix:** Read lm_head in CHUNKS. For each chunk of `chunk_size` vocab rows:
1. Read `chunk_size × hidden` elements from disk (e.g., 1024 rows × 4096 = 4M elements = 8MB BF16)
2. Compute `logits_chunk = hidden @ chunk.T` → `chunk_size` logits
3. Copy logits to the right position in the output vector
4. Discard the chunk

**Code sketch:**
```rust
let lm_head_meta = self.shards.lookup("lm_head.weight").ok_or("missing lm_head")?;
// lm_head shape: [vocab, hidden]
let chunk_size = 1024;
let mut logits = vec![0.0f32; vocab_size];
let lm_head_name = "lm_head.weight";
for chunk_start in (0..vocab_size).step_by(chunk_size) {
    let chunk_end = (chunk_start + chunk_size).min(vocab_size);
    let n_rows = chunk_end - chunk_start;
    // Read chunk_size rows starting at row chunk_start
    let chunk = self.shards.read_tensor_slice_f32(
        lm_head_name,
        chunk_start * h,  // offset in elements
        n_rows * h,       // count in elements
    )?;
    let chunk_t = Tensor::from_vec(chunk, vec![n_rows, h]);
    let hidden_t = Tensor::from_vec(hidden.clone(), vec![1, h]);
    let logits_chunk = hidden_t.matmul(&chunk_t.transpose());
    logits[chunk_start..chunk_end].copy_from_slice(&logits_chunk.data);
}
```

**Verify:** Peak RAM during lm_head computation stays under ~50MB (one chunk), not 4GB.

#### STEP 6: Run the streaming forward end-to-end

**Run:** `cargo build --release --no-default-features --bin test_streaming_forward && timeout 300 ./target/release/test_streaming_forward`

**Expect (with placeholder DeltaNet):**
- Embed reads in <1ms (8KB)
- Each layer loads in ~1-2s (disk read + BF16→f32 convert)
- Each layer computes in <1s (BLAS matmul)
- Total: ~30-60s for 32 layers (dominated by disk I/O)
- lm_head: ~5-10s (chunked reads)
- Top-5 predictions print

**If output is garbage:** That's expected if DeltaNet is still placeholder. The pipeline WORKS (no crash, reasonable timing), just the numbers are wrong.

**If it hangs:** Check which layer it hangs on. The eprintln timing logs will show exactly where.

#### STEP 7: Validate correctness against Python

**Reference:** The Python backend (`--engine safetensor`) produces "Paris" as top-1 for "The capital of France is".

**Test:** Run the streaming forward on the SAME prompt. Compare top-5 logits with Python's output.

**If top-1 is "Paris":** The architecture is correct. Move to Step 8.
**If top-1 is NOT "Paris":** The DeltaNet placeholder is producing wrong values. Implement real DeltaNet (Step 3) and re-test.

#### STEP 8: Multi-token generation loop

**Add a `generate` method to `StreamingOrnith`:**
```rust
pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, String> {
    let mut ids = self.tok.encode(prompt, 1024);
    for i in 0..max_tokens {
        let last = *ids.last().unwrap() as i32;
        let logits = self.forward_one_token(last, ids.len() - 1)?;
        let next = Self::argmax(&logits);
        ids.push(next as i32);
        if next == self.tok.eos_token_id() { break; }
        // Print token
        let text = self.tok.decode(&[next as i32]);
        print!("{text}");
        std::io::Write::flush(&mut std::io::stdout()).ok()?;
    }
    Ok(self.tok.decode(&ids))
}
```

**Note:** This won't produce coherent multi-token output yet because:
- No KV cache (each token re-processes from scratch)
- DeltaNet state is not carried between tokens
- Attention layers don't have RoPE or KV cache

But it will produce SOMETHING — each token independently. That's enough to verify the pipeline.

#### STEP 9: Measure and compare

**Run the same prompt through both backends:**
```bash
# Python backend (reference)
time ./target/release/leafcutter run "/home/xander/Downloads/models/ornith safetensor" --engine safetensor --prompt "The capital of France is"

# Native Rust backend
time ./target/release/leafcutter run "/home/xander/Downloads/models/ornith safetensor" --engine native --prompt "The capital of France is"
```

**Measure:**
- Time per token (target: <12s/tok to beat Python)
- Peak RSS (target: <500MB, vs Python's ~4GB)
- Output correctness (does it say "Paris"?)

### File Summary After All Steps

| File | What it does |
|------|-------------|
| `src/streaming_ornith.rs` | Main streaming forward pass |
| `src/safetensors_loader.rs` | Disk I/O + BF16→f32 (slice reads) |
| `src/bpe_tokenizer.rs` | Tokenizer (unchanged) |
| `src/ornith_config.rs` | Config (unchanged) |
| `src/model/tensor.rs` | Tensor + BLAS matmul (unchanged) |
| DELETED: `src/ornith_forward.rs` | Old wrong architecture |
| DELETED: `src/safetensor_tensors.rs` | Clone-on-access cache |
| DELETED: `src/engine_keymap.rs` | GGUF name mapping |

### What to Do RIGHT NOW (if pair-programming)

1. Open `rust/src/streaming_ornith.rs`
2. Fix the borrow checker: clone `self.cfg` fields to locals at top of `forward_one_token` and all helper methods
3. Delete the three files marked DELETED above
4. Remove their `pub mod` lines from `lib.rs`
5. Run: `cargo build --release --no-default-features --bin test_streaming_forward`
6. If it compiles, run: `timeout 120 ./target/release/test_streaming_forward`
7. Report whether it produces output or crashes

Then I (Hermes) will verify the timing and correctness, and we proceed to Step 3 (real DeltaNet).
