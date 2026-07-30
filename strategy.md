# LeafcutterLLM Strategy — Complete Build Guide

> **Date:** 2026-07-30
> **Goal:** Beat AirLLM in speed. Run large models on small hardware. Pure Rust, minimal deps.
> **Workflow:** Hermes writes detailed plans + code. You build, test, report. Hermes diagnoses.

---

## PART 1: CURRENT STATE

### What works
- `src/safetensors_loader.rs` — reads safetensors, has slice reads
- `src/bpe_tokenizer.rs` — BPE tokenizer, verified
- `src/ornith_config.rs` — parses config.json
- `src/model/tensor.rs` — `Tensor::matmul` uses `matrixmultiply::sgemm` (BLAS speed)
- `src/streaming_ornith.rs` — streaming forward pass, all 32 layers run (137s), real DeltaNet written but NOT compiled/tested yet
- `src/bin/test_streaming_forward.rs` — test binary

### What's broken / unfinished
- `streaming_ornith.rs` has a dead `scale_factor` variable on line 329 that will cause a compile warning (not error, but clean it)
- `streaming_ornith.rs` lm_head section (lines 122-138) loads the ENTIRE 2GB lm_head tensor — needs chunked reading
- `streaming_ornith.rs` has leftover `final_norm` lookup (line 113-116) that's unused — remove it
- `streaming_ornith.rs` attention_forward (line 400-405) uses `2*h` for q_proj but it might just be `[h, h]` — need to verify tensor shapes
- DeltaNet correctness is unverified — might produce garbage

### Files to delete (already deleted in git, but check)
- `src/ornith_forward.rs` — gone
- `src/safetensor_tensors.rs` — gone
- `src/engine_keymap.rs` — gone

---

## PART 2: STEP-BY-STEP BUILD GUIDE

### STEP 1: Clean up streaming_ornith.rs (5 min)

**File:** `rust/src/streaming_ornith.rs`

**1a. Remove dead line 329:**
Find this line:
```rust
let scale_factor = decay_h.ln() + (beta_h * qk_dot).ln(); // not right, let me just do it directly
```
Delete it entirely.

**1b. Remove unused final_norm lookup (lines 113-116):**
Find:
```rust
let final_norm = self
    .shards
    .lookup("model.language_model.norm.weight")
    .ok_or("missing final norm")?;
```
Delete it (we read the weights directly on line 117-119 already).

**1c. Remove unused `lm_head_transposed` variable (line 132):**
Find:
```rust
let lm_head_transposed = lm_head_t.transpose();
```
Delete it (we use `lm_head_t.transpose()` inline on line 133).

**Compile:**
```bash
cd /home/xander/Documents/portfolio/LeafcutterLLM/rust
cargo build --release --no-default-features --bin test_streaming_forward 2>&1 | grep "^error" | head -10
```

Fix any remaining errors. Report what they are.

---

### STEP 2: Replace lm_head with chunked reading (10 min)

**File:** `rust/src/streaming_ornith.rs`, in `forward_one_token`, replace lines 122-138 (the entire section 4 "LM head")

**Replace this block:**
```rust
// 4. LM head: logits = hidden @ lm_head.T
let lm_head = self.shards.read_tensor_f32("lm_head.weight")?;
let vocab = vocab_size;
let mut logits = vec![0.0f32; vocab];
let lm_head_t = Tensor::from_vec(lm_head, vec![vocab, h]);
let hidden_t = Tensor::from_vec(hidden, vec![1, h]);
let logits_t = hidden_t.matmul(&lm_head_t.transpose());
logits = logits_t.data;
logits.truncate(vocab);
```

**With this:**
```rust
// 4. LM head: stream in chunks to avoid loading 2GB at once.
// lm_head.weight is [vocab, hidden]. Read 1024 rows at a time.
let lm_head_name = "lm_head.weight";
let mut logits = vec![0.0f32; vocab_size];
let chunk_size = 1024;
let hidden_t = Tensor::from_vec(hidden.clone(), vec![1, h]);
for chunk_start in (0..vocab_size).step_by(chunk_size) {
    let chunk_end = (chunk_start + chunk_size).min(vocab_size);
    let n_rows = chunk_end - chunk_start;
    let chunk = self.shards.read_tensor_slice_f32(
        lm_head_name,
        chunk_start * h,
        n_rows * h,
    )?;
    let chunk_t = Tensor::from_vec(chunk, vec![n_rows, h]);
    let logits_chunk = hidden_t.matmul(&chunk_t.transpose());
    logits[chunk_start..chunk_end].copy_from_slice(&logits_chunk.data);
}
```

**Compile and test:**
```bash
cargo build --release --no-default-features --bin test_streaming_forward 2>&1 | grep "^error" | head -10
timeout 300 ./target/release/test_streaming_forward 2>&1 | tail -30
```

**Expected:** Compiles. Runs all 32 layers. Produces logits. Top-5 prints. Might be garbage ("2" or similar) because DeltaNet may still have issues.

**Report:** The top-5 predictions and total time.

---

### STEP 3: Verify and fix DeltaNet conv1d layout (15 min)

The conv1d weight layout is the most likely source of garbage output. The safetensors file stores conv1d.weight as a flat array. We need to figure out if it's `[conv_k, conv_dim]` or `[conv_dim, conv_k]`.

**3a. Check the actual shape in safetensors:**
```bash
cd /home/xander/Documents/portfolio/LeafcutterLLM/rust
# Read the safetensors header to find conv1d shape
python3 -c "
import json, sys
with open('/home/xander/Downloads/models/ornith safetensor/model-00001-of-00004.safetensors', 'rb') as f:
    n = int.from_bytes(f.read(8), 'little')
    header = json.loads(f.read(n))
    key = 'model.language_model.layers.0.linear_attn.conv1d.weight'
    if key in header:
        print(json.dumps(header[key], indent=2))
    else:
        print('not found in shard 0')
        # Try other shards
"
```

If the shape is `[4, 8192]` (i.e., `[conv_k, conv_dim]`), our current code is correct.
If the shape is `[8192, 4]` (i.e., `[conv_dim, conv_k]`), we need to change line 242 from:
```rust
out[c] = conv_w[3 * conv_dim + c] * qkv_proj.data[c];
```
to:
```rust
out[c] = conv_w[c * conv_k + 3] * qkv_proj.data[c];
```

**3b. Also check in_proj_a, in_proj_b, in_proj_z shapes:**
```bash
python3 -c "
import json
with open('/home/xander/Downloads/models/ornith safetensor/model-00001-of-00004.safetensors', 'rb') as f:
    n = int.from_bytes(f.read(8), 'little')
    header = json.loads(f.read(n))
    for key in sorted(header.keys()):
        if 'layers.0.linear_attn' in key:
            print(f'{key}: dtype={header[key][\"dtype\"]} shape={header[key][\"shape\"]}')
"
```

This will tell us:
- `in_proj_a.weight` shape — should be `[32, 4096]` (n_v, hidden)
- `in_proj_b.weight` shape — should be `[32, 4096]` (n_v, hidden)
- `in_proj_z.weight` shape — should be `[4096, 4096]` (hidden, hidden) or `[4096, 4096]` (v_total, hidden)?
- `in_proj_qkv.weight` shape — should be `[8192, 4096]` (conv_dim, hidden)
- `A_log` shape — should be `[32]`
- `dt_bias` shape — should be `[32]`
- `norm.weight` shape — should be `[128]`
- `out_proj.weight` shape — should be `[4096, 4096]`

If any of these differ from what our code assumes, the Tensor::from_vec shapes in deltanet_forward need to be adjusted.

**Report:** The actual shapes from the safetensors header.

---

### STEP 4: Fix attention_forward q_proj shape (10 min)

**File:** `rust/src/streaming_ornith.rs`, in `attention_forward`

The current code on line 401 uses `vec![2 * h, h]` for q_proj:
```rust
let q_all = hidden_t.matmul(&Tensor::from_vec(q_w.clone(), vec![2 * h, h]).transpose());
```

This assumes q_proj outputs 2*hidden (Q + gate). But it might just be [h, h]. Check:

```bash
python3 -c "
import json
with open('/home/xander/Downloads/models/ornith safetensor/model-00001-of-00004.safetensors', 'rb') as f:
    n = int.from_bytes(f.read(8), 'little')
    header = json.loads(f.read(n))
    for key in sorted(header.keys()):
        if 'layers.0.self_attn' in key:
            print(f'{key}: dtype={header[key][\"dtype\"]} shape={header[key][\"shape\"]}')
"
```

If q_proj is `[h, h]` = `[4096, 4096]`, change line 401 to:
```rust
let q = hidden_t.matmul(&Tensor::from_vec(q_w.clone(), vec![h, h]).transpose());
```
And remove the q_all/q_data split (lines 401-403).

If q_proj is `[2*h, h]`, the current code is correct — it splits Q (first h) and a gate (second h).

Also check k_proj and v_proj shapes. They should be `[n_kv * head_dim, h]` = `[1024, 4096]`.

**Report:** The actual shapes of q_proj, k_proj, v_proj, o_proj.

---

### STEP 5: Validate correctness (5 min)

After fixing shapes, run:
```bash
cargo build --release --no-default-features --bin test_streaming_forward 2>&1 | grep "^error" | head -10
timeout 300 ./target/release/test_streaming_forward 2>&1 | tail -30
```

**Success:** Top-1 is "Paris" or at least a real English word.
**Partial success:** Top-5 contains some real words but not "Paris".
**Failure:** Top-5 is garbage (numbers, single chars, random tokens).

If failure, check:
1. Conv1d layout (Step 3)
2. Weight shapes (Step 3, 4)
3. QKV split order — might be KQV or VKQ instead of QKV
4. A_log convention — might already be A (negative), not log(A). If A_log values are already negative, change line 295 from `let a = -a_log_val.exp()` to `let a = a_log_val`
5. Decay might not apply at pos=0 — try setting decay=1.0 and beta=1.0 (line 322-323) to see if output changes

**Report:** Top-5 predictions, total time, and whether anything changed.

---

### STEP 6: Wire as --engine native in the CLI (20 min)

**File:** `rust/src/bin/leafcutter.rs` (or wherever cmd_run is)

Find where `--engine` is matched. Add a branch for `native`:

```rust
"native" => {
    use leafcutter::streaming_ornith::StreamingOrnith;
    use std::path::Path;
    let mut model = StreamingOrnith::open(Path::new(&model_path))
        .map_err(|e| format!("open model: {e}"))?;

    // Encode prompt
    let mut ids = model.tok.encode(&prompt, 1024);
    println!("Prompt tokens: {ids:?}");

    // Generate
    for i in 0..max_tokens {
        let last = *ids.last().unwrap() as i32;
        let t0 = std::time::Instant::now();
        let logits = model.forward_one_token(last, ids.len() - 1)
            .map_err(|e| format!("forward: {e}"))?;
        let next = StreamingOrnith::argmax(&logits);
        let elapsed = t0.elapsed();

        ids.push(next as i32);
        let text = model.tok.decode(&[next as i32]);
        print!("{text}");
        std::io::Write::flush(&mut std::io::stdout()).ok()?;
        eprintln!("\n  [token {} in {:.1}s]", i + 1, elapsed.as_secs_f64());

        // Stop on EOS (check a few common EOS tokens)
        if next == 151643 || next == 151645 { break; }  // <|im_end|> etc
    }
    println!();
}
```

You need to find the actual location where engines are dispatched. Run:
```bash
grep -n "engine" src/bin/leafcutter.rs | head -20
```

**Test:**
```bash
cargo build --release --no-default-features --bin leafcutter
./target/release/leafcutter run "/home/xander/Downloads/models/ornith safetensor" --engine native --prompt "The capital of France is" --max-tokens 5
```

**Report:** Does it work from the CLI? What tokens does it produce?

---

### STEP 7: Add a generate() method to StreamingOrnith (10 min)

**File:** `rust/src/streaming_ornith.rs`

Add this method to the `impl StreamingOrnith` block (before the closing `}`):

```rust
/// Generate text autoregressively.
/// NOTE: This does NOT carry KV state or DeltaNet state between tokens.
/// Each token is processed independently. Output will not be coherent
/// for multi-token generation until state caching is added.
pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, String> {
    let mut ids = self.tok.encode(prompt, 1024);
    let prompt_len = ids.len();
    eprintln!("[generate] prompt: {prompt}");
    eprintln!("[generate] tokens: {ids:?}");

    for i in 0..max_tokens {
        let last = *ids.last().unwrap() as i32;
        let t0 = std::time::Instant::now();
        let logits = self.forward_one_token(last, ids.len() - 1)?;
        let next = Self::argmax(&logits);
        let elapsed = t0.elapsed();

        ids.push(next as i32);
        let text = self.tok.decode(&[next as i32]);
        print!("{text}");
        std::io::Write::flush(&mut std::io::stdout()).ok()?;
        eprintln!("\n  [tok {}/{} in {:.1}s, id={next}]", i + 1, max_tokens, elapsed.as_secs_f64());

        // Common Ornith EOS tokens
        if next == 151643 || next == 151645 || next == 2 { break; }
    }
    println!();
    Ok(self.tok.decode(&ids[prompt_len..]))
}
```

**Test:**
```bash
cargo build --release --no-default-features --bin test_streaming_forward
# Update test_streaming_forward.rs to call generate() instead of forward_one_token
# Then run:
timeout 600 ./target/release/test_streaming_forward
```

**Expected:** Produces 5-10 tokens. Each token takes ~4-5s. Total ~50s for 10 tokens.
Output won't be coherent (no state between tokens) but should not crash.

**Report:** What tokens come out? How long per token?

---

### STEP 8: Measure and compare (5 min)

```bash
# Time the Python backend (reference)
time ./target/release/leafcutter run "/home/xander/Downloads/models/ornith safetensor" \
    --engine safetensor --prompt "The capital of France is" --max-tokens 3

# Time the native Rust backend
time ./target/release/leafcutter run "/home/xander/Downloads/models/ornith safetensor" \
    --engine native --prompt "The capital of France is" --max-tokens 3

# Check peak RAM
/usr/bin/time -v ./target/release/leafcutter run "/home/xander/Downloads/models/ornith safetensor" \
    --engine native --prompt "The capital of France is" --max-tokens 1 2>&1 | grep "Maximum resident"
```

**Targets:**
| Metric | Python backend | Native backend (target) | AirLLM |
|--------|---------------|------------------------|--------|
| Time/token | ~12s | <12s | ~12s |
| Peak RAM | ~4GB | <500MB | ~2GB |
| Correctness | "Paris" | "Paris" | "Paris" |

**Report:** All three numbers for both backends.

---

## PART 3: KNOWN ISSUES AND DIAGNOSTICS

### If DeltaNet output is garbage

**Issue 1: Conv1d weight layout**
Safetensors stores conv1d weights as `[out_channels, in_channels, kernel_size]` or a flattened version. For a 1D conv with `in_channels=1, out_channels=8192, kernel_size=4`, the shape could be:
- `[4, 8192]` (PyTorch Conv1d: out_channels, in_channels, kernel_size → transpose for our use)
- `[8192, 4]` (alternative layout)
- `[8192, 1, 4]` (3D, flattened to `[8192 * 4]`)

To debug, print the conv weight values:
```rust
eprintln!("[debug] conv_w first 20: {:?}", &conv_w[..20]);
eprintln!("[debug] conv_w len: {}", conv_w.len());
```
Then try both layouts and see which produces non-zero, reasonable output.

**Issue 2: A_log convention**
Some models store `A_log` as the log of the diagonal. Others store `A` directly (already negative). Check:
```rust
eprintln!("[debug] A_log first 10: {:?}", &a_log[..10]);
```
If values are like `0.0, -1.0, -2.0, ...` → they're already A (negative), use `let a = a_log_val` instead of `let a = -a_log_val.exp()`.
If values are like `1.0, 2.0, 3.0, ...` → they're log(|A|), use `let a = -a_log_val.exp()`.

**Issue 3: QKV split order**
We assume Q comes first, then K, then V. But some models use KQV or VKQ. If the output is garbage, try swapping the slice ranges:
- Q: `0..q_total` (current)
- K: `q_total..q_total + k_total` (current)
- V: `q_total + k_total..` (current)

Try swapping Q and K:
- Q: `q_total..q_total + k_total`
- K: `0..q_total`

**Issue 4: in_proj_z dimensions**
`in_proj_z.weight` might be `[4096, 4096]` (h, h) producing z of shape [4096], not [v_total=4096]. In our case v_total = 32 * 128 = 4096 = h, so it matches. But if the shape is different, adjust the Tensor::from_vec shape.

### If attention_forward is wrong

**Issue 1: q_proj shape**
If q_proj is `[4096, 4096]` (not `[8192, 4096]`), there's no gate split. Change the code to just use q_proj directly.

**Issue 2: RoPE**
We're not applying RoPE. For pos=0 (single token), RoPE doesn't matter (rotation by 0). For multi-token, we need RoPE. But for validating correctness on the first token, no RoPE is fine.

**Issue 3: Attention gate**
If q_proj outputs 2*h, the second half is an output gate (like GLM's attention). The gate should be applied as: `output = output * silu(gate)`. Add this after the attention loop:
```rust
if q_all.data.len() == 2 * h {
    let gate_data = &q_all.data[h..];
    for i in 0..h {
        let g = gate_data[i];
        attn_out[i] *= g / (1.0 + (-g).exp()); // silu(gate)
    }
}
```

---

## PART 4: ORNITH ARCHITECTURE REFERENCE

### Config (from config.json text_config)
```
hidden_size:              4096
num_hidden_layers:        32  (24 linear_attention + 8 full_attention)
num_attention_heads:     16  (for full_attention layers)
num_key_value_heads:      4   (GQA ratio 4:1)
head_dim:                 256
intermediate_size:        12288
vocab_size:               248320
rms_norm_eps:             1e-6
rope_theta:               500000
partial_rotary_factor:    0.5

linear_num_key_heads:     16
linear_num_value_heads:   32
linear_key_head_dim:      128
linear_value_head_dim:    128
linear_conv_kernel_dim:   4
```

### Linear attention (DeltaNet) layer — 14 weight tensors
```
input_layernorm.weight              [4096]
linear_attn.in_proj_qkv.weight      [8192, 4096]  → Q(2048)+K(2048)+V(4096)
linear_attn.in_proj_a.weight         [32, 4096]    → decay alpha
linear_attn.in_proj_b.weight         [32, 4096]    → beta gate
linear_attn.in_proj_z.weight         [4096, 4096]  → z-gate (silu)
linear_attn.conv1d.weight            [4, 8192] or [8192, 4] → CHECK
linear_attn.A_log                   [32]          → log of decay diagonal
linear_attn.dt_bias                  [32]          → decay bias
linear_attn.norm.weight              [128]         → per-head RMSNorm
linear_attn.out_proj.weight          [4096, 4096]  → output projection
post_attention_layernorm.weight      [4096]
mlp.gate_proj.weight                 [12288, 4096]
mlp.up_proj.weight                   [12288, 4096]
mlp.down_proj.weight                 [4096, 12288]
```

### Full attention layer — 9 weight tensors
```
input_layernorm.weight              [4096]
self_attn.q_proj.weight             [4096, 4096] or [8192, 4096] → CHECK
self_attn.k_proj.weight             [1024, 4096]  (4 heads × 256 dim)
self_attn.v_proj.weight             [1024, 4096]
self_attn.o_proj.weight             [4096, 4096]
post_attention_layernorm.weight     [4096]
mlp.gate_proj.weight                 [12288, 4096]
mlp.up_proj.weight                   [12288, 4096]
mlp.down_proj.weight                 [4096, 12288]
```

### DeltaNet forward (single token, pos=0)
```
1. normed = rmsnorm(hidden, input_layernorm)
2. qkv = normed @ in_proj_qkv^T  → [8192]
3. conv: for pos=0, out[c] = conv_w[3*8192 + c] * qkv[c], then SiLU
4. Split: Q[0:2048], K[2048:4096], V[4096:8192]
5. L2-normalize Q and K per-head (128 dims, 16 heads)
6. Q *= 1/sqrt(128)
7. alpha = hidden @ in_proj_a^T  → [32]
8. A = -exp(A_log)  → [32]    (or A = A_log if already negative)
9. decay = exp(softplus(alpha + dt_bias) * A)  → [32]
10. beta = sigmoid(hidden @ in_proj_b^T)  → [32]
11. For each (qk_head=0..15, v_head=0..31):
    qk_dot = dot(Q[qk_head], K[qk_head])
    output[v_head] = beta[v_head] * qk_dot * V[v_head]
12. Per-head RMSNorm (128 dims, 32 heads) with norm.weight
13. z = hidden @ in_proj_z^T → [4096]
14. output *= silu(z)
15. result = output @ out_proj^T → [4096]
```

### Full attention forward (single token, pos=0)
```
1. normed = rmsnorm(hidden, input_layernorm)
2. Q = normed @ q_proj^T → [4096] (16 heads × 256)
3. K = normed @ k_proj^T → [1024] (4 heads × 256)
4. V = normed @ v_proj^T → [1024]
5. [optional] gate split: if Q is 2*h, Q=Q[:h], gate=Q[h:]
6. For pos=0: attention to itself, softmax=1.0, output=V (broadcast via GQA)
7. [optional] apply gate: output *= silu(gate)
8. result = output @ o_proj^T → [4096]
```

### MLP forward (SwiGLU)
```
1. gate = hidden @ gate_proj^T → [12288]
2. up = hidden @ up_proj^T → [12288]
3. inter = silu(gate) * up
4. result = inter @ down_proj^T → [4096]
```

---

## PART 5: FILE INVENTORY

| File | Status | What to do |
|------|--------|-----------|
| `src/safetensors_loader.rs` | ✅ Done | Has slice reads. Leave it. |
| `src/bpe_tokenizer.rs` | ✅ Done | Leave it. |
| `src/ornith_config.rs` | ✅ Done | Leave it. |
| `src/model/tensor.rs` | ✅ Done | `Tensor::matmul` uses BLAS. Leave it. |
| `src/streaming_ornith.rs` | 🔧 Fix + test | Steps 1-5 |
| `src/bin/test_streaming_forward.rs` | ✅ Exists | May need to update for generate() |
| `src/bin/leafcutter.rs` | 🔧 Add native | Step 6 |

---

## PART 6: WHAT TO DO RIGHT NOW

1. Open `rust/src/streaming_ornith.rs`
2. Do Step 1 (clean up dead code) — 5 min
3. Do Step 2 (chunked lm_head) — 10 min
4. Do Step 3 (check safetensors shapes with python) — 15 min
5. Do Step 4 (fix attention shapes) — 10 min
6. Compile + test (Step 5) — 5 min
7. Report results

Then I'll diagnose any issues and give you Steps 6-8.

If you get stuck on anything, come back with:
- What you tried
- What the error/output was
- What the actual safetensors shapes are

That's all I need to fix any issue.
