# LeafcutterLLM: GGUF Integration Strategy

> Based on comprehensive analysis of llama.cpp (ggml-org/llama.cpp @ /home/xander/Documents/portfolio/leafcutter_max/llama.cpp/)

---

## 1. Executive Summary

**Goal:** Interactive chat with GGUF models via LeafcutterLLM's Rust streaming engine — including small quantized models (Q4_K, Q6_K) so users can run capable models on low storage. The primary use case is *storage-constrained* systems: users who don't need to save storage can just run safetensors; the GGUF path exists to serve quantized models.

**Current state (Aug 2026):** Phases 1-5 and 5B **complete**. The engine loads GGUF weights (Q8_0, Q4_K_M, Q6_K all verified) and **generates coherent, on-topic text** from Ornith-9B — the model's real reasoning block — in the interactive `leafcutter run ornith` chat REPL, fully native. Performance: ~0.78 s/tok steady-state compute-bound; lm_head uses a **Q6_K block cache** (~87.8 ms/tok, −3 GB RAM vs the earlier f32 cache). Peak chat RAM **~8.1 GB**, 1.2–1.65 tok/s. Test suite green: **161 pass / 0 fail / 3 ignored**.

**Project status (2026-08-01):** **PROJECT WRAP-UP.** The original tripolar mission — "lighter than airllm, faster + smarter than colibri" — is delivered for the flagship model: Ornith 1.0 9B runs natively, coherently, and interactively. Remaining work is future/optional (see Roadmap in README): zero-copy `load_layer`, SIMD expansion, distributed inference, GPU backends.

```
Loaded GGUF ─→ Bridge ─→ Engine ─→ Tokenizer ─→ Chat template ─→ Coherent output
    ✅            ✅        ✅          ✅           ✅             ✅ (slow)
```

Verified outputs (same 73-token prompt, temp 0):
| Quant | Output |
|-------|--------|
| Q4_K_M (5.3 GB) | `The user said "Hello - this is a simple greeting...` |
| Q6_K (6.9 GB) | `The user said "Hello - this is a simple...` (token-identical) |
| Ollama Q4_K_M ref | `The user has simply said "Hello"...` |

Tokens 1-2 match Ollama; token 3 diverges (ours `said`, ref `has`) — a small residual numeric difference, not a structural bug.

Remaining before a fast working `cargo run`:
1. ✅ Chat template — `apply_chat_template_from_gguf()` wired; `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n` matches Ollama's `ornith` renderer
2. ✅ Tokenizer — `GgufBpeTokenizer::from_gguf()` (vocab=248320) verified; HF fallback (vocab 248070) correctly rejected
3. ❌ **Performance** — weights re-read + re-dequantized from disk per token; `MADV_DONTNEED` evicts the whole mmap after every layer. See §10.5 for measured numbers and the fix plan
4. ❌ Streaming `chat`/`run` path lacks prompt prefill — only the last prompt token is processed
5. ❓ Residual token-3 logit divergence vs llama.cpp — needs layer-by-layer diff to pin down

**Reference:** llama.cpp provides the complete reference implementation (171 C++ model architectures, GGUF v3 format, weight quantization, CPU/GPU backends).

---

## 2. GGUF File Format — Complete Specification

### 2.1 File Structure (Binary Layout)

```
Offset  | Content
--------|--------
0       | Magic: "GGUF" (4 bytes, uint8[4])
4       | Version: uint32_t (currently 3)
8       | n_tensors: int64_t
16      | n_kv: int64_t
24      | KV Pairs (variable length, see §2.2)
...     | Tensor Info (n_tensors entries, see §2.3)
...     | ALIGNMENT padding (default 32 bytes)
...     | Tensor Data Blob (raw bytes, one contiguous block)
```

### 2.2 KV Pair Format

Each KV pair:
```
Key:   uint64_t length, char data[length] (no null terminator)
Type:  int32_t (gguf_type enum)
Value: depends on type:
  - UINT8/INT8:     int8_t (1 byte)
  - UINT16/INT16:   int16_t (2 bytes)
  - UINT32/INT32:   int32_t (4 bytes)
  - FLOAT32:        float (4 bytes)
  - UINT64/INT64:   int64_t (8 bytes)
  - FLOAT64:        double (8 bytes)
  - BOOL:           int8_t (1 byte)
  - STRING:         uint64_t length, char data[length]
  - ARRAY:          int32_t element_type, uint64_t n_elements, element_data...
```

### 2.3 Tensor Info Format

Each tensor:
```
Name:   uint64_t length, char name[length] (e.g. "blk.0.attn_norm.weight")
Dims:   uint32_t n_dims
Shape:  int64_t ne[0..n_dims-1]
Type:   int32_t (ggml_type enum, e.g. GGML_TYPE_F32=0, GGML_TYPE_BF16=30)
Offset: uint64_t (byte offset from start of tensor data blob)
```

### 2.4 Key GGML Types

| Enum | Value | Name | Block Size | Bytes/Block |
|------|-------|------|-----------|-------------|
| GGML_TYPE_F32 | 0 | float32 | 1 | 4 |
| GGML_TYPE_F16 | 1 | float16 | 1 | 2 |
| GGML_TYPE_BF16 | 30 | bfloat16 | 1 | 2 |
| GGML_TYPE_Q8_0 | 8 | q8_0 quantized | 32 | 34 |

### 2.5 Alignment

- Default alignment: 32 bytes (GGUF_DEFAULT_ALIGNMENT = 32)
- Alignment key: `"general.alignment"` (uint32) in KV pairs overrides default
- Each tensor's offset in the data blob is a multiple of alignment
- Each tensor's data is padded to alignment boundary in the blob
- The gap between end of tensor info and start of data blob is padded to alignment

### 2.6 Key Implementation in llama.cpp

The actual GGUF parsing is in `ggml/src/gguf.cpp`:
- `gguf_init_from_reader()` — main parser (lines 451-898)
- `gguf_context` struct holds: version, kv[], info[], alignment, offset, data
- `gguf_tensor_info` holds: ggml_tensor (name, shape, type), offset
- Three init paths: file, buffer, callback (all converge on `gguf_init_from_reader`)
- Tensor data is optionally loaded: `params.no_alloc` controls memory mapping vs loading

---

## 3. Tensor Name Mapping (Safetensors → GGUF)

### 3.1 Qwen3.5/Ornith-1.0-9B Mapping

Every GGUF tensor name follows the pattern: `blk.{layer_id}.{tensor_kind}.{suffix}`

| Safetensors Name | GGUF Name | Shape (Ornith 9B) |
|-----------------|-----------|-------------------|
| `model.embed_tokens.weight` | `token_embd.weight` | [4096, 248320] |
| `model.norm.weight` | `output_norm.weight` | [4096] |
| `lm_head.weight` | `output.weight` | [4096, 248320] |
| `model.layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` | [4096] |
| `model.layers.{i}.post_attention_layernorm.weight` | `blk.{i}.attn_post_norm.weight` | [4096] |
| `model.layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight` | [4096, 4096] (full attn only) |
| `model.layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight` | [4096, 1024] (full attn only) |
| `model.layers.{i}.self_attn.v_proj.weight` | `blk.{i}.attn_v.weight` | [4096, 1024] (full attn only) |
| `model.layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_out.weight` | [4096, 4096] (full attn only) |
| `model.layers.{i}.self_attn.q_norm.weight` | `blk.{i}.attn_q_norm.weight` | [256] (full attn only) |
| `model.layers.{i}.self_attn.k_norm.weight` | `blk.{i}.attn_k_norm.weight` | [256] (full attn only) |
| `model.layers.{i}.self_attn.in_proj_qkv.weight` | `blk.{i}.attn_qkv.weight` | [4096, 8192] (linear attn only) |
| `model.layers.{i}.self_attn.in_proj_z.weight` | `blk.{i}.attn_gate.weight` | [4096, 4096] (linear attn only) |
| `model.layers.{i}.self_attn.in_proj_b.weight` | `blk.{i}.ssm_beta.weight` | [4096, 32] (linear attn only) |
| `model.layers.{i}.self_attn.in_proj_a.weight` | `blk.{i}.ssm_alpha.weight` | [4096, 32] (linear attn only) |
| `model.layers.{i}.self_attn.A_log` | `blk.{i}.ssm_a` | [32] (linear attn only) |
| `model.layers.{i}.self_attn.dt_proj.bias` | `blk.{i}.ssm_dt.bias` | [32] (linear attn only) |
| `model.layers.{i}.self_attn.conv1d.weight` | `blk.{i}.ssm_conv1d.weight` | [4, 8192] (see NOTE) |
| `model.layers.{i}.self_attn.o_norm.weight` | `blk.{i}.ssm_norm.weight` | [128] (linear attn only) |
| `model.layers.{i}.self_attn.out_proj.weight` | `blk.{i}.ssm_out.weight` | [4096, 4096] (linear attn only) |
| `model.layers.{i}.mlp.gate_proj.weight` | `blk.{i}.ffn_gate.weight` | [4096, 12288] |
| `model.layers.{i}.mlp.down_proj.weight` | `blk.{i}.ffn_down.weight` | [12288, 4096] |
| `model.layers.{i}.mlp.up_proj.weight` | `blk.{i}.ffn_up.weight` | [4096, 12288] |

**NOTE — conv1d dimension ordering (UPDATED 2026-07-31):**
- Safetensors: `[conv_dim (8192), kernel_size (4)]` — each row is a filter tap, columns are channels
- GGUF: `[kernel_size (4), conv_dim (8192)]` — ne[0]=kernel_size, ne[1]=channels; data is **channel-major** (flat `c*conv_k + k`)
- The `ggml_ssm_conv` op expects ne[0]=kernel_size, ne[1]=channels
- **Empirically verified**: no transpose is needed when loading from GGUF — the earlier `needs_transpose` heuristic was a bug and was removed. Tap order: `out[t] = sum_j w[j]*x[t-(d_conv-1)+j]`, w[0]=oldest (ops.cpp:9557-9616).

### 3.2 Dimension Convention

GGML stores tensors in **column-major** (Fortran) order:
- `ne[0]` = innermost dimension (contiguous in memory)
- `ne[n_dims-1]` = outermost dimension
- For a matrix `[M, N]`: ne[0]=M (rows), ne[1]=N (columns)
- Row stride: `nb[1] = ne[0] * type_size / blck_size`

Safetensors uses **row-major** order:
- Shape `[M, N]`: dimension 0 = rows, dimension 1 = columns

When loading from GGUF, the `ne[]` array IS the shape (no transpose needed). The tensor data is stored with `ne[0]` contiguous.

---

## 4. Qwen3.5/Ornith Architecture from C++ Reference

### 4.1 Architecture Dispatch

`qwen35.cpp` defines two graph contexts:
- `graph` — main forward pass (all layers, hybrid linear + full attention)
- `graph_mtp` — MTP speculative draft head (full attention only)

Hybrid detection: `hparams.is_recr(il)` checks if layer `il` uses linear attention.
- By default: every layer with `(i+1) % 4 != 0` is recurrent (layers 0,1,2,4,5,6,... are linear; 3,7,11,... are full attention)
- Overridable via `LLM_KV_ATTENTION_RECURRENT_LAYERS` array in GGUF metadata

### 4.2 Full Attention Layer (build_layer_attn)

Called for layers where `!is_recr(il)` (layers 3, 7, 11, ... in Ornith 9B):

```
1. Q projection:     Qcur_full = wq @ cur           -> [n_embd_head*2 * n_head, n_tokens]
2. Split Q + gate:   Qcur = Qcur_full[:, 0:n_embd_head]
                      gate = Qcur_full[:, n_embd_head:2*n_embd_head]
3. Q norm:           Qcur = rms_norm(Qcur, attn_q_norm)
4. K projection:     Kcur = wk @ cur
5. K norm:           Kcur = rms_norm(Kcur, attn_k_norm)
6. V projection:     Vcur = wv @ cur
7. RoPE:            Qcur, Kcur = rope_multi(Qcur, Kcur, positions, sections)
8. Attention:        attn = softmax(Q @ K^T / sqrt(d)) @ V
9. Gate:             attn = attn * sigmoid(gate)
10. Output proj:     cur = wo @ attn
```

Key details:
- Q projection is fused with gate: the Wq weight outputs `[n_embd_head*2, n_head]` = 2 * 256 * 16 = 8192 dims
- RoPE uses `ggml_rope_multi` with sections array (e.g., [64, 0, 0, 0] for partial_rotary_factor=0.25)
- K normalization is grouped KV (n_head_kv=4, repeats to n_head=16 for attention)
- Gate is sigmoid (NOT silu) — this was Bug #4 in our Rust implementation

### 4.3 Linear Attention Layer (build_layer_attn_linear)

Called for layers where `is_recr(il)` (layers 0,1,2,4,5,6,... in Ornith 9B):

```
1. QKV projection:   qkv_mixed = wqkv @ cur       -> [key_dim*2 + value_dim] = 8192
2. Z gate projection: z = wqkv_gate @ cur           -> [value_dim] = 4096
3. Beta projection:   beta = ssm_beta @ cur         -> [n_v_heads] = 32
4. Alpha projection:  alpha = ssm_alpha @ cur       -> [n_v_heads] = 32
5. Alpha bias:        alpha_biased = alpha + ssm_dt
6. Alpha softplus:    alpha_softplus = softplus(alpha_biased)
7. Gate:              gate = alpha_softplus * ssm_a   -> [n_v_heads]
   NOTE: ssm_a (= A_log) is multiplied directly — NO negation, NO exp
8. Conv state:        conv_input = concat(conv_states, qkv_mixed, dim=0)
9. Conv1d:            conv_output = ssm_conv(conv_input, conv_kernel)
10. SiLU:             conv_output = silu(conv_output)
11. Split Q, K, V:    q, k, v = split(conv_output, [d_k*n_k, d_k*n_k, d_v*n_v])
12. L2 norm:          q = l2_norm(q), k = l2_norm(k)
13. Head repeat:      if n_k_heads != n_v_heads, repeat q,k to match v heads
14. Delta Net:        output, new_state = delta_net(q, k, v, gate, beta, state)
15. Gate-norm:        attn_out = build_norm_gated(output, ssm_norm, z)
                      = rms_norm(output, ssm_norm) * silu(z)
16. Output proj:      cur = ssm_out @ attn_out
```

LAYER DIMENSIONS (Ornith 9B):
- `d_inner` = 4096 (ssm_d_inner = hidden_size)
- `head_k_dim` = 128 (ssm_d_state)
- `head_v_dim` = d_inner / num_v_heads = 4096 / 32 = 128
- `num_k_heads` = 16 (ssm_n_group)
- `num_v_heads` = 32 (ssm_dt_rank)
- `key_dim` = 128 * 16 = 2048
- `value_dim` = 128 * 32 = 4096
- `conv_dim` = key_dim * 2 + value_dim = 2048*2 + 4096 = 8192
- `conv_kernel` = 4 (ssm_d_conv)

### 4.4 The A_log Convention (Warning: GGUF vs Safetensor difference)

**CRITICAL DISTINCTION:** The conversion script (`conversion/qwen.py:298-299`) transforms A_log **at conversion time**:

```python
if name.endswith(".A_log"):
    data_torch = -torch.exp(data_torch)  # transforms raw A_log into -exp(A_log)
```

So in the GGUF file, `blk.{i}.ssm_a` stores `-exp(raw_A_log)` — values that are ALWAYS negative (confirmed range for Ornith: -0.14 to -0.009). The conversion time transformation ensures the stored values are always negative, so the C++ runtime can multiply them directly.

**The C++ code** at `qwen35.cpp:376`:
```cpp
ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  
```
This works because `ssm_a` in GGUF is already `-exp(A_log)` (always negative).

Then at `delta-net-base.cpp:339-340`:
```cpp
g = ggml_exp(ctx0, g);   // g = exp(softplus(dt) * (-exp(A_log)))
s = ggml_mul(ctx0, s, g);  // s *= exp(softplus(dt) * (-exp(A_log)))
```

**The Rust code** (reading directly from safetensors, not GGUF) does:
```rust
let a = -a_log_val.exp();        // a = -exp(A_log)
decay[head] = (dt * a).exp();    // = exp(dt * (-exp(A_log)))
```

**Both produce identical results.** ✓

| Context | Stored Value | Runtime Formula | Result |
|---------|-------------|-----------------|--------|
| C++ + GGUF | `ssm_a = -exp(A_log)` | `exp(dt * ssm_a)` | `exp(dt * (-exp(A_log)))` |
| Rust + safetensors | `A_log` raw | `exp(dt * (-exp(A_log)))` | `exp(dt * (-exp(A_log)))` |

**IMPLICATION for GGUF integration:** When switching to GGUF-based loading, the Rust code MUST change to:
```rust
let a = a_log_val;  // For GGUF: a_log_val is already -exp(raw_A_log)
```
(Because the GGUF stores the pre-transformed value.)

### 4.5 FFN Layer

```cpp
cur = build_ffn(cur,
    ffn_up, NULL, ffn_up_s,        // up_proj
    ffn_gate, NULL, ffn_gate_s,    // gate_proj
    ffn_down, NULL, ffn_down_s,    // down_proj
    NULL,
    LLM_FFN_SILU, LLM_FFN_PAR, il);
```

Standard SwiGLU FFN: `output = down(silu(gate(x)) * up(x))`

---

## 5. DeltaNet Computation from C++ (delta-net-base.cpp)

### 5.1 Autoregressive Path (1 token at a time)

`build_delta_net_autoregressive()` (lines 289-371):

```
Given: q[K,H_k,1,S], k[K,H_k,1,S], v[V,H_v,1,S], g[1,H_v,1,S], b[1,H_v,1,S], s[V,V,H_v,S]

1. Scale q: q = q / sqrt(K)
2. Permute: all dims to [*, 1, H, S]  (expand n_tokens dim)
3. Reshape g: [1, 1, H_v, S], b: [1, 1, H_v, S]
4. Decay state: s = s * exp(g)        -- g = softplus(dt) * A_log
5. Predict:       sk = sum_rows(s * k, dim=0)   -- [1, V, H_v, S]
6. v_pred:        d = (v - sk^T) * b            -- [V, 1, H_v, S]
7. Update:        s = s + k * d^T               -- [V, V, H_v, S]
8. Output:        o = sum_rows(s * q, dim=0)    -- [V, 1, H_v, S]
9. Reshape:       o -> [V, H_v, 1, S]
```

This is the classical DeltaNet delta rule:
- `s[t] = s[t-1] * exp(g[t]) + k[t] * (v[t] - s[t-1] * exp(g[t]) * k[t])^T * b[t]`

The C++ implements this efficiently with matrix operations.

### 5.2 Chunking Path (multiple tokens at once)

`build_delta_net_chunking()` (lines 16-287):

For n_tokens > 1, performs chunked parallel delta net computation:
1. Pads to chunk_size boundary (CS=64 for GDA, CS=16 for KDA)
2. Computes cumulative decay sums: `g_cs = cumsum(g, dim=1)`
3. Builds decay mask: `decay_mask = tril(exp(g_cs_j - g_cs_i))`
4. Computes attention matrix via linear solve
5. Computes chunk output and state update
6. State is accumulated across chunks

### 5.3 Conv1d State Management

`build_conv_state()` (lines 449-525):

The conv state is stored as `[kernel_size - 1, conv_dim]` per sequence.

On each token:
1. Retrieve conv_states from cache: shape `[kernel_size-1, conv_dim, n_seqs]`
2. Transpose qkv_mixed (current token): `[conv_dim, n_seq_tokens, n_seqs]`
3. Concatenate: `conv_input = concat(conv_states, qkv_mixed, dim=0)`
   -> shape `[kernel_size-1 + n_seq_tokens, conv_dim, n_seqs]`
4. Run `ggml_ssm_conv(conv_input, conv_kernel)`
   - convolves along dim 0 with kernel `[kernel_size, conv_dim]`
   - output shape `[conv_dim, n_seq_tokens, n_seqs]`
5. Write back conv state: copy last `kernel_size-1` rows of conv_input back to cache

**Key geometry for n_seq_tokens=1:**
- conv_input = [conv_states (3 rows), qkv_mixed (1 row)] = [4, conv_dim, n_seqs]
- conv_kernel = [4, conv_dim]
- ssm_conv computes: for each output channel c, result[c] = sum_{t=0..3} kernel[t, c] * input[t, c]
- This is a standard 1D convolution with kernel_size=4, stride=1

**This matches our Rust conv implementation** — the "double shift" bug we fixed was correct. Our current conv logic (shift buffer, write current at tap 3, sum all 4 taps) is equivalent to the C++ conv.

---

## 6. Hyperparameters (GGUF KV Keys)

### 6.1 Qwen3.5 Specific Keys

| GGUF Key | C++ Access | Purpose | Ornith 9B Value |
|----------|-----------|---------|-----------------|
| `general.architecture` | `llm_arch_from_string()` | Arch enum | `"qwen35"` |
| `general.alignment` | `gguf_get_alignment()` | Data alignment | 32 (default) |
| `general.name` | `llama_model.name` | Model name | `"Ornith-1.0-9B"` |
| `llama.context_length` | `hparams.n_ctx` | Max context length | 32768 |
| `llama.embedding_length` | `hparams.n_embd` | Hidden dim | 4096 |
| `llama.block_count` | `hparams.n_layer` | Number of layers | 32 |
| `llama.feed_forward_length` | `hparams.n_ff` | FFN intermediate | 12288 |
| `llama.attention.head_count` | `hparams.n_head` | Num attention heads | 16 |
| `llama.attention.head_count_kv` | `hparams.n_head_kv` | Num KV heads | 4 |
| `llama.attention.layer_norm_rms_epsilon` | `hparams.f_norm_rms_eps` | RMS norm eps | 1e-6 |
| `llama.rope.dimension_count` | `hparams.n_rot` | Rotary dims | 256 |
| `llama.rope.freq_base` | `hparams.rope_freq_base` | RoPE theta | 10000000 |
| `llama.rope.scaling.type` | `hparams.rope_scaling_type_train` | RoPE scaling | `"default"` (none) |
| `llama.ssm.conv_kernel` | `hparams.ssm_d_conv` | Conv1d kernel | 4 |
| `llama.ssm.inner_size` | `hparams.ssm_d_inner` | DeltaNet inner | 4096 |
| `llama.ssm.state_size` | `hparams.ssm_d_state` | State dim (d_k) | 128 |
| `llama.ssm.time_step_rank` | `hparams.ssm_dt_rank` | Num v heads | 32 |
| `llama.ssm.group_count` | `hparams.ssm_n_group` | Num k heads | 16 |
| `llama.rope.dimension_sections` | `hparams.rope_sections` | RoPE sections | [64,0,0,0] |
| `llama.attention.recurrent_layers` | `hparams.is_recr_impl` | Per-layer recr flag | [bool; n_layer] |
| `llama.attention.full_attention_interval` | (alternative) | Every Nth full | 4 |
| `general.file_type` | `ml.get_key()` | Quant format | 1 (F32) or 2 (BF16) |

### 6.2 Determining Layer Type

```python
# Default: every (layer_index + 1) % 4 == 0 is FULL attention
full_attn_interval = 4
for i in range(n_layer):
    is_recurrent[i] = (i + 1) % full_attn_interval != 0

# Layer types for n_layer=32:
# L0: recr  L1: recr  L2: recr  L3: full (since (3+1)%4==0)
# L4: recr  L5: recr  L6: recr  L7: full
# ... (pattern repeats every 4 layers)
```

This means Ornith 9B has: 24 linear layers + 8 full attention layers.

---

## 7. Current LeafcutterLLM State & Integration Plan

### 7.1 What Already Exists

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| GGUF parser | `rust/src/model/gguf.rs` (836 lines) | Functional | Reads metadata, tensor info, data |
| GGUF loader | `rust/src/model/loader.rs` (1213 lines) | Functional | Weight loading + K-quant parsing (Q4_K/Q5_K/Q6_K/Q8_K) |
| GGUF weight bridge | `rust/src/gguf_provider.rs` (411 lines) | Functional | Name mapping, A_log inversion, conv1d direct load |
| Streaming engine | `rust/src/streaming_ornith.rs` (773 lines) | Working | Streams all 32 layers; V-head + norm fixes applied |
| Full native engine | `rust/src/inference/engine.rs` (1808 lines) | Working | `forward_native`, `generate`; prefetch; deltanet path |
| DeltaNet layer | `rust/src/inference/deltanet.rs` (554 lines) | Working | V-head interleave + conv1d fixes applied |
| K/V cache | `rust/src/cache/deltanet_state.rs` | Working | State + conv buffer |
| GGUF-native tokenizer | `rust/src/tokenizer/gguf_bpe.rs` | Working | Reads `tokenizer.ggml.tokens` from GGUF metadata |

### 7.2 What Has Changed

**GGUF Integration (Phases 1-2, COMPLETE):**

1. ✅ Created `gguf_provider.rs` — GGUF weight bridge with name mapping, A_log inversion, conv1d direct load
2. ✅ Added `WeightProvider` trait — abstracts over safetensor (Shards) and GGUF (GGUFWeightProvider)
3. ✅ Modified `StreamingOrnith` to use `Box<dyn WeightProvider>` — supports both at runtime
4. ✅ Added `StreamingOrnith::open_gguf()` — loads model from .gguf + tokenizer.json
5. ✅ Added `extract_ornith_config()` — reads all model hyperparams from GGUF metadata
6. ✅ Auto-detection in `main.rs` — `.gguf` files dispatched to GGUF engine

**Critical bug found & fixed during Phase 2:**
- `extract_ornith_config()` was falling back to `head_dim (256)` for `linear_key_head_dim` instead of reading `qwen35.ssm.state_size (128)` from GGUF metadata
- This caused wrong conv_dim (10240 vs 8192) → runtime panic in conv1d transpose
- **Fix**: added `qwen35.ssm.state_size`, `qwen35.ssm.conv_kernel`, `qwen35.ssm.group_count` as metadata sources (commit: [applied in gguf_provider.rs])

**Three correctness bugs fixed to reach coherent output (July 2026):**

1. **V-head pairing was blocked, must be interleaved.** llama.cpp pairs v_head `h_v` with q/k head `h_v % n_qk` (`ggml_repeat_4d`, llama-model.cpp:523-525). Both engines used `h_v = h_qk * r + v_idx`. Fixed in `streaming_ornith.rs` (step 8) and `inference/deltanet.rs`. Confirmed by the converter's `_LinearAttentionVReorderBase` (conversion/qwen.py:355-390): GGUF stores V heads in tiled order `[k0_v0, k1_v0, ..., k0_v1, ...]` (n_v=32, n_qk=16, r=2).
2. **Norm weights have `+1` pre-baked in GGUF.** Converter adds `data_torch + 1` to EVERY norm weight except `linear_attn.norm.weight` (conversion/qwen.py:304-305); llama.cpp `build_norm` multiplies directly (llama-graph.cpp:1451-1484). Our engine applied a second `+1` — removed from `rms_norm` and the full-attn q/k norm. Verified empirically: `attn_norm.weight` mean=1.033, `attn_q_norm` mean=1.341 (≈1+γ baked in).
3. **Conv1d layout + tap order.** GGUF stores `ssm_conv1d.weight` channel-major `[4, 8192]` (flat `c*conv_k + k`); the previous transpose heuristic was wrong and was removed in `gguf_provider.rs`. llama.cpp tap convention is `out[t] = sum_j w[j]*x[t-(d_conv-1)+j]` (w[0]=oldest, ops.cpp:9557-9616); the full engine's conv kernel now uses channel-major indexing + reversed taps.

**Quantization breakthrough (July 2026):** Q4_K, Q5_K, Q6_K, Q8_K dequant kernels all implement the GGUF block formats correctly (verified byte-by-byte against the llama.cpp spec). Q4_K_M and Q6_K produce **token-identical** output — proving the dequant paths are correct.

### 7.3 Implementation Phases

#### ✅ Phase 1: GGUF Weight Loading Bridge (COMPLETE)
- Created `src/gguf_provider.rs` — a standalone bridge module
- Provides `load_gguf_layer_weights()` and `load_gguf_non_layer_weights()`
- Name mapping: GGUF names (blk.{i}.ssm_alpha.weight) → streaming engine names (linear_attn.in_proj_a.weight)
- A_log handling: GGUF stores `ssm_a = -exp(A_log)`. Bridge recovers raw `A_log = ln(-ssm_a)` for the engine (which applies `-exp()` itself)
- Conv1d: GGUF stores `[kernel_size, conv_dim]` channel-major — loaded directly (no transpose; the earlier transpose heuristic was a bug, removed 2026-07-31)
- Uses existing `GGUFile::get_tensor_row_f32()` for dequantization, supporting all quant types

#### ✅ Phase 2: Engine Integration (COMPLETE)
- `WeightProvider` trait with `Send + Sync`, default `load_layer_weights` with rayon parallelism
- `GGUFWeightProvider` implements `WeightProvider` with row-based slice reads + cached non-layer weights
- `Shards` implements `WeightProvider` (trivial wrapper around existing methods)
- `StreamingOrnith` uses `Box<dyn WeightProvider>` — supports both safetensor and GGUF at runtime
- `StreamingOrnith::open_gguf(gguf_path, tokenizer_path)` — opens GGUF, extracts config from metadata
- `main.rs` auto-detects `.gguf` files and dispatches to the correct engine
- `extract_ornith_config()` reads all params from GGUF metadata (including `qwen35.ssm.*` keys)

#### ✅ Phase 3: Chat Template + Tokenizer (COMPLETE)
- Add chat message formatting to the REPL (`<|im_start|>user\n...<|im_end|>`, system prompt, etc.)
- Wire tokenizer from GGUF metadata or require `tokenizer.json` path
- Handle `<think>` blocks in model output (strip for display, keep for reasoning)
- Handle stop tokens (`<|im_end|>` = 248046, `<|endoftext|>` = 248044 — verified 2026-08-01)

#### Phase 4: Testing & Hardening
- Test with `/home/xander/Downloads/models/ornith-1.0-9b-Q8_0.gguf`
- Verify logits match safetensor path for first layer
- Measure BF16 precision drift impact on output quality
- Profile and optimize

---

## 8. Key C++ Reference Files

| File | Lines | Purpose | Key Functions |
|------|-------|---------|---------------|
| `src/models/qwen35.cpp` | 644 | Qwen3.5 model builder | `build_layer_attn`, `build_layer_attn_linear`, `build_qkvz`, `build_norm_gated`, `build_layer_ffn` |
| `src/models/delta-net-base.cpp` | 606 | DeltaNet core | `build_delta_net_autoregressive`, `build_delta_net_chunking`, `build_conv_state`, `build_recurrent_attn`, `build_delta_net` |
| `ggml/src/gguf.cpp` | 1697 | GGUF file I/O | `gguf_init_from_reader`, all getter/setter functions |
| `ggml/include/gguf.h` | 211 | GGUF API | All public API declarations |
| `src/llama-arch.cpp` | 1031 | Architecture registry | Architecture → tensor name mapping |
| `src/llama-arch.h` | 713 | Architecture enums | `llm_arch`, `llm_kv`, `llm_tensor` enums |
| `src/llama-model-loader.cpp` | 1698 | Weight loading | Tensor creation from GGUF |
| `src/llama-graph.cpp` | 3525 | Graph building | `build_attn`, `build_ffn`, `build_rs`, `build_norm` |
| `src/llama-context.cpp` | 4160 | Inference | `llama_decode`, `llama_encode`, `process_ubatch` |
| `src/llama-memory-recurrent.cpp` | 1264 | Recurrent state | State management for DeltaNet |
| `convert_hf_to_gguf.py` | 296 | Conversion script | CLI entry point |
| `gguf-py/gguf/tensor_mapping.py` | 2622 | Name mapping | HF tensor → GGUF name for all 171 architectures |
| `gguf-py/gguf/constants.py` | 5008 | GGUF constants | Arch IDs, tensor IDs, KV keys |

---

## 9. Architecture Registration for New Models

To add a new architecture to LeafcutterLLM, the pattern from llama.cpp is:

1. **Define architecture enum** — e.g., `LLM_ARCH_ORNITH` matching `"ornith"` string
2. **Register tensor names** — Map logical tensor names (ATTN_QKV, SSM_A, etc.) to GGUF name patterns
3. **Load hyperparameters** — Read KV metadata from GGUF into model config struct
4. **Create weight tensors** — Allocate/find tensors by name with correct shapes
5. **Build graph** — Wire tensor ops matching the model architecture

For Ornith (Qwen3.5), this is already done by llama.cpp. We just need to replicate it in Rust.

---

## 10. Performance Considerations

### 10.1 Quantization Support

The **storage-constrained user is the primary target**: small quantized models (Q4_K, Q6_K) that fit in limited storage. (Users who aren't storage-constrained can just download safetensors.)

Dequant kernels implemented and verified (July 2026): **Q4_K, Q5_K, Q6_K, Q8_K, Q8_0, Q4_0, Q4_1, IQ4_NL, IQ4_XS** — Q4_K and Q6_K produce token-identical output, proving correctness. On the Ornith-9B family: Q4_K_M = 5.3 GB, Q6_K = 6.9 GB, Q8_0 = 8.9 GB (vs ~18 GB f32 safetensors).

Unsupported types (listed in `quant.rs::is_supported()`): Q2_K, Q3_K, Q5_0, Q5_1, Q8_1, and the IQ-family beyond IQ4 — these fail with a clear "Unsupported quant type" error rather than corrupt weights.

Note: K-quants are dequantized on-the-fly inside AVX2 GEMM kernels (`q4_k_gemm.rs`, `q6_k_gemm.rs`); the engine does **not** hold dequantized f32 copies (that would be ~4× RAM).

### 10.2 Memory Mapping

GGUF supports memory-mapped loading via `params.no_alloc`:
- Metadata + tensor info is always loaded into memory
- Tensor data can be memory-mapped from disk (mmap)
- Only accessed pages are loaded from disk (lazy page fault)
- Saves RAM: model stays on disk until needed, OS handles paging

### 10.3 Streaming Inference

The current streaming approach (process one token at a time, maintain state) already matches llama.cpp's recurrent inference pattern (`build_delta_net_autoregressive` for n_seq_tokens=1). No fundamental changes needed.

### 10.4 Measured Performance (2026-07-31, Ornith-9B Q4_K_M, same hardware)

Same prompt (73 tokens), same machine (AMD Ryzen 7 5800HS, 8C/16T, AVX2, CPU-only):

| Metric | Ollama 0.5.x | Leafcutter | Ratio |
|--------|-------------|-----------|-------|
| Generation | 5.12 t/s (0.195 s/tok) | ~0.31 t/s (3.2 s/tok) | **16× slower** |
| Prompt processing | 30 t/s | n/a (no prefill) | — |

Leafcutter wall times (20 max tokens): 48.5 s with prefetch, 78.7 s without (prefetch = 1.6×). Thread count (4 vs 16) made **no** difference — the bottleneck is I/O, not compute.

**Post-fix (Phase 5 cache landed):** 20 max tokens = **34.6 s** wall (was 48.5 s). First token ~19.8 s (cold load of all 32 layers once), then **~0.78 s/tok** steady-state (was ~2.4 s/tok). `LEAFCUTTER_NO_CACHE=1` reproduces the old behavior (55.8 s/20 tok) → cache is the ~1.6× win, and it is **bit-exact** (identical generated text with and without cache). Remaining gap to Ollama (0.195 s/tok) is now **compute-bound**, dominated by dequant+GEMM kernels plus lm_head.

**lm_head update (2026-08-01):** f32 `output.weight` cache implemented in engine.rs (`load_lm_head_cache`, 3.79 GiB) → 20-token run = **28.96 s** (was 34.6 s), bit-exact. But lm_head is **memory-bandwidth-bound at ~180 ms/token** (reads 4 GB f32/token) — NOT the hoped ~2 ms. Quantized Q6_K GEMM (`q6_k_matmul_transposed_b`) measured **87.8 ms avg**, 2× faster and −3 GB RAM. Both approaches still leave lm_head as the dominant per-token cost; Top-K preselection remains the big win (167→~0.2 ms est.).

### 10.5 Root-Cause: Weights Re-Loaded From Disk Every Token

`Engine::forward_native()` (inference/engine.rs:941) streams layers one at a time, but for **every token** it:
1. Calls `load_layer(0..32)` — re-parses + re-dequantizes **all 32 layers' weights** from the mmap (Q4_K blocks re-built from raw bytes each call)
2. Calls `model.file.drop_pages_from_cache()` **after every layer** (engine.rs:1134) → `MADV_DONTNEED` on the **entire 5.6 GB mmap**, evicting all pages from the OS page cache
3. So the next layer's read re-faults from disk every single token

This is the design documented in loader.rs:3 ("Only one layer's weights are resident in RAM at any time") — intended to bound RSS, but it means the whole 5.6 GB file is re-read from disk ~32× per generated token. Ollama/llama.cpp keep the model resident (mmap without MADV_DONTNEED, or fully loaded) and only dequantize in-kernel during the matmul.

**Fix plan (in priority order):**
1. ✅ **Cache weights in RAM across tokens** (DONE — `GGUFModel.layer_cache`, `get_layer()` in loader.rs). Holds all 32 layers' raw Q4_K/Q6_K blocks as `Arc<HashMap<String, Tensor>>` (~5.6 GB, fits RAM), so `load_layer` re-parse/dequant happens once per layer instead of once per token. Verified bit-exact vs `LEAFCUTTER_NO_CACHE=1`.
2. ✅ **Gate `drop_pages_from_cache()`** (DONE). MADV_DONTNEED on the whole 5.6 GB mmap after every layer evicted the page cache → full disk re-read per token. Now opt-in via `LEAFCUTTER_DROP_PAGES=1` (default OFF).
3. **Add prompt prefill to the streaming path** so `chat`/`run` don't throw away all but the last prompt token (this is also a correctness gap, not just perf).
4. **Batch the prompt through `forward_native(tokens)` once** (already exists for the full engine) and only stream the decode loop one token at a time.
5. **Fuse dequant into the GEMM inner loop** — current `q4_k_matmul_transposed_b` dequantizes a full column then does a separate SIMD dot; and the 167 ms/token `lm_head` dequant should be amortized (cache dequantized head, or fuse with top-k sampling).

Expected result: the I/O cost (~3 s/token) collapses to ~0; per-token cost becomes pure matmul (~tens of ms), approaching Ollama's 5 t/s and beyond with AVX2.

### 10.6 GPU Detection

Current: CPU-only (AVX2/FMA dequant + matmul kernels; 16 threads). The dev machine has an **AMD Radeon Vega iGPU** (Cezanne), which llama.cpp supports via the Vulkan backend.

Considerations (from llama.cpp, do not guess):
- llama.cpp GPU offload is per-layer (`--gpu-layers N`); for hybrid SSM models the recurrent/DeltaNet layers benefit less from GPU than the full-attention + FFN layers do.
- Vulkan is the portable path for AMD iGPUs (no ROCm for Vega integrated on Linux). `libvulkan_radeon.so` is present on this system.
- A Vega iGPU (~2-4 GB shared VRAM) cannot hold a 5.6 GB Q4_K model; full offload is impossible, partial offload (a few FFN/attention layers) is the realistic ceiling. CPU with AVX2 is likely competitive for this class of machine.
- Recommendation: **do the RAM-cache fix first (§10.5)** — it's a 16× win on existing hardware with zero new dependencies. Treat GPU (Vulkan partial offload) as a later, optional phase; auto-detect Vulkan presence and expose `--gpu-layers` to mirror llama.cpp, but don't let it block Phase 5.

---

## 11. Current Rust Code Status

### ✅ Verified Correct (No Changes Needed)

| Component | Status | Notes |
|-----------|--------|-------|
| A_log decay formula | **Correct** | `a = -a_log_val.exp()` matches Python `-exp(A_log)` — verified against C++ reference |
| Conv1d buffer | **Correct** | Buffer shift + 4-tap sum matches `ggml_ssm_conv` (channel-major, w[0]=oldest) |
| State update order | **Correct** | Decay → predict → update matches C++ `build_delta_net_autoregressive` |
| RMSNorm `(w)` — NO `+1` | **Correct** | GGUF bakes `+1` into norm weights at conversion (conversion/qwen.py:304-305); runtime multiplies directly. `linear_attn.norm.weight` is the one exception (no `+1` baked) |
| Sigmoid full-attention gate | **Correct** | Uses `sigmoid`, matches `qwen35.cpp:326` |
| GLM-style RoPE | **Correct** | Split-pair `(i, i+half)` matches `ggml_rope_multi` with sections |
| Softplus stability | **Correct** | Guards against `exp(x)` overflow in f32 |
| V-head pairing | **Correct** | Interleaved `h_v % n_qk` (llama.cpp `ggml_repeat_4d`), NOT blocked |
| Q4_K / Q6_K dequant | **Correct** | Token-identical output between Q4_K_M and Q6_K |
| Conv1d layout | **Correct** | No transpose; GGUF is channel-major `[4, 8192]` flat `c*conv_k + k` |

### 💡 Ollama Modelfile Findings

`ollama show ornith:9b` reveals:

| Setting | Value |
|---------|-------|
| System prompt | `You are Ornith, an open-source agentic coding assistant. Think step by step in a reasoning block, then act. Use the provided tools when they help. Be concise, correct, and direct: write working code and explain only what is non-obvious.` |
| Stop token | `<\|im_end\|>` (id 248046) |
| Template | `{{ .Prompt }}` (raw — no wrapping; renderer handles formatting) |
| Renderer | `ornith` (custom — likely wraps in `<\|im_start\|>user...`) |
| Temperature | 0.6 |
| top_k | 20 |
| top_p | 0.95 |

Key insight: Ollama's Modelfile uses `TEMPLATE {{ .Prompt }}` — the raw input goes straight to the `ornith` renderer, which formats it before feeding the model. Our REPL needs to do the same wrapping: `<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n<think>\n`.

### ⚠️ Known Issue: BF16 vs f32 Precision

The model was trained in BF16. Running in f32 causes the recurrent state to drift over 32 layers:

| Precision | " Paris" Logit | Top Token |
|-----------|---------------|-----------|
| Python BF16 (reference) | 16.25 | " Paris" |
| Python f32 | 0.43 | "\n" |
| Rust f32 (current) | 0.15 | "\n" |

Rust f32 and Python f32 agree within floating-point epsilon for layers 0-2 and within ~1% for layer 31. The divergence from BF16 is inherent — not a bug. Impact on real chat quality is unknown until tested.

---

## 12. Next Steps (Priority Order)

### ✅ Phase 1 — COMPLETE
GGUF weight loading bridge (`gguf_provider.rs`): name mapping, A_log inversion, conv1d direct load.

### ✅ Phase 2 — COMPLETE
Engine integration: `WeightProvider` trait, `GGUFWeightProvider`, `StreamingOrnith::open_gguf()`, auto-detection in `main.rs`.

### ✅ Phase 3 — Chat Template & Tokenizer (COMPLETE, July 2026)
- Chat message formatting verified against Ollama's `ornith` renderer (ollama/model/renderers/ornith.go → qwen35.go): system + user + `\n<|im_start|>assistant\n<think>` matches byte-for-byte
- GGUF-native tokenizer wired (`GgufBpeTokenizer::from_gguf`, vocab=248320); HF tokenizer (248070) auto-rejected on vocab mismatch
- Stop token `<|im_end|>` (id **248046**, verified via `check_ornith_vocab` against the GGUF tokenizer table) is the ChatML EOS; generation stops at it. `<|endoftext|>` is a *different* token at id 248044. (Earlier doc versions wrongly said `<|im_end|>` = 248044 — corrected 2026-08-01.)
- Verified special-token IDs from the GGUF tokenizer table: `248044=<|endoftext|>`, `248045=<|im_start|>`, `248046=<|im_end|>`, `248047=<|object_ref_start|>`, `248068=<think>`, `248069=</think>`
- `<think>` reasoning blocks are generated and should be stripped from display (ollama PARSER ornith does this)
- System prompt matches Ollama Modelfile exactly

### ✅ Phase 4 — End-to-End Test (COMPLETE, July 2026)
- Tested on all three quantizations of Ornith-9B: Q4_K_M, Q6_K, Q8_0
- All produce coherent, on-topic English reasoning blocks (`The user said "Hello - this is a simple...`)
- Tokens 1-2 match Ollama reference; token 3 diverges (residual numeric diff, under investigation)
- Q4_K_M ↔ Q6_K token-identical output confirms dequant correctness

### 🚧 Phase 5 — Performance (IN PROGRESS — the current blocker)

**What Phase 5 actually needs (REASSESSED 2026-07-31):**

The strategy doc was stale on two items:
1. **Prompt prefill EXISTS** — `generate_native()` and `generate_streaming_with_stops()` both call `forward_native(tokens)` before the decode loop. The `"only last prompt token processed"` claim is outdated — the prefill path is wired.

2. **I/O bottleneck largely fixed** — `GGUFModel.layer_cache + drop_pages_from_cache()` gated off: 34.6 s/20 tok (was 48.5 s), ~0.78 s/tok steady-state (was ~2.4 s/tok). Bit-exact.

**The REAL remaining bottleneck is `lm_head_projection` at 167 ms/token** (identified in §10.4):
- Dequantizes the entire 248K×4096 vocab table one row at a time (248,320 par_iter calls per token)
- Each call: mmap read + q4_k/q6_k dequant + SIMD dot product
- 167 ms is 3-4× the rest of the layer pipeline combined

**Fix options for lm_head:**
1. **Cache dequantized lm_head weights** — ~4 GB for f32 (= peak RSS goes from ~5.6 GB to ~9.6 GB, which may not fit). Instead, keep the raw Q4_K blocks and do the dequant once per token (current, 167ms). Could try caching only the 8192 most-frequent token rows (3% of vocab covers ~95% of tokens in practice) — warm-cache the rest.
2. **Fuse dequant into SIMD dot** — already what `get_tensor_row_f32_into` + `simd_dot_product` does. The bottleneck is the serial row-by-row mmap+dequant.
3. **Top-K preselection** — if top_k=20, you only need the 20 highest logits. You can sample from a partial dequant by maintaining a min-heap of the top K rows as you iterate. Worst-case: you dequant all rows (same as today). Average case: you dequant ~K + extra to break ties (~few hundred rows). This is the best optimization for the decoding loop: 248K dequants → ~200 dequants = 1000× speedup on the lm_head step.
4. **Batch dequant** — dequant the entire lm_head tensor to f32 in one shot using `rayon` from the raw Q4_K blocks in `layer_cache`. One large parallel job (248K rows × 4096 cols) is faster than 248K individual jobs because: (a) cheaper dispatch overhead, (b) better cache locality, (c) better SIMD utilization on contiguous data. Estimated: 248K rows × 4096 / (4096/32 blocks) = 248K × 128 blocks = ~31M block dequants, each ~60-100 cycles = ~3 billion cycles = ~100 ms at 3 GHz. Roughly matches the current 167 ms, so batch dequant alone won't help much.

**Recommended lm_head fix approach (in order):**
1. **Top-K preselection** — the `par_iter` over all 248K rows is wasted when you only need the top 20. Implementation: iterate sequentially once, maintaining a min-heap of the top-N (top_k=40) rows. Only dequant a row when it might enter the heap. The iterative scan means no par_iter overhead, but ~1000× fewer dequants → lm_head goes from 167 ms to ~0.2 ms.
2. **Output-layer weight cache** — if memory allows, store `output.weight` dequantized as f32 (248320 × 4096 × 4 = ~4 GB). With the layer cache already at ~5.6 GB, total would be ~9.6 GB. On a 16 GB machine this fits, but on 8 GB it doesn't. Make it optional (`LEAFCUTTER_CACHE_HEAD=1`).

**lm_head measurements (2026-08-01, after implementing f32 output cache):**
- **f32 output cache IMPLEMENTED** — `load_lm_head_cache()` in engine.rs dequantizes `output.weight` (Q6_K [4096, 248320], 834 MB raw) once into f32 (`cached_lm_head`, 1,017,118,720 elements = 3.79 GiB) at model load; 20-token run dropped from 34.6 s → **28.96 s**, output bit-exact vs no-cache (`LEAFCUTTER_NO_CACHE=1` gate in loader.rs proves layer-cache bit-exactness).
- **BUT lm_head is still ~180 ms/token** (NOT the hoped ~2 ms). It is **memory-bandwidth-bound**: reads 4 GB of f32 per token; dequant is no longer the cost — the dot product over 248K×4096 f32 is.
- **Quantized Q6_K GEMM alternative measured: 87.8 ms avg** via `q6_k_matmul_transposed_b` (bench binary `bench_lmhead_q.rs`, since deleted) — 2× faster than f32 cache AND uses ~3 GB less RAM. Note: `q6_k_matmul` (non-transposed variant) panicked at q6_k_gemm.rs:226 in the bench, so the transposed path is the proven one.
- **Decision needed**: keep f32 cache (simple, 28.96 s, +3.79 GB) vs swap to quantized Q6_K block cache (~87.8 ms/tok lm_head, −3 GB RAM). Both uncommitted (git status: `M rust/src/bin/prof_lm_head.rs`, `M rust/src/inference/engine.rs`).

**GPU detection is still CPU-only.** AMD Radeon Vega iGPU (Cezanne) is present. Monitoring: leave as a Phase-6 optional polish, not blocking Phase 5.

### ✅ Phase 5B — Ollama-Like UX (COMPLETE, 2026-08-01)

**Trigger:** User tested `generate --raw "Hello"` and was unhappy: (1) noisy output (per-layer `materializing` spam, banners), (2) wants an Ollama-like flow where *they* type their own prompt and the model responds properly (not a raw/premade prompt), (3) response quality was bad and no thinking block was shown. Clarified: "I dont need it to show me the layers, just the responds streaming, and the thinking as well."

**Root-cause diagnosis:**
1. `--raw` **bypasses the chat template entirely** (main.rs:1544) — Ornith is a ChatML reasoning model; fed raw `Hello` it does raw text-continuation (`...I'm a student in the University of Malaya...`), not an understanding.
2. Even without `--raw`, `cmd_generate_native` uses `apply_chat_template_from_gguf` (chat_template.rs:58) which ends at `<|im_start|>assistant\n` with **NO `<think>`** opener. The correct template is `profiles::render_chat_prompt` (profiles.rs:425-426) which appends `\n<think>\n` for `opens_with_thinking` models — but `generate` doesn't use it.
3. `generate` is **non-streaming + noisy**: batches at the end (main.rs:1597 `engine.generate()`); loader prints `[loader] materializing ssm_conv1d.weight` ×24 (loader.rs:621-623, ungated), plus `[lm_head] cached...`, `⚠️ HF tokenizer vocab mismatch`, `📝 Prompt tokens`, `🌿 banner`, `DeltaNet: qk_heads=...`.

**Key enabler:** the fast `Engine` already has everything needed — `generate_streaming_with_stops` (engine.rs:729) does prefill + KV cache + layer cache + per-token callback, and the thinking-block callback pattern (swallow `<think>`=248068/`</think>`=248069, print thinking as `💭…`) already exists in `cmd_run` (main.rs:996-1029).

**Plan (all shipped):**
- **A. Native streaming chat REPL on the fast Engine** — `run --engine native` now streams via the fast `Engine`:
  1. Prompt: `profiles::resolve_profile(arch)` → `render_chat_prompt(&profile, system, &history)` — adds `<think>` for Ornith, matches Ollama exactly.
  2. Streaming: `engine.generate_streaming_with_stops(tokens, max, temp=0.6, top_p=0.95, &stop_ids, cb)` with the existing thinking-block callback.
  3. Stop tokens from profile: `<|im_end|>`=**248046**, `<|endoftext|>`=248044 (verified via GGUF tokenizer).
  4. Multi-turn: keep `history: Vec<(role, content)>`, re-render each turn.
  5. Sampling = ornith profile defaults (temp 0.6, top_k 20, top_p 0.95).
- **B. Quiet by default** — loader/engine debug logs gated behind `LEAFCUTTER_DEBUG=1` / `-v`; only the model-load header (banner + arch + layer count) shows. Stream only generated text.
- **C. `generate` fixed for correctness** (one-shot path) — non-`--raw` uses `render_chat_prompt` (with `<think>`) and routes through `generate_streaming_with_stops` so output streams and stops at `<|im_end|>`.
- **D. Acceptance test PASSED** — `leafcutter run ornith` → thinking block streams as `💭…`, then the answer, clean stop, quiet output. Example session:
  ```
  >>> hey there
  💭The user is just saying "hey there"...
  Hey! 👋 I'm Ornith — your open-source agentic coding assistant...
  ornith-1.0-9b-Q4_K_M.gguf | out=105 | 63.46s | 1.65 tok/s | RAM 8.1 GB
  ```

**Post-Phase-5B wrap-up (same session):** two correctness bugs found by the user and fixed:
- **GPT-2 byte-level decode corruption** — `Hey! 👋` printed as `Hey! ��` because multi-byte chars split across byte-level tokens were lossy-decoded per token. Fixed with `decode_bytes()` + a streaming UTF-8 buffer (`emit_complete_utf8`) in engine.rs; emoji/Latin-1 now render correctly.
- **lm_head f32 cache → Q6_K block cache** — the 3.79 GiB f32 cache was swapped for native Q6_K blocks (~0.8 GB) computed via `q6_k_matmul_transposed_b`. Peak chat RAM dropped **11.1 → 8.1 GB** and lm_head is ~2× faster (87.8 ms vs ~180 ms). Bit-identical logits.

**Verified tokenizer facts (2026-08-01, via GGUF tokenizer table):** `248044=<|endoftext|>`, `248045=<|im_start|>`, `248046=<|im_end|>`, `248047=<|object_ref_start|>`, `248068=<think>`, `248069=</think>`. profiles.rs ornith stop tokens (248046/248044) are **correct**.

### Phase 6 — Polish
- BF16 dequantization in tensor.rs matmul
- Memory-mapped weight access with page eviction (MADV_DONTNEED) — RE-EVALUATE: this actively kills performance (§10.5)
- Handle edge cases: multi-turn chat, streaming output, tool calls
