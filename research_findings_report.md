# Research Findings Report: GGUF Dimensions, Qwen3.5 Architecture, GQA/KV Cache

## 1. GGUF Dimension Convention: CONFIRMED

### The Theory (User's Hypothesis)
GGUF stores tensor dimensions as `[inner_dim, outer_dim]` (columns-first), which is the reverse of PyTorch's `[outer_dim, inner_dim]` (rows-first). Loading requires: reverse dims, reshape, then transpose for matmul.

### Our Verification

We created a GGUF file with a **traceable weight matrix** (each element has a unique value encoding its row/column position), saved it, read it back, and verified the reconstruction:

```python
# Original PyTorch weight: [out_dim=4, in_dim=10]
W = [[0,1,2,3,4,5,6,7,8,9],
     [1000,1001,...,1009],
     [2000,2001,...,2009],
     [3000,3001,...,3009]]

# GGUF stores shape as: [10, 4] (REVERSED)
# GGUF stores data row-major: [0,1,2,3,4,5,6,7,8,9,1000,1001,...]

# After reverse+reshape: [4, 10] - MATCHES original exactly
```

**Result: MATCH confirmed at every element position.**

### Full Matmul Pipeline Verification

We also verified the complete matmul pipeline for GQA's non-square K projection:

```
Hidden states:  [batch=1, seq=2, hidden_size=16]
W_k (PyTorch):  [kv_dim=8, hidden_size=16]   (GQA: kv_dim < hidden_size)
k = hidden @ W_k.T -> [1, 2, 8]

GGUF stores W_k as: [16, 8]
After reverse+reshape: [8, 16] (matches PyTorch)
k = hidden @ W_k_loaded.T -> [1, 2, 8]

MATCH: torch.allclose(k_torch, k_loaded) = True
```

### Verdict: THE FIX IS CORRECT

The user's fix logic is mathematically sound:
1. Read GGUF dimensions: `[in_dim, out_dim]`
2. Reverse: `[out_dim, in_dim]`
3. Reshape data to reversed shape
4. Use `W.T` in matmul: `output = input @ W.T`

---

## 2. Qwen3.5 Architecture: CRITICAL FINDING

### What Qwen3.5 Actually Is

Qwen3.5 is **NOT** a pure transformer with sliding window attention like Qwen3.
It is a **hybrid architecture** combining:

| Component | Type | Ratio | Purpose |
|-----------|------|-------|---------|
| Gated DeltaNet | Linear attention / SSM-like | 3/4 layers | Efficient long-context processing |
| Gated Attention | Full softmax attention | 1/4 layers | Precise content retrieval |
| FFN | SwiGLU | Every layer | Feed-forward processing |

### Official Config (Qwen3.5-0.8B)

From the official `config.json` on HuggingFace:

```json
{
  "model_type": "qwen3_5_text",
  "hidden_size": 1024,
  "num_hidden_layers": 24,
  "intermediate_size": 3584,
  "vocab_size": 248320,
  "max_position_embeddings": 262144,
  
  "full_attention_interval": 4,
  "layer_types": [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ... (repeats 6 times)
  ],
  
  // FULL ATTENTION layers (every 4th):
  "num_attention_heads": 8,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "attn_output_gate": true,
  
  // LINEAR ATTENTION layers (Gated DeltaNet):
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 16,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "linear_conv_kernel_dim": 4,
  
  "rope_parameters": {
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "rope_type": "default"
  }
}
```

### Key Implications for Your Engine

**1. Two COMPLETELY Different Attention Mechanisms**

| Aspect | Full Attention Layers | Linear Attention (Gated DeltaNet) Layers |
|--------|----------------------|------------------------------------------|
| KV cache | Standard K/V tensors per head | **Recurrent state matrix S** (fixed size) |
| Attention | `softmax(Q @ K.T / sqrt(d)) @ V` | Delta rule state update (no softmax) |
| RoPE | Applied to Q and K | NOT used (uses CausalConv1D instead) |
| Head dim | 256 | 128 (different!) |
| KV heads | 2 | 16 (different!) |
| Cache type | Append-only (grows with seq) | Fixed-size recurrent state |
| Position | RoPE position IDs | Conv1D + internal gating |

**2. The SSM Code Path Question**

Your engine auto-detects SSM vs attention from tensor names. **This IS correct** — Qwen3.5's linear attention layers ARE effectively SSM layers. They use:
- A recurrent state matrix `S` instead of KV cache
- Delta rule updates instead of attention scores
- Gated decay mechanisms (similar to Mamba-2)

**3. Tensor Name Patterns for Auto-Detection**

```
Full attention layers:
  blk.{i}.attn_q.weight, blk.{i}.attn_k.weight, blk.{i}.attn_v.weight
  blk.{i}.attn_output.weight
  blk.{i}.attn_q_norm.weight, blk.{i}.attn_k_norm.weight  (Q/K RMSNorm)
  blk.{i}.attn_output_gate.weight  (output gating)

Linear attention (Gated DeltaNet) layers:
  blk.{i}.linear_q.weight, blk.{i}.linear_k.weight, blk.{i}.linear_v.weight
  blk.{i}.linear_conv1d.weight  (CausalConv1D replaces RoPE)
  blk.{i}.linear_decay_gate.weight, blk.{i}.linear_alpha.weight
  blk.{i}.linear_output_gate.weight
  
Shared:
  blk.{i}.ffn_gate.weight, blk.{i}.ffn_up.weight, blk.{i}.ffn_down.weight
  blk.{i}.attn_norm.weight, blk.{i}.ffn_norm.weight
```

### Verdict: If Your Engine Only Handles Standard Attention + Sliding Window

**You CANNOT run Qwen3.5 correctly yet.** You need:
1. A Gated DeltaNet / linear attention implementation
2. Different head dimensions per layer type (256 vs 128)
3. Recurrent state management for linear attention layers
4. Conv1D kernel for positional encoding in linear layers

However, **the weight transpose fix is still correct** for both layer types — the dimension reversal applies to ALL GGUF tensors.

---

## 3. RoPE seq_offset Tracking

### How Position IDs Work in Prefill vs Decode

```
Phase 1: PREFILL (process prompt tokens all at once)
  Input:  ["The", "cat", "sat"]  (3 tokens)
  position_ids: [0, 1, 2]
  RoPE computes cos/sin for positions 0, 1, 2
  KV cache stores K/V for positions 0, 1, 2
  seq_offset after prefill: 3

Phase 2: DECODE (generate one token at a time)
  Iteration 1:
    Input: ["on"] (1 new token)
    position_ids: [3]  (continues from prefill)
    RoPE computes cos/sin for position 3 only
    KV cache appends K/V at position 3
    seq_offset: 4
    
  Iteration 2:
    Input: ["the"]
    position_ids: [4]
    RoPE computes cos/sin for position 4 only
    KV cache appends K/V at position 4
    seq_offset: 5
```

### Key Rules for seq_offset

1. **After prefill**: `seq_offset = prompt_length`
2. **Each decode step**: `position_id = seq_offset`, then `seq_offset += 1`
3. **RoPE must use absolute positions** (not relative within the batch)
4. **KV cache concatenates along sequence dimension** (position 0, 1, 2, 3, 4...)

### Common Bug Pattern

```python
# WRONG: Resetting position to 0 each decode step
position_ids = torch.arange(new_tokens.shape[1])  # [0] each time!

# CORRECT: Continue from where prefill left off
position_ids = torch.arange(seq_offset, seq_offset + new_tokens.shape[1])
```

---

## 4. GQA KV Cache Shape Verification

### The Correct Shape

```python
# KV cache shape: [batch_size, num_kv_heads, seq_len, head_dim]
# NOT [batch_size, num_heads, ...] — fewer heads for K/V!

# Example: Llama-3 8B
hidden_size = 4096
num_heads = 32       # Query heads
num_kv_heads = 8     # Key/Value heads (GQA: 4x fewer)
head_dim = 128       # 4096 / 32

# After prefill of 100 tokens:
kv_cache_k.shape = [1, 8, 100, 128]  # 8 KV heads, not 32!
kv_cache_v.shape = [1, 8, 100, 128]

# At attention time: repeat K/V heads to match query head count:
k_expanded = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
# [1, 8, 100, 128] -> [1, 32, 100, 128] (each KV head shared by 4 query heads)
```

### How repeat_kv Maps to PyTorch

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    hidden_states: [batch, num_kv_heads, seq_len, head_dim]
    returns:       [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # Expand and reshape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

# In your Rust code, this is equivalent to:
# for each query head group (num_kv_groups):
#   share the same K/V head across group_size query heads
```

### Qwen3.5 Special Case: Different KV Heads Per Layer Type

```python
# FULL ATTENTION layers (every 4th layer):
num_kv_heads = 2
head_dim = 256
kv_cache_shape = [batch, 2, seq_len, 256]

# LINEAR ATTENTION layers (Gated DeltaNet):
linear_num_kv_heads = 16
linear_head_dim = 128
# NO KV CACHE — uses recurrent state matrix S instead!
# State shape per head: [d_k, d_v] = [128, 128]
```

---

## 5. Reference Comparison Script (llama-cpp-python)

```python
#!/usr/bin/env python3
"""
Reference inference script using llama-cpp-python.
Run this on a known model, then compare token-by-token with your native engine.
"""
from llama_cpp import Llama
import sys

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "model.gguf"
PROMPT = "The capital of France is"

def run_reference():
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=0,  # CPU for determinism
        seed=42,         # Fixed seed
        verbose=False,
    )
    
    # Tokenize the prompt to get input token IDs
    tokens = llm.tokenize(PROMPT.encode())
    print(f"Prompt: '{PROMPT}'")
    print(f"Input token IDs: {tokens}")
    print(f"Input token count: {len(tokens)}")
    
    # Greedy generation (temperature=0)
    result = llm(
        PROMPT,
        max_tokens=20,
        temperature=0.0,      # Greedy — deterministic
        top_p=1.0,
        top_k=1,              # Only consider top token
        repeat_penalty=1.0,   # No repetition penalty
        seed=42,
    )
    
    generated_text = result["choices"][0]["text"]
    print(f"\nGenerated text: '{generated_text}'")
    
    # Now get per-token logits for comparison
    print("\n--- Per-Token Logits (first 5 steps) ---")
    
    # Get logits for the prompt (prefill)
    output = llm(input_ids=tokens, embedding=False)
    prompt_logits = output["choices"][0].get("logprobs", {})
    
    # Generate step by step, printing logits
    current_tokens = list(tokens)
    for step in range(5):
        output = llm(input_ids=current_tokens, embedding=False)
        logits = output["choices"][0]["logprobs"]["token_logprobs"]
        
        # Get the next token (last logit)
        next_token_logits = output["choices"][0]["logprobs"]
        
        result = llm.create_completion(
            prompt="",
            input_ids=current_tokens,
            max_tokens=1,
            temperature=0.0,
            seed=42,
        )
        next_token = result["choices"][0]["logprobs"]["tokens"][0]
        next_token_id = llm.tokenize(next_token.encode(), add_bos=False)[0]
        
        print(f"Step {step}: token_id={next_token_id}, token='{next_token}'")
        current_tokens.append(next_token_id)
    
    print(f"\nFinal token sequence: {current_tokens}")
    print(f"Decoded: {llm.detokenize(current_tokens).decode('utf-8', errors='replace')}")

if __name__ == "__main__":
    run_reference()
```

**Install and run:**
```bash
pip install llama-cpp-python --no-cache-dir
python reference_inference.py model.gguf
```

---

## Summary of All Findings

| Topic | Finding | Impact |
|-------|---------|--------|
| GGUF dimensions | **[in_dim, out_dim]** reversed from PyTorch. Reverse+reshape+transpose fix is **correct** | High — this was the main bug |
| Qwen3.5 architecture | **Hybrid**: Gated DeltaNet (3/4) + Gated Attention (1/4). NOT sliding window. Uses SSM-like recurrent state | **Critical** — engine needs linear attention support |
| seq_offset tracking | Absolute position IDs, increment from prefill length. `position = seq_offset` then `seq_offset++` each decode step | Medium — verify your increment logic |
| GQA KV cache | Shape `[batch, num_kv_heads, seq_len, head_dim]`, expanded via repeat_interleave. Qwen3.5 has DIFFERENT dims per layer type | Medium — verify cache shape matches |

### Recommended Next Steps

1. **Apply the weight transpose fix** — it's mathematically correct
2. **Test with a PURE attention model first** (Llama, Mistral) before tackling Qwen3.5
3. **For Qwen3.5 support**, implement:
   - Gated DeltaNet forward pass (delta rule state update)
   - Separate cache paths: KV cache for attention layers, recurrent state for linear layers
   - Per-layer head dimension routing (256 for attention, 128 for linear)
