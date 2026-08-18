# Comprehensive AI Model Architecture & GGUF Reference Guide

> **Wrap-up note (2026-08-01):** Reference guide used during development. The
> architectures shipped end-to-end on the native GGUF engine are Llama, Mistral/
> Ministral, Qwen2, Gemma, Phi, and **Qwen3.5 / Ornith (DeltaNet hybrid)** —
> see `ARCHITECTURE.md` for the current state. The Qwen3.5/Ornith
> architecture details here match the verified implementation (see
> the CHANGELOG 2026-06-30 entry and `ARCHITECTURE.md`).

## Table of Contents
1. [Transformer Fundamentals](#1-transformer-fundamentals)
2. [Llama Architecture (Full Code)](#2-llama-architecture)
3. [Qwen3 Architecture (Full Code)](#3-qwen3-architecture)
4. [Mistral Architecture (Full Code)](#4-mistral-architecture)
5. [Mixtral MoE Architecture](#5-mixtral-moe-architecture)
6. [GGUF Format & Parsing](#6-gguf-format--parsing)
7. [Inference Engines](#7-inference-engines)
8. [Key Open Source Resources](#8-key-open-source-resources)

---

## 1. Transformer Fundamentals

### 1.1 Core Attention Mechanism ("Attention Is All You Need")

The attention mechanism computes three vectors for each token: **Query (Q)**, **Key (K)**, and **Value (V)**.

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

### 1.2 Complete Transformer from Scratch (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# SELF-ATTENTION (THE HEART OF THE TRANSFORMER)
# ============================================================
class SelfAttention(nn.Module):
    """Single-head self-attention mechanism"""
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"
        
        # Q, K, V linear projections
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]  # batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split into heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # Apply linear projections
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Compute attention scores: Q @ K^T
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # Apply mask if provided (for causal/decoding)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # Scale and apply softmax
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        
        # Weighted sum of values: attn @ V
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


# ============================================================
# MULTI-HEAD ATTENTION (Used in all modern LLMs)
# ============================================================
class MultiHeadAttention(nn.Module):
    """Multi-head attention with scaled dot-product"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


# ============================================================
# POSITION-WISE FEED-FORWARD NETWORK
# ============================================================
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ============================================================
# ENCODER LAYER
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# ============================================================
# DECODER LAYER (for autoregressive generation)
# ============================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked self-attention (causal)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Cross-attention to encoder output
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# ============================================================
# POSITIONAL ENCODING
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ============================================================
# COMPLETE TRANSFORMER
# ============================================================
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        return self.fc_out(dec_output)
```

---

## 2. Llama Architecture

**Source:** `huggingface/transformers/src/transformers/models/llama/modeling_llama.py`

Llama uses a **decoder-only** transformer with these key innovations:
- **RMSNorm** instead of LayerNorm (more stable)
- **RoPE** (Rotary Position Embedding) - no separate positional encoding needed
- **GQA** (Grouped Query Attention) - shares K/V heads across query heads
- **SwiGLU activation** in the MLP

### 2.1 Llama RMSNorm

```python
class LlamaRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

### 2.2 RoPE (Rotary Position Embedding)

```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.rope_type = config.rope_parameters["rope_type"]
        
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies RoPE to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

### 2.3 Llama MLP (SwiGLU)

```python
class LlamaMLP(nn.Module):
    """SwiGLU MLP: gate_proj uses SiLU, multiplies with up_proj"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]  # typically "silu"

    def forward(self, x):
        # SwiGLU: silu(gate) * up -> down
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### 2.4 GQA - Grouped Query Attention

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat K/V heads to match number of query heads (for GQA)"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Llama uses GQA: num_key_value_heads < num_attention_heads
# e.g., 8 K/V heads for 32 query heads means n_rep = 4
```

### 2.5 Llama Attention (Complete)

```python
class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project and reshape
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Use cached K/V for generation
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # GQA: repeat K/V heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

### 2.6 Llama Decoder Layer (Building Block)

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, past_key_values=None):
        # Pre-norm architecture: norm BEFORE attention and MLP
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states  # Residual connection

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states  # Residual connection
        return hidden_states
```

### 2.7 Complete Llama Model

```python
class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None):
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Create position IDs
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)
        
        # Create causal mask
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, past_seen_tokens)
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Pass through decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

class LlamaForCausalLM(nn.Module):
    """Llama for text generation"""
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, labels=None, **kwargs):
        hidden_states = self.model(input_ids, **kwargs)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
        return {"loss": loss, "logits": logits}
```

---

## 3. Qwen3 Architecture

**Source:** `huggingface/transformers/src/transformers/models/qwen3/modeling_qwen3.py`

Qwen3 is very similar to Llama but with these **key differences**:
1. **Sliding Window Attention** - alternating layers use full vs. sliding attention
2. **Q/K RMSNorm** - applies normalization to query and key heads
3. **No bias in MLP** - `bias=False` in all linear layers

### 3.1 Qwen3 Attention (Key Differences from Llama)

```python
class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        # KEY DIFFERENCE 1: Layer types alternate between "full" and "sliding"
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        
        # KEY DIFFERENCE 2: RMSNorm on Q and K (improves training stability)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        # KEY DIFFERENCE 3: Sliding window for alternating layers
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        # KEY DIFFERENCE: Apply Q/K Norm BEFORE transpose
        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Attention with sliding window support
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

### 3.2 Qwen3 Model Structure

```python
class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        
        # Track if model uses sliding window layers
        self.has_sliding_layers = "sliding_attention" in config.layer_types

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None):
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Position IDs
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)
        
        # KEY DIFFERENCE: Create different masks for full vs sliding attention
        full_mask = create_causal_mask(config=self.config, inputs_embeds=inputs_embeds, 
                                       attention_mask=attention_mask, past_key_values=past_key_values,
                                       position_ids=position_ids)
        sliding_mask = None
        if self.has_sliding_layers:
            sliding_mask = create_sliding_window_causal_mask(
                config=self.config, inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, past_key_values=past_key_values,
                position_ids=position_ids
            )
        
        causal_mask_mapping = {
            "full_attention": full_mask,
            "sliding_attention": sliding_mask,
        }
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Pass through layers with appropriate mask
        for i, decoder_layer in enumerate(self.layers):
            mask = causal_mask_mapping[self.config.layer_types[i]]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )
        
        hidden_states = self.norm(hidden_states)
        return hidden_states
```

---

## 4. Mistral Architecture

**Source:** `huggingface/transformers/src/transformers/models/mistral/modeling_mistral.py`

Mistral is essentially Llama + **Sliding Window Attention** on ALL layers:

```python
class MistralAttention(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        
        # Same projections as Llama
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # KEY: Sliding window attention applied
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class MistralModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            MistralDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MistralRotaryEmbedding(config=config)

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None):
        inputs_embeds = self.embed_tokens(input_ids)
        
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)
        
        # KEY: Use sliding window causal mask if configured
        if self.config.sliding_window is not None:
            causal_mask = create_sliding_window_causal_mask(
                config=self.config, inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, past_key_values=past_key_values,
                position_ids=position_ids
            )
        else:
            causal_mask = create_causal_mask(config=self.config, inputs_embeds=inputs_embeds,
                                             attention_mask=attention_mask, past_key_values=past_key_values,
                                             position_ids=position_ids)
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states
```

---

## 5. Mixtral MoE Architecture

**Source:** `huggingface/transformers/src/transformers/models/mixtral/modeling_mixtral.py`

Mixtral replaces the MLP in each layer with a **Mixture of Experts** - 8 expert MLPs, routing each token to the top-2 experts.

### 5.1 MoE Block

```python
class MixtralSparseMoeBlock(nn.Module):
    """
    Mixture of Experts block with Top-2 routing.
    Each token is routed to 2 out of 8 experts.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts  # 8
        self.top_k = config.num_experts_per_tok       # 2
        
        # gating network
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        
        # 8 expert MLPs
        self.experts = nn.ModuleList([
            MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # router_logits: (batch * seq_len, num_experts)
        router_logits = self.gate(hidden_states)
        
        # Top-k routing weights
        routing_weights, selected_experts = torch.topk(
            torch.softmax(router_logits, dim=1, dtype=torch.float), 
            self.top_k, 
            dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        # Dispatch to experts
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        # Expert loop: for each expert, find which tokens route to it
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)  # (num_experts, top_k, num_tokens)
        
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.shape[0] == 0:
                continue
                
            # Gather tokens for this expert
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            # Compute expert output, weighted by routing weight
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            # Accumulate
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
```

---

## 6. GGUF Format & Parsing

### 6.1 What is GGUF?

GGUF (GGML Universal Format) is a binary format for storing ML models, optimized for:
- **Fast loading** via memory mapping
- **Self-describing** metadata (architecture, config, tokenizer)
- **Multiple quantization types** (Q4_K_M, Q8_0, F16, etc.)
- **Single file** contains both weights AND tokenizer

**File Structure:**
```
[Header: 24 bytes]
  - Magic: "GGUF" (4 bytes)
  - Version: uint32 (4 bytes)
  - Tensor Count: uint64 (8 bytes)
  - KV Count: uint64 (8 bytes)

[Metadata: Key-Value pairs]
  - general.architecture: "llama"
  - llama.context_length: 4096
  - llama.block_count: 32
  - llama.embedding_length: 4096
  - tokenizer.ggml.tokens: ["<s>", "the", "a", ...]
  - ... (30-50 key-value pairs)

[Tensor Info: name, shape, dtype, offset]
  - token_embd.weight: [4096, 32000] Q4_K
  - blk.0.attn_q.weight: [4096, 4096] Q4_K
  - blk.0.attn_output.weight: [4096, 4096] Q4_K
  - ...

[Padding to alignment]

[Tensor Data: raw bytes]
```

### 6.2 GGUF Parser in Python (llama.cpp)

**Source:** `ggml-org/llama.cpp/gguf-py/gguf/gguf_reader.py`

```python
import numpy as np
from collections import OrderedDict
from typing import NamedTuple, Any

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

class GGUFValueType:
    UINT8 = 0;    INT8 = 1;     UINT16 = 2;   INT16 = 3
    UINT32 = 4;   INT32 = 5;    FLOAT32 = 6;  BOOL = 7
    STRING = 8;   ARRAY = 9;    UINT64 = 10;  INT64 = 11;  FLOAT64 = 12

class ReaderTensor(NamedTuple):
    name: str
    tensor_type: int      # quantization type
    shape: np.ndarray     # uint32 array
    n_elements: int
    n_bytes: int
    data_offset: int      # offset in file to data
    data: np.ndarray      # the actual tensor data

class GGUFReader:
    """Parse GGUF files - reads metadata and tensor info"""
    
    def __init__(self, path: str):
        # Memory-map the file for fast access
        self.data = np.memmap(path, mode='r')
        offs = 0
        
        # 1. Read header
        magic = self._get(offs, np.uint32, override_order='<')[0]
        if magic != GGUF_MAGIC:
            raise ValueError('Invalid GGUF magic number')
        offs += 4
        
        version = self._get(offs, np.uint32)[0]
        offs += 4
        
        tensor_count = self._get(offs, np.uint64)[0]
        offs += 8
        
        kv_count = self._get(offs, np.uint64)[0]
        offs += 8
        
        # 2. Parse metadata key-value pairs
        self.fields: OrderedDict[str, Any] = OrderedDict()
        offs = self._build_fields(offs, kv_count)
        
        # 3. Parse tensor info
        offs, tensor_fields = self._build_tensor_info(offs, tensor_count)
        
        # Apply alignment
        alignment = self.fields.get('general.alignment', 32)
        padding = offs % alignment
        if padding != 0:
            offs += alignment - padding
        
        # 4. Build tensor data references
        self.data_offset = offs
        self.tensors: list[ReaderTensor] = []
        self._build_tensors(offs, tensor_fields)
    
    def _get(self, offset, dtype, count=1, override_order=None):
        """Read data from memory-mapped file"""
        itemsize = np.dtype(dtype).itemsize
        arr = self.data[offset:offset + itemsize * count].view(dtype=dtype)[:count]
        return arr
    
    def _build_fields(self, offs, count):
        """Parse metadata key-value section"""
        for _ in range(count):
            # Read key string
            key_len = self._get(offs, np.uint64)[0]
            offs += 8
            key = bytes(self._get(offs, np.uint8, key_len)).decode('utf-8')
            offs += key_len
            
            # Read value type
            val_type = self._get(offs, np.uint32)[0]
            offs += 4
            
            # Read value based on type
            val, offs = self._read_value(offs, val_type)
            self.fields[key] = val
        return offs
    
    def _read_value(self, offs, val_type):
        """Read a single value of given type"""
        if val_type == GGUFValueType.UINT32:
            return self._get(offs, np.uint32)[0], offs + 4
        elif val_type == GGUFValueType.INT32:
            return self._get(offs, np.int32)[0], offs + 4
        elif val_type == GGUFValueType.FLOAT32:
            return self._get(offs, np.float32)[0], offs + 4
        elif val_type == GGUFValueType.UINT64:
            return self._get(offs, np.uint64)[0], offs + 8
        elif val_type == GGUFValueType.STRING:
            slen = self._get(offs, np.uint64)[0]
            offs += 8
            s = bytes(self._get(offs, np.uint8, slen)).decode('utf-8')
            return s, offs + slen
        elif val_type == GGUFValueType.ARRAY:
            elem_type = self._get(offs, np.uint32)[0]
            offs += 4
            arr_len = self._get(offs, np.uint64)[0]
            offs += 8
            elements = []
            for _ in range(arr_len):
                val, offs = self._read_value(offs, elem_type)
                elements.append(val)
            return elements, offs
        elif val_type == GGUFValueType.BOOL:
            return bool(self._get(offs, np.uint8)[0]), offs + 1
        else:
            raise ValueError(f"Unsupported type: {val_type}")
    
    def _build_tensor_info(self, offs, count):
        """Parse tensor metadata section"""
        fields = []
        for _ in range(count):
            # Tensor name
            name_len = self._get(offs, np.uint64)[0]
            offs += 8
            name = bytes(self._get(offs, np.uint8, name_len)).decode('utf-8')
            offs += name_len
            
            # Dimensions
            n_dims = self._get(offs, np.uint32)[0]
            offs += 4
            dims = self._get(offs, np.uint64, n_dims)
            offs += 8 * n_dims
            
            # Data type
            dtype = self._get(offs, np.uint32)[0]
            offs += 4
            
            # Data offset (relative to tensor data start)
            data_offset = self._get(offs, np.uint64)[0]
            offs += 8
            
            fields.append({
                'name': name,
                'dims': dims,
                'dtype': dtype,
                'data_offset': data_offset
            })
        return offs, fields
    
    def _build_tensors(self, start_offs, fields):
        """Create tensor data references"""
        for field in fields:
            name = field['name']
            dims = field['dims']
            ggml_type = field['dtype']
            offset = field['data_offset']
            
            n_elements = int(np.prod(dims))
            np_dims = tuple(reversed(dims.tolist()))
            
            # Calculate bytes based on quantization type
            block_size, type_size = self._get_quant_sizes(ggml_type)
            n_bytes = n_elements * type_size // block_size
            
            data_offs = int(start_offs + offset)
            
            # Read tensor data
            if ggml_type == 0:  # F32
                tensor_data = self._get(data_offs, np.float32, n_elements).reshape(np_dims)
            elif ggml_type == 1:  # F16
                tensor_data = self._get(data_offs, np.float16, n_elements).reshape(np_dims)
            else:
                # Quantized - read as raw bytes
                tensor_data = self._get(data_offs, np.uint8, n_bytes)
            
            self.tensors.append(ReaderTensor(
                name=name, tensor_type=ggml_type,
                shape=dims, n_elements=n_elements,
                n_bytes=n_bytes, data_offset=data_offs,
                data=tensor_data
            ))
    
    def _get_quant_sizes(self, ggml_type):
        """Return (block_size, type_size) for quantization type"""
        # GGML quant sizes lookup
        sizes = {
            0: (1, 4),    # F32
            1: (1, 2),    # F16
            2: (1, 4),    # Q4_0 (32 elements -> 18 bytes)
            3: (1, 2),    # Q4_1
            # ... add more as needed
        }
        return sizes.get(ggml_type, (1, 4))
    
    def get_tensor(self, name: str):
        """Get tensor by name"""
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(f"Tensor {name} not found")
    
    def get_metadata(self):
        """Get all metadata"""
        return dict(self.fields)
```

### 6.3 Using the GGUF Python Library

```python
# Install: pip install gguf
import gguf

# Read a GGUF file
reader = gguf.GGUFReader("model.gguf")

# Access metadata
print(reader.fields["general.architecture"])  # "llama"
print(reader.fields["llama.block_count"])     # 32
print(reader.fields["llama.context_length"])  # 4096

# List all tensors
for tensor in reader.tensors:
    print(f"{tensor.name}: shape={tensor.shape}, type={tensor.tensor_type}")

# Get specific tensor
embedding_tensor = reader.get_tensor("token_embd.weight")
print(f"Embedding: {embedding_tensor.data.shape}")
```

### 6.4 JavaScript GGUF Parser (HuggingFace)

```javascript
// Install: npm install @huggingface/gguf
import { gguf } from "@huggingface/gguf";

const { metadata, tensorInfos } = await gguf("model.gguf");

console.log(metadata);
// {
//     version: 3,
//     tensor_count: 291n,
//     "general.architecture": "llama",
//     "llama.block_count": 32,
//     ...
// }

console.log(tensorInfos);
// [
//     { name: "token_embd.weight", shape: [4096n, 32000n], dtype: 15 },
//     { name: "blk.0.attn_q.weight", shape: [4096n, 4096n], dtype: 15 },
//     ...
// ]
```

---

## 7. Inference Engines

### 7.1 llama.cpp (C/C++)

**Key files for model loading:**
- `llama-model.cpp` - Model architecture loading
- `llama-model-loader.cpp` - GGUF tensor loading

**Model loading flow:**
```cpp
// 1. Create loader
llama_model_loader ml(fname, splits, use_mmap, check_tensors);

// 2. Load architecture from metadata
model.load_arch(ml);    // reads "general.architecture" 
model.load_hparams(ml); // reads all hyperparameters
model.load_vocab(ml);   // reads tokenizer

// 3. Load tensors
model.load_tensors(ml);

// 4. Load tensor data from file
ml.load_all_data(ctx, bufs, use_mlock);
```

**Using llama-cpp-python:**
```python
from llama_cpp import Llama

# Load GGUF model
llm = Llama(model_path="model.gguf", n_ctx=4096, n_gpu_layers=35)

# Or download from HuggingFace
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen3-8B-GGUF",
    filename="qwen3-8b-q4_k_m.gguf",
    n_ctx=4096,
)

# Generate
output = llm("Hello, how are you?", max_tokens=100)
print(output["choices"][0]["text"])

# Chat format
output = llm.create_chat_completion([
    {"role": "user", "content": "Explain quantum computing"}
])
```

### 7.2 HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load from HuggingFace Hub
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Generate
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))

# With cache for faster generation
past_kv = None
for i in range(max_tokens):
    outputs = model(**inputs, past_key_values=past_kv, use_cache=True)
    past_kv = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1], dim=-1)
    inputs = {"input_ids": next_token.unsqueeze(0)}
```

### 7.3 vLLM (High-Throughput Serving)

```python
from vllm import LLM, SamplingParams

# Initialize
llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    dtype="float16",
    gpu_memory_utilization=0.95
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512
)

# Batch inference
prompts = ["Explain AI:", "What is Python?"]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### 7.4 SGLang (Qwen3.5 recommended)

```bash
# Install: uv pip install 'git+https://github.com/sgl-project/sglang.git'

# Launch server
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-4B \
    --port 8000 \
    --tp-size 1 \
    --context-length 262144 \
    --reasoning-parser qwen3

# Multi-token prediction for faster generation
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-4B \
    --tp-size 1 \
    --context-length 262144 \
    --speculative-algo NEXTN \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4
```

---

## 8. Key Open Source Resources

| Resource | URL | What You'll Find |
|----------|-----|------------------|
| HuggingFace Transformers | `github.com/huggingface/transformers` | All model implementations (Llama, Qwen, Mistral, Phi, Gemma) |
| llama.cpp | `github.com/ggml-org/llama.cpp` | C++ inference engine, GGUF format, Python bindings |
| llama-cpp-python | `github.com/abetlen/llama-cpp-python` | Python bindings for llama.cpp |
| Qwen3 Official | `github.com/QwenLM/Qwen3` | Qwen3 models, docs, examples |
| vLLM | `github.com/vllm-project/vllm` | High-throughput inference engine |
| SGLang | `github.com/sgl-project/sglang` | Fast serving framework |
| GGUF JS Parser | `npm:@huggingface/gguf` | JavaScript GGUF parser |
| GGUF Py Package | `gguf-py` in llama.cpp repo | Python GGUF reader/writer |
| Mixtral | `github.com/mistralai/mistral-inference` | Official Mistral inference |
| DeepSpeed | `github.com/microsoft/DeepSpeed` | Microsoft inference optimization |

### Model Architecture Files in Transformers

| Model | File Path |
|-------|-----------|
| Llama | `src/transformers/models/llama/modeling_llama.py` |
| Qwen3 | `src/transformers/models/qwen3/modeling_qwen3.py` |
| Mistral | `src/transformers/models/mistral/modeling_mistral.py` |
| Mixtral | `src/transformers/models/mixtral/modeling_mixtral.py` |
| Phi3/4 | `src/transformers/models/phi3/modeling_phi3.py` |
| Gemma | `src/transformers/models/gemma/modeling_gemma.py` |
| Gemma2 | `src/transformers/models/gemma2/modeling_gemma2.py` |
| Cohere | `src/transformers/models/cohere/modeling_cohere.py` |
| Falcon | `src/transformers/models/falcon/modeling_falcon.py` |
| GPT2 | `src/transformers/models/gpt2/modeling_gpt2.py` |
| GPT-NeoX | `src/transformers/models/gpt_neox/modeling_gpt_neox.py` |
| MPT | `src/transformers/models/mpt/modeling_mpt.py` |
| Baichuan | `src/transformers/models/baichuan/modeling_baichuan.py` |

---

## Quick Reference: Model Architecture Comparison

| Feature | Llama | Mistral | Qwen3 | Mixtral |
|---------|-------|---------|-------|---------|
| Attention | Full | Sliding Window | Alternating Full/Sliding | Sliding Window |
| Q/K Norm | No | No | Yes (RMSNorm) | No |
| MLP | SwiGLU | SwiGLU | SwiGLU | **MoE (8 experts)** |
| Activation | SiLU | SiLU | SiLU | SiLU |
| RoPE | Yes | Yes | Yes | Yes |
| GQA | Yes | Yes | Yes | Yes |
| Bias in Attention | Configurable | No | Configurable | No |
| Bias in MLP | Configurable | No | No | No |
| Context | 4K-128K | 32K | 262K | 32K |

---

## How to Build Your Own Model Loader

Based on all the research above, here's a skeleton for building a program that loads both GGUF and HuggingFace models:

```python
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional

class UnifiedModelLoader:
    """Loads both GGUF and HuggingFace format models"""
    
    def __init__(self, model_path: str, format: str = "auto"):
        self.model_path = model_path
        self.format = self._detect_format() if format == "auto" else format
        self.config = {}
        self.tensors = {}
    
    def _detect_format(self):
        path = Path(self.model_path)
        if path.suffix == ".gguf":
            return "gguf"
        elif (path / "config.json").exists():
            return "huggingface"
        elif path.suffix in [".bin", ".safetensors"]:
            return "huggingface"
        raise ValueError(f"Cannot detect format for {self.model_path}")
    
    def load(self):
        if self.format == "gguf":
            return self._load_gguf()
        elif self.format == "huggingface":
            return self._load_huggingface()
    
    def _load_gguf(self):
        """Load GGUF file using gguf library"""
        import gguf
        reader = gguf.GGUFReader(self.model_path)
        
        # Read config from metadata
        arch = reader.fields.get("general.architecture", "unknown")
        self.config = {
            "architecture": arch,
            "hidden_size": reader.fields.get(f"{arch}.embedding_length", 4096),
            "num_hidden_layers": reader.fields.get(f"{arch}.block_count", 32),
            "num_attention_heads": reader.fields.get(f"{arch}.attention.head_count", 32),
            "num_key_value_heads": reader.fields.get(f"{arch}.attention.head_count_kv", 32),
            "intermediate_size": reader.fields.get(f"{arch}.feed_forward_length", 11008),
            "rms_norm_eps": reader.fields.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5),
            "vocab_size": len(reader.fields.get("tokenizer.ggml.tokens", [])),
            "rope_theta": reader.fields.get(f"{arch}.rope.freq_base", 10000.0),
        }
        
        # Build tensor dictionary
        for tensor in reader.tensors:
            self.tensors[tensor.name] = tensor.data
        
        return self.config, self.tensors
    
    def _load_huggingface(self):
        """Load HuggingFace format (bin/safetensors + config.json)"""
        import json
        from safetensors.torch import load_file
        
        path = Path(self.model_path)
        
        # Load config
        config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Load tensors
        if (path / "model.safetensors").exists():
            self.tensors = load_file(path / "model.safetensors")
        elif (path / "pytorch_model.bin").exists():
            self.tensors = torch.load(path / "pytorch_model.bin", map_location="cpu")
        else:
            # Handle sharded models
            import glob
            files = sorted(glob.glob(str(path / "*.safetensors")))
            for f in files:
                self.tensors.update(load_file(f))
        
        return self.config, self.tensors
    
    def build_model(self) -> nn.Module:
        """Build a PyTorch model from loaded config and tensors"""
        arch = self.config.get("architecture", self.config.get("model_type", "llama"))
        
        if arch in ["llama", "qwen3"]:
            return self._build_llama_like()
        elif arch == "mixtral":
            return self._build_mixtral()
        else:
            raise NotImplementedError(f"Architecture {arch} not yet supported")
    
    def _build_llama_like(self):
        """Build Llama/Qwen3 model"""
        from transformers import LlamaConfig, LlamaForCausalLM
        
        config = LlamaConfig(
            hidden_size=self.config["hidden_size"],
            num_hidden_layers=self.config["num_hidden_layers"],
            num_attention_heads=self.config["num_attention_heads"],
            num_key_value_heads=self.config.get("num_key_value_heads", self.config["num_attention_heads"]),
            intermediate_size=self.config["intermediate_size"],
            rms_norm_eps=self.config["rms_norm_eps"],
            vocab_size=self.config["vocab_size"],
            rope_theta=self.config.get("rope_theta", 10000.0),
        )
        
        model = LlamaForCausalLM(config)
        
        # Map tensors to model state dict
        # (requires tensor name mapping based on source format)
        state_dict = self._map_tensor_names(self.tensors)
        model.load_state_dict(state_dict, strict=False)
        
        return model
    
    def _map_tensor_names(self, tensors: Dict) -> Dict:
        """Map tensor names from source format to model format"""
        # GGUF names: "blk.0.attn_q.weight" -> "model.layers.0.self_attn.q_proj.weight"
        # This mapping depends on the source architecture
        mapped = {}
        for name, tensor in tensors.items():
            new_name = name  # Add actual mapping logic here
            mapped[new_name] = tensor
        return mapped
```

---

*This reference guide was compiled from the following open-source projects:*
- HuggingFace Transformers (github.com/huggingface/transformers)
- llama.cpp (github.com/ggml-org/llama.cpp)
- Qwen3 (github.com/QwenLM/Qwen3)
- vLLM (github.com/vllm-project/vllm)
- SGLang (github.com/sgl-project/sglang)
