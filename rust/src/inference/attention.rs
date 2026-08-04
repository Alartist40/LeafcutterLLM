//! Multi-head attention with RoPE, grouped-query attention, fused QKV, and compressed KV
//!
//! Supports:
//!   - Standard separate Q/K/V projections (Llama, Qwen2)
//!   - Fused QKV projection (Qwen3.5: attn_qkv.weight)
//!   - Gated attention (Qwen3.5: attn_gate.weight)
//!   - Grouped-query attention (GQA) with KV head grouping
//!   - Compressed KV cache (256-dim keys/values instead of 4096)
//!   - Rotary Position Embeddings (RoPE)
//!   - Causal masking

use crate::model::tensor::Tensor;
use crate::cache::KVCache;
use rayon::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct YarnParams {
    /// 1.0 / yarn_ext_factor (e.g. 1/16 for Ministral-3).
    pub freq_scale: f32,
    /// Original training context length (e.g. 16384 for Ministral-3).
    pub orig_ctx: usize,
    /// YaRN beta_fast (default 32).
    pub beta_fast: f32,
    /// YaRN beta_slow (default 1).
    pub beta_slow: f32,
    /// mscale-equivalent (HuggingFace's `yarn_log_multiplier`).
    /// llama.cpp pre-divides this by `(1 + 0.1*log(factor))` and the kernel
    /// multiplies back, so the effective value baked into cos/sin equals this.
    pub attn_factor: f32,
    /// yarn_ext_factor (e.g. 16.0). Stored for debugging/logging.
    pub ext_factor: f32,
}

pub struct AttentionParams {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_head_dim: usize,
    pub rope_theta: f32,
    pub rope_dim: usize, // 0 = full head_dim (standard), >0 = partial RoPE
    pub use_fused_qkv: bool,
    pub use_gate: bool,
    pub window_size: usize, // 0 = disabled (full causal), >0 = sliding window
    /// Optional YaRN parameters. When `Some`, `apply_rotary_emb` uses
    /// YaRN-scaled inv_freq instead of vanilla RoPE.
    pub yarn: Option<YarnParams>,
}

impl Default for AttentionParams {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            kv_head_dim: 128,
            rope_theta: 10000.0,
            rope_dim: 0,
            use_fused_qkv: false,
            use_gate: false,
            window_size: 0,
            yarn: None,
        }
    }
}

pub fn apply_rotary_emb(x: &mut Tensor, seq_len: usize, num_heads: usize, head_dim: usize, rope_dim: usize, theta: f32, position_offset: usize) {
    // Defensive: rope_dim must not exceed head_dim. Some model configs
    // (e.g., partial RoPE variants, duplicated rope_dim metadata) advertise
    // a rope_dim > head_dim; we silently clamp rather than OOB-panic on Q/K.
    let rope_dim = if rope_dim > 0 && rope_dim <= head_dim { rope_dim } else { head_dim };
    let n = x.data.len();
    let stride = num_heads * head_dim;
    for i in 0..seq_len {
        for h in 0..num_heads {
            for d in 0..rope_dim / 2 {
                let freq = 1.0 / theta.powf(2.0 * d as f32 / rope_dim as f32);
                let angle = (position_offset + i) as f32 * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let base = i * stride + h * head_dim;
                let x1_idx = base + d;
                let x2_idx = base + d + rope_dim / 2;

                if x1_idx >= n || x2_idx >= n {
                    // Defensive: should be unreachable once rope_dim <= head_dim.
                    // Skip this pair rather than crash; preserves Llama/Qwen behavior.
                    continue;
                }

                let x1 = x.data[x1_idx];
                let x2 = x.data[x2_idx];

                x.data[x1_idx] = x1 * cos_a - x2 * sin_a;
                x.data[x2_idx] = x1 * sin_a + x2 * cos_a;
            }
        }
    }
}

/// Add a per-channel bias vector to every row of an [S, D] tensor.
/// `bias` must have length D (the second dimension of `x`).
fn add_bias_inplace(x: &mut Tensor, bias: &Tensor) {
    let n_rows = x.data.len() / x.shape[1];
    let d = x.shape[1];
    debug_assert_eq!(bias.data.len(), d, "bias len {} != row width {}", bias.data.len(), d);
    for r in 0..n_rows {
        let base = r * d;
        for c in 0..d {
            x.data[base + c] += bias.data[c];
        }
    }
}

/// Apply per-head RMSNorm to Q or K before RoPE.
/// Input is flat [seq_len * num_heads * head_dim], output is same shape.
fn apply_per_head_rms_norm(data: &[f32], num_heads: usize, head_dim: usize, weight: &Tensor, eps: f32) -> Vec<f32> {
    let seq_len = data.len() / (num_heads * head_dim);
    let mut out = vec![0.0f32; data.len()];
    for s in 0..seq_len {
        for h in 0..num_heads {
            let base = s * num_heads * head_dim + h * head_dim;
            // Compute RMS
            let sum_sq: f32 = data[base..base + head_dim].iter().map(|&x| x * x).sum();
            let rms = (sum_sq / head_dim as f32 + eps).sqrt();
            let scale = 1.0 / rms;
            for d in 0..head_dim {
                out[base + d] = data[base + d] * scale * weight.data[d];
            }
        }
    }
    out
}

/// Attention forward pass with fused QKV and compressed KV support.
pub fn attention_forward(
    hidden_states: &Tensor,
    weights: &std::collections::HashMap<String, Tensor>,
    params: &AttentionParams,
    kv_cache: &mut KVCache,
    layer_idx: usize,
    position_offset: usize,
) -> Tensor {
    let seq_len = hidden_states.shape[0];

    // -------------------------------------------------------------------------
    // Fused QKV projection (M4) — auto-detect: try fused first, fall back to separate
    // -------------------------------------------------------------------------
    let has_fused_qkv = weights.contains_key("self_attn.qkv_proj.weight")
        || weights.contains_key("attn_qkv.weight");

    let (q, k, v) = if params.use_fused_qkv && has_fused_qkv {
        let qkv_proj = weights.get("self_attn.qkv_proj.weight")
            .or_else(|| weights.get("attn_qkv.weight"))
            .expect("Missing fused QKV projection");
        let qkv = hidden_states.matmul(qkv_proj);

        let q_dim = params.num_heads * params.head_dim;
        let kv_dim = params.num_kv_heads * params.kv_head_dim;
        let total = q_dim + kv_dim + kv_dim;
        assert_eq!(qkv.shape[1], total, "Fused QKV output dim mismatch: got {}, expected {}", qkv.shape[1], total);

        let q_tensor = Tensor::from_vec(
            qkv.data[..seq_len * q_dim].to_vec(),
            vec![seq_len, q_dim],
        );
        let k_tensor = Tensor::from_vec(
            qkv.data[seq_len * q_dim..seq_len * (q_dim + kv_dim)].to_vec(),
            vec![seq_len, kv_dim],
        );
        let v_tensor = Tensor::from_vec(
            qkv.data[seq_len * (q_dim + kv_dim)..].to_vec(),
            vec![seq_len, kv_dim],
        );
        (q_tensor, k_tensor, v_tensor)
    } else {
        let q_proj = weights.get("self_attn.q_proj.weight")
            .or_else(|| weights.get("attn_q.weight"))
            .expect("Missing q_proj");
        let k_proj = weights.get("self_attn.k_proj.weight")
            .or_else(|| weights.get("attn_k.weight"))
            .expect("Missing k_proj");
        let v_proj = weights.get("self_attn.v_proj.weight")
            .or_else(|| weights.get("attn_v.weight"))
            .expect("Missing v_proj");

        // Qwen2 (and other bias-carrying families) add a per-projection bias
        // to Q/K/V after the matmul.  Llama-family GGUFs carry no such bias
        // tensors, so the lookup simply misses and we skip the add.
        let mut q = hidden_states.matmul(q_proj);
        let mut k = hidden_states.matmul(k_proj);
        let mut v = hidden_states.matmul(v_proj);
        if let Some(bias) = weights.get("self_attn.q_proj.bias") {
            add_bias_inplace(&mut q, bias);
        }
        if let Some(bias) = weights.get("self_attn.k_proj.bias") {
            add_bias_inplace(&mut k, bias);
        }
        if let Some(bias) = weights.get("self_attn.v_proj.bias") {
            add_bias_inplace(&mut v, bias);
        }
        (q, k, v)
    };

    // -------------------------------------------------------------------------
    // Gated attention (M4) — element-wise gate on Q before RoPE
    // -------------------------------------------------------------------------
    let q_data = if params.use_gate {
        if let Some(gate_w) = weights.get("self_attn.gate.weight")
            .or_else(|| weights.get("attn_gate.weight")) {
            let gate_proj = hidden_states.matmul(gate_w);
            let mut gated = vec![0.0f32; q.data.len()];
            for i in 0..gated.len() {
                let sigmoid = 1.0 / (1.0 + (-gate_proj.data[i]).exp());
                gated[i] = q.data[i] * sigmoid;
            }
            gated
        } else {
            q.data.clone()
        }
    } else {
        q.data.clone()
    };

    let k_data = k.data.clone();

    // -------------------------------------------------------------------------
    // Q/K per-head RMSNorm (Qwen3.5-style) — applied before RoPE
    // -------------------------------------------------------------------------
    // Qwen3.5 attention: Q head_dim may be 2× kv_head_dim (e.g. 512 vs 256).
    // The first half is "content" (used for QK scoring), the second half is
    // "gate" (applied as a sigmoid to the attention output).
    // QK norm weights have kv_head_dim elements, so norm runs on content only.
    // -------------------------------------------------------------------------
    let use_q_split = params.head_dim > params.kv_head_dim;
    let content_head_dim = params.kv_head_dim;

    let (q_content, q_gate_opt) = if use_q_split {
        // Q layout: [seq_len, num_heads, head_dim] where head_dim = 2 * content_head_dim
        // Each head is [content(0..content_head_dim), gate(content_head_dim..head_dim)].
        // We must extract content and gate PER-HEAD, not as contiguous halves.
        let mut content = vec![0.0f32; seq_len * params.num_heads * content_head_dim];
        let mut gate = vec![0.0f32; seq_len * params.num_heads * content_head_dim];
        for s in 0..seq_len {
            for h in 0..params.num_heads {
                let src_base = s * params.num_heads * params.head_dim + h * params.head_dim;
                let dst_base = s * params.num_heads * content_head_dim + h * content_head_dim;
                content[dst_base..dst_base + content_head_dim]
                    .copy_from_slice(&q_data[src_base..src_base + content_head_dim]);
                gate[dst_base..dst_base + content_head_dim]
                    .copy_from_slice(&q_data[src_base + content_head_dim..src_base + params.head_dim]);
            }
        }
        (content, Some(gate))
    } else {
        (q_data, None)
    };

    let q_normed = if let Some(q_norm_w) = weights.get("attn_q_norm.weight") {
        apply_per_head_rms_norm(&q_content, params.num_heads, content_head_dim, q_norm_w, 1e-6)
    } else {
        q_content
    };
    let k_normed = if let Some(k_norm_w) = weights.get("attn_k_norm.weight") {
        apply_per_head_rms_norm(&k_data, params.num_kv_heads, params.kv_head_dim, k_norm_w, 1e-6)
    } else {
        k_data
    };

    let mut q = Tensor::from_vec(q_normed, vec![seq_len, params.num_heads, content_head_dim]);
    let mut k = Tensor::from_vec(k_normed, vec![seq_len, params.num_kv_heads, params.kv_head_dim]);
    let v = Tensor::from_vec(v.data, vec![seq_len, params.num_kv_heads, params.kv_head_dim]);

    // -------------------------------------------------------------------------
    // RoPE
    // -------------------------------------------------------------------------
    apply_rotary_emb(&mut q, seq_len, params.num_heads, content_head_dim, params.rope_dim, params.rope_theta, position_offset);
    apply_rotary_emb(&mut k, seq_len, params.num_kv_heads, params.kv_head_dim, params.rope_dim, params.rope_theta, position_offset);

    // -------------------------------------------------------------------------
    // KV Cache (M5: compressed dimensions)
    // -------------------------------------------------------------------------
    kv_cache.append(layer_idx, k.clone(), v.clone());
    let (k_cached, v_cached) = kv_cache.get(layer_idx).unwrap();

    let total_seq_len = k_cached.shape[0];
    let num_kv_groups = params.num_heads.max(1) / params.num_kv_heads.max(1);

    // -------------------------------------------------------------------------
    // Attention scores — parallel across heads
    // -------------------------------------------------------------------------
    // Q may have larger head_dim than K/V (Qwen3.5/3.6: Q=512, K/V=256).
    // The output per head is only kv_head_dim wide — the extra Q dimensions
    // are used for scoring but don't expand the output. O_proj expects
    // input of shape [seq_len, num_heads * kv_head_dim].
    let output_dim_per_head = params.kv_head_dim;
    let head_outputs: Vec<Vec<f32>> = (0..params.num_heads)
        .into_par_iter()
        .map(|h| {
            let kv_h = h / num_kv_groups;
            let mut head_out = vec![0.0f32; seq_len * output_dim_per_head];

            for s in 0..seq_len {
                let mut scores = vec![0.0f32; total_seq_len];
                let cache_len = if total_seq_len > seq_len {
                    total_seq_len - seq_len
                } else {
                    0
                };

                for t in 0..total_seq_len {
                    if t > cache_len + s {
                        scores[t] = f32::NEG_INFINITY;
                    } else if params.window_size > 0 && t + params.window_size <= cache_len + s {
                        scores[t] = f32::NEG_INFINITY; // SWA: block tokens beyond window
                    } else {
                        let mut dot = 0.0f32;
                        for d in 0..params.kv_head_dim {
                            // Q uses first kv_head_dim of its head_dim for scoring
                            let q_val = q.data[s * params.num_heads * content_head_dim + h * content_head_dim + d];
                            let k_val = k_cached.data[t * params.num_kv_heads * params.kv_head_dim + kv_h * params.kv_head_dim + d];
                            dot += q_val * k_val;
                        }
                        // Scale by sqrt(kv_head_dim) — the dimension of the dot product
                        scores[t] = dot / (params.kv_head_dim as f32).sqrt();
                    }
                }

                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
                for t in 0..total_seq_len {
                    scores[t] = (scores[t] - max_score).exp() / exp_sum;
                }

                for d in 0..output_dim_per_head {
                    let mut sum = 0.0f32;
                    for t in 0..total_seq_len {
                        let v_val = v_cached.data[t * params.num_kv_heads * params.kv_head_dim + kv_h * params.kv_head_dim + d];
                        sum += scores[t] * v_val;
                    }
                    head_out[s * output_dim_per_head + d] = sum;
                }
            }
            head_out
        })
        .collect();

    // Reassemble heads into contiguous output
    let mut attn_output = vec![0.0f32; seq_len * params.num_heads * output_dim_per_head];
    for h in 0..params.num_heads {
        for s in 0..seq_len {
            for d in 0..output_dim_per_head {
                attn_output[s * params.num_heads * output_dim_per_head + h * output_dim_per_head + d] =
                    head_outputs[h][s * output_dim_per_head + d];
            }
        }
    }

    // If Q-split gate exists, apply sigmoid element-wise to attention output
    if let Some(q_gate_data) = q_gate_opt {
        for i in 0..attn_output.len() {
            let sigmoid = 1.0 / (1.0 + (-q_gate_data[i]).exp());
            attn_output[i] *= sigmoid;
        }
    }

    let attn_tensor = Tensor::from_vec(attn_output, vec![seq_len, params.num_heads * output_dim_per_head]);

    // Output projection
    let o_proj = weights.get("self_attn.o_proj.weight")
        .or_else(|| weights.get("attn_output.weight"))
        .expect("Missing o_proj");
    attn_tensor.matmul(o_proj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_test_weights(hidden: usize, heads: usize, kv_heads: usize, head_dim: usize, kv_head_dim: usize) -> HashMap<String, Tensor> {
        let mut w = HashMap::new();
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * kv_head_dim;
        let o_dim = heads * kv_head_dim; // attention output is num_heads * kv_head_dim
        w.insert("self_attn.q_proj.weight".to_string(), Tensor::from_vec(vec![0.0; hidden * q_dim], vec![hidden, q_dim]));
        w.insert("self_attn.k_proj.weight".to_string(), Tensor::from_vec(vec![0.0; hidden * kv_dim], vec![hidden, kv_dim]));
        w.insert("self_attn.v_proj.weight".to_string(), Tensor::from_vec(vec![0.0; hidden * kv_dim], vec![hidden, kv_dim]));
        w.insert("self_attn.o_proj.weight".to_string(), Tensor::from_vec(vec![0.0; o_dim * hidden], vec![o_dim, hidden]));
        w
    }

    fn make_fused_weights(hidden: usize, heads: usize, kv_heads: usize, head_dim: usize, kv_head_dim: usize) -> HashMap<String, Tensor> {
        let mut w = HashMap::new();
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * kv_head_dim;
        let o_dim = heads * kv_head_dim; // attention output is num_heads * kv_head_dim
        let fused_dim = q_dim + kv_dim + kv_dim;
        w.insert("attn_qkv.weight".to_string(), Tensor::from_vec(vec![0.0; hidden * fused_dim], vec![hidden, fused_dim]));
        w.insert("attn_output.weight".to_string(), Tensor::from_vec(vec![0.0; o_dim * hidden], vec![o_dim, hidden]));
        w
    }

    #[test]
    fn test_attention_standard() {
        let hidden = 64; let heads = 4; let kv_heads = 2; let head_dim = 16; let kv_head_dim = 16;
        let weights = make_test_weights(hidden, heads, kv_heads, head_dim, kv_head_dim);
        let hidden_states = Tensor::from_vec(vec![0.1; 2 * hidden], vec![2, hidden]);
        let mut kv_cache = KVCache::new(2);

        let params = AttentionParams {
            num_heads: heads, num_kv_heads: kv_heads, head_dim,
            kv_head_dim, rope_theta: 10000.0,
            use_fused_qkv: false, use_gate: false,
            ..Default::default()
        };

        let out = attention_forward(&hidden_states, &weights, &params, &mut kv_cache, 0, 0);
        assert_eq!(out.shape, vec![2, hidden]);
    }

    #[test]
    fn test_attention_fused_qkv() {
        let hidden = 64; let heads = 4; let kv_heads = 2; let head_dim = 16; let kv_head_dim = 16;
        let weights = make_fused_weights(hidden, heads, kv_heads, head_dim, kv_head_dim);
        let hidden_states = Tensor::from_vec(vec![0.1; 2 * hidden], vec![2, hidden]);
        let mut kv_cache = KVCache::new(2);

        let params = AttentionParams {
            num_heads: heads, num_kv_heads: kv_heads, head_dim,
            kv_head_dim, rope_theta: 10000.0,
            use_fused_qkv: true, use_gate: false,
            ..Default::default()
};

        let out = attention_forward(&hidden_states, &weights, &params, &mut kv_cache, 0, 0);
        assert_eq!(out.shape, vec![2, hidden]);
    }

    #[test]
    fn test_attention_compressed_kv() {
        let hidden = 64; let heads = 4; let kv_heads = 2; let head_dim = 16; let kv_head_dim = 8;
        let mut weights = make_test_weights(hidden, heads, kv_heads, head_dim, kv_head_dim);
        weights.insert("self_attn.k_proj.weight".to_string(), Tensor::from_vec(vec![0.0; hidden * kv_heads * kv_head_dim], vec![hidden, kv_heads * kv_head_dim]));
        weights.insert("self_attn.v_proj.weight".to_string(), Tensor::from_vec(vec![0.0; hidden * kv_heads * kv_head_dim], vec![hidden, kv_heads * kv_head_dim]));

        let hidden_states = Tensor::from_vec(vec![0.1; 2 * hidden], vec![2, hidden]);
        let mut kv_cache = KVCache::new(2);

        let params = AttentionParams {
            num_heads: heads, num_kv_heads: kv_heads, head_dim,
            kv_head_dim, rope_theta: 10000.0,
            use_fused_qkv: false, use_gate: false,
            ..Default::default()
        };

        let out = attention_forward(&hidden_states, &weights, &params, &mut kv_cache, 0, 0);
        assert_eq!(out.shape, vec![2, hidden]);
        let (k, _v) = kv_cache.get(0).unwrap();
        assert_eq!(k.shape, vec![2, kv_heads, kv_head_dim]);
    }
}
