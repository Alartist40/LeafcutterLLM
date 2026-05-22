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

pub struct AttentionParams {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_head_dim: usize,
    pub rope_theta: f32,
    pub use_fused_qkv: bool,
    pub use_gate: bool,
}

impl Default for AttentionParams {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            kv_head_dim: 128,
            rope_theta: 10000.0,
            use_fused_qkv: false,
            use_gate: false,
        }
    }
}

pub fn apply_rotary_emb(x: &mut Tensor, seq_len: usize, num_heads: usize, head_dim: usize, theta: f32, position_offset: usize) {
    for i in 0..seq_len {
        for h in 0..num_heads {
            for d in 0..head_dim / 2 {
                let freq = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
                let angle = (position_offset + i) as f32 * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let base = i * num_heads * head_dim + h * head_dim;
                let x1_idx = base + d;
                let x2_idx = base + d + head_dim / 2;

                let x1 = x.data[x1_idx];
                let x2 = x.data[x2_idx];

                x.data[x1_idx] = x1 * cos_a - x2 * sin_a;
                x.data[x2_idx] = x1 * sin_a + x2 * cos_a;
            }
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
        (
            hidden_states.matmul(q_proj),
            hidden_states.matmul(k_proj),
            hidden_states.matmul(v_proj),
        )
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
    let q_normed = if let Some(q_norm_w) = weights.get("attn_q_norm.weight") {
        apply_per_head_rms_norm(&q_data, params.num_heads, params.head_dim, q_norm_w, 1e-6)
    } else {
        q_data
    };
    let k_normed = if let Some(k_norm_w) = weights.get("attn_k_norm.weight") {
        apply_per_head_rms_norm(&k_data, params.num_kv_heads, params.kv_head_dim, k_norm_w, 1e-6)
    } else {
        k_data
    };

    // -------------------------------------------------------------------------
    // Adaptive reshape to [seq_len, heads, head_dim] for RoPE
    // Qwen3.5 attention layers may have larger Q dims (e.g. 4096 vs expected 2048)
    // -------------------------------------------------------------------------
    let expected_q_dim = params.num_heads * params.head_dim;
    let actual_q_dim = q_normed.len() / seq_len;
    let _effective_q_heads = if actual_q_dim >= expected_q_dim {
        params.num_heads
    } else {
        actual_q_dim / params.head_dim
    };

    // Truncate or pad q_normed to expected_q_dim
    let q_data_trimmed: Vec<f32> = if actual_q_dim == expected_q_dim {
        q_normed
    } else if actual_q_dim > expected_q_dim {
        (0..seq_len).flat_map(|s| q_normed[s * actual_q_dim..s * actual_q_dim + expected_q_dim].to_vec()).collect()
    } else {
        let mut padded = vec![0.0f32; seq_len * expected_q_dim];
        for s in 0..seq_len {
            for d in 0..actual_q_dim {
                padded[s * expected_q_dim + d] = q_normed[s * actual_q_dim + d];
            }
        }
        padded
    };

    let mut q = Tensor::from_vec(q_data_trimmed, vec![seq_len, params.num_heads, params.head_dim]);
    let mut k = Tensor::from_vec(k_normed, vec![seq_len, params.num_kv_heads, params.kv_head_dim]);
    let v = Tensor::from_vec(v.data, vec![seq_len, params.num_kv_heads, params.kv_head_dim]);

    // -------------------------------------------------------------------------
    // RoPE
    // -------------------------------------------------------------------------
    apply_rotary_emb(&mut q, seq_len, params.num_heads, params.head_dim, params.rope_theta, position_offset);
    apply_rotary_emb(&mut k, seq_len, params.num_kv_heads, params.kv_head_dim, params.rope_theta, position_offset);

    // -------------------------------------------------------------------------
    // KV Cache (M5: compressed dimensions)
    // -------------------------------------------------------------------------
    kv_cache.append(layer_idx, k.clone(), v.clone());
    let (k_cached, v_cached) = kv_cache.get(layer_idx).unwrap();

    let total_seq_len = k_cached.shape[0];
    let num_kv_groups = params.num_heads.max(1) / params.num_kv_heads.max(1);

    // -------------------------------------------------------------------------
    // Attention scores
    // -------------------------------------------------------------------------
    let mut attn_output = vec![0.0f32; seq_len * params.num_heads * params.head_dim];

    for h in 0..params.num_heads {
        let kv_h = h / num_kv_groups;
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
                } else {
                    let mut dot = 0.0f32;
                    for d in 0..params.kv_head_dim {
                        let q_val = q.data[s * params.num_heads * params.head_dim + h * params.head_dim + d];
                        let k_val = k_cached.data[t * params.num_kv_heads * params.kv_head_dim + kv_h * params.kv_head_dim + d];
                        dot += q_val * k_val;
                    }
                    scores[t] = dot / (params.head_dim as f32).sqrt();
                }
            }

            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
            for t in 0..total_seq_len {
                scores[t] = (scores[t] - max_score).exp() / exp_sum;
            }

            for d in 0..params.head_dim {
                let mut sum = 0.0f32;
                for t in 0..total_seq_len {
                    let v_val = v_cached.data[t * params.num_kv_heads * params.kv_head_dim + kv_h * params.kv_head_dim + d.min(params.kv_head_dim - 1)];
                    sum += scores[t] * v_val;
                }
                attn_output[s * params.num_heads * params.head_dim + h * params.head_dim + d] = sum;
            }
        }
    }

    let attn_tensor = Tensor::from_vec(attn_output, vec![seq_len, params.num_heads * params.head_dim]);

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
        w.insert("self_attn.q_proj.weight".to_string(), Tensor::from_vec(vec![0.0; hidden * q_dim], vec![hidden, q_dim]));
        w.insert("self_attn.k_proj.weight".to_string(), Tensor::from_vec(vec![0.0; hidden * kv_dim], vec![hidden, kv_dim]));
        w.insert("self_attn.v_proj.weight".to_string(), Tensor::from_vec(vec![0.0; hidden * kv_dim], vec![hidden, kv_dim]));
        w.insert("self_attn.o_proj.weight".to_string(), Tensor::from_vec(vec![0.0; q_dim * hidden], vec![q_dim, hidden]));
        w
    }

    fn make_fused_weights(hidden: usize, heads: usize, kv_heads: usize, head_dim: usize, kv_head_dim: usize) -> HashMap<String, Tensor> {
        let mut w = HashMap::new();
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * kv_head_dim;
        let fused_dim = q_dim + kv_dim + kv_dim;
        w.insert("attn_qkv.weight".to_string(), Tensor::from_vec(vec![0.0; hidden * fused_dim], vec![hidden, fused_dim]));
        w.insert("attn_output.weight".to_string(), Tensor::from_vec(vec![0.0; q_dim * hidden], vec![q_dim, hidden]));
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
