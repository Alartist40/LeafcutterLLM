//! Multi-head attention implementation

use crate::model::tensor::Tensor;
use crate::cache::KVCache;

pub struct AttentionParams {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
}

pub fn apply_rotary_emb(x: &mut Tensor, seq_len: usize, num_heads: usize, head_dim: usize, theta: f32) {
    for i in 0..seq_len {
        for h in 0..num_heads {
            for d in 0..head_dim / 2 {
                let freq = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
                let angle = i as f32 * freq;
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

/// Grouped-query attention forward pass
pub fn attention_forward(
    hidden_states: &Tensor,
    weights: &std::collections::HashMap<String, Tensor>,
    params: &AttentionParams,
    kv_cache: &mut KVCache,
    layer_idx: usize,
) -> Tensor {
    let seq_len = hidden_states.shape[0];

    let q_proj = weights.get("self_attn.q_proj.weight").expect("Missing q_proj");
    let k_proj = weights.get("self_attn.k_proj.weight").expect("Missing k_proj");
    let v_proj = weights.get("self_attn.v_proj.weight").expect("Missing v_proj");
    let o_proj = weights.get("self_attn.o_proj.weight").expect("Missing o_proj");

    // Project to Q, K, V
    let q = hidden_states.matmul(q_proj);
    let k = hidden_states.matmul(k_proj);
    let v = hidden_states.matmul(v_proj);

    // Reshape to [seq, heads, head_dim]
    let mut q = q.reshape(vec![seq_len, params.num_heads, params.head_dim]);
    let mut k = k.reshape(vec![seq_len, params.num_kv_heads, params.head_dim]);
    let v = v.reshape(vec![seq_len, params.num_kv_heads, params.head_dim]);

    // Apply RoPE to Q and K
    apply_rotary_emb(&mut q, seq_len, params.num_heads, params.head_dim, params.rope_theta);
    apply_rotary_emb(&mut k, seq_len, params.num_kv_heads, params.head_dim, params.rope_theta);

    // Update KV cache (stores as f16 for 2× RAM savings)
    kv_cache.append(layer_idx, k.clone(), v.clone());
    let (k_cached, v_cached) = kv_cache.get(layer_idx).unwrap();

    let total_seq_len = k_cached.shape[0];
    let num_kv_groups = params.num_heads / params.num_kv_heads;

    let mut attn_output = vec![0.0f32; seq_len * params.num_heads * params.head_dim];

    for h in 0..params.num_heads {
        let kv_h = h / num_kv_groups;

        for s in 0..seq_len {
            let mut scores = vec![0.0f32; total_seq_len];
            // Compute attention scores with causal masking
            // Position s can only attend to positions 0..=s (and previously cached positions)
            let cache_len = total_seq_len - seq_len; // positions already in cache before this forward
            for t in 0..total_seq_len {
                if t > cache_len + s {
                    // Causal mask: future positions get -inf
                    scores[t] = f32::NEG_INFINITY;
                } else {
                    let mut dot = 0.0f32;
                    for d in 0..params.head_dim {
                        let q_val = q.data[s * params.num_heads * params.head_dim + h * params.head_dim + d];
                        let k_val = k_cached.data[t * params.num_kv_heads * params.head_dim + kv_h * params.head_dim + d];
                        dot += q_val * k_val;
                    }
                    scores[t] = dot / (params.head_dim as f32).sqrt();
                }
            }

            // Softmax (masked positions are -inf, so exp(-inf) = 0)
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
            for t in 0..total_seq_len {
                scores[t] = (scores[t] - max_score).exp() / exp_sum;
            }

            // Apply to V
            for d in 0..params.head_dim {
                let mut sum = 0.0f32;
                for t in 0..total_seq_len {
                    let v_val = v_cached.data[t * params.num_kv_heads * params.head_dim + kv_h * params.head_dim + d];
                    sum += scores[t] * v_val;
                }
                attn_output[s * params.num_heads * params.head_dim + h * params.head_dim + d] = sum;
            }
        }
    }

    let attn_tensor = Tensor::from_vec(attn_output, vec![seq_len, params.num_heads * params.head_dim]);
    attn_tensor.matmul(o_proj)
}
