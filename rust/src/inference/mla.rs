//! Multi-Latent Attention (MLA) — DeepSeek-2 / GLM-DSA
//!
//! Architecture:
//!   1. attn_q_a      : hidden → q_lora_rank    (down)
//!   2. attn_q_a_norm : rms_norm on latent
//!   3. attn_q_b      : q_lora_rank → n_heads * (qk_nope + qk_rope)  (up)
//!   4. attn_kv_a_mqa : hidden → kv_lora_rank + qk_rope_head_dim (down; rope absorbed)
//!   5. attn_kv_a_norm: rms_norm on latent (length kv_lora_rank)
//!   6. attn_k_b      : kv_lora_rank → kv_heads * qk_nope          (up)
//!   7. attn_v_b      : kv_lora_rank → kv_heads * v_head_dim       (up)
//!   8. RoPE only on qk_rope portion of Q and the absorbed-rope portion of K.
//!   9. Standard scaled dot-product attention with causal mask.
//!  10. attn_output (o_proj) : n_heads * v_head_dim → hidden
//!
//! KV-cache strategy: store the *compressed* latent (kv_lora_rank + qk_rope_head_dim)
//! per token rather than the per-head K/V.  This is the memory win MLA exists for.
//! Reconstruction to per-head K/V happens on the read path of the next call.
//! (Caller is responsible for swapping in `MlaKVCache` rather than the existing
//! `KVCache`.  The Engine wiring is done by switching which cache is used for
//! MLA layers.)

use crate::cache::KVCache;
use crate::model::tensor::Tensor;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MlaParams {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
}

impl Default for MlaParams {
    fn default() -> Self {
        // Kimi K2.6 defaults, smaller for testing.
        Self {
            num_heads: 64,
            num_kv_heads: 1,
            qk_nope_head_dim: 192,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            rope_theta: 50000.0,
            norm_eps: 1e-5,
        }
    }
}

/// MLA forward.  `hidden_states` shape: `[seq_len, hidden]`.  Returns
/// `[seq_len, hidden]`.  KV cache stores the *latent* forms, not the per-head
/// K/V tensors, so the on-disk working set is bounded.
pub fn mla_forward(
    hidden_states: &Tensor,
    weights: &HashMap<String, Tensor>,
    params: &MlaParams,
    kv_cache: &mut KVCache,
    layer_idx: usize,
    position_offset: usize,
) -> Tensor {
    let seq_len = hidden_states.shape[0];
    let hidden_size = hidden_states.shape[1];

    // — Step 1: Q down → norm → up.
    let q_a = weights
        .get("attn_q_a.weight")
        .expect("MLA requires attn_q_a.weight");
    let q_a_norm = weights
        .get("attn_q_a_norm.weight")
        .expect("MLA requires attn_q_a_norm.weight");
    let q_b = weights
        .get("attn_q_b.weight")
        .expect("MLA requires attn_q_b.weight");

    // q_lat = rms_norm(hidden @ q_a.T, q_a_norm)
    let h_q_a = hidden_states.matmul(&q_a.transpose()); // [seq_len, q_lora_rank]
    let q_lat = h_q_a.rms_norm(q_a_norm, params.norm_eps); // [seq_len, q_lora_rank]
    let q_full = q_lat.matmul(&q_b.transpose()); // [seq_len, num_heads * (qk_nope + qk_rope)]

    let q_total_dim = params.qk_nope_head_dim + params.qk_rope_head_dim;
    let q_total = params.num_heads * q_total_dim;
    assert_eq!(
        q_full.shape[1], q_total,
        "MLA Q up-projection output dim {} != expected {} (num_heads * (qk_nope + qk_rope))",
        q_full.shape[1], q_total
    );

    // — Step 2: KV compressed projection (kv_lat || k_rope_absorbed).
    let kv_a = weights
        .get("attn_kv_a_mqa.weight")
        .expect("MLA requires attn_kv_a_mqa.weight");
    let kv_a_norm = weights
        .get("attn_kv_a_norm.weight")
        .expect("MLA requires attn_kv_a_norm.weight");

    let kv_flat = hidden_states.matmul(&kv_a.transpose());
    let kv_total = params.kv_lora_rank + params.qk_rope_head_dim;
    assert_eq!(
        kv_flat.shape[1], kv_total,
        "MLA KV down output dim {} != expected {} (kv_lora_rank + qk_rope_head_dim)",
        kv_flat.shape[1], kv_total
    );

    let kv_lora_with_rope: Vec<f32> = kv_flat.data.clone(); // [seq_len, kv_total]

    // — Step 3: split into kv_lat and absorbed-rope chunk, then norm kv_lat.
    let kv_lat = {
        let mut out = Vec::with_capacity(seq_len * params.kv_lora_rank);
        for s in 0..seq_len {
            out.extend_from_slice(
                &kv_flat.data[s * kv_total..s * kv_total + params.kv_lora_rank],
            );
        }
        Tensor::from_vec(out, vec![seq_len, params.kv_lora_rank])
    };
    let k_rope_raw = {
        let mut out = Vec::with_capacity(seq_len * params.qk_rope_head_dim);
        for s in 0..seq_len {
            out.extend_from_slice(
                &kv_flat.data[s * kv_total + params.kv_lora_rank..(s + 1) * kv_total],
            );
        }
        Tensor::from_vec(out, vec![seq_len, params.qk_rope_head_dim])
    };
    let kv_lat_norm = kv_lat.rms_norm(kv_a_norm, params.norm_eps);

    // — Append the *latent* kv to the cache.  We store as one flattend
    //   `[seq_len, kv_lora_rank + qk_rope_head_dim]` row to mirror standard
    //   KVCache's flat per-layer storage shape contract.
    let latent_shape = vec![seq_len, kv_total];
    {
        // We pass dummy K and V (just the latent reused) so KVCache stores it
        // without invasive changes — `KVCache` accepts any two same-shaped
        // tensors.  On read we'll re-decode through k_b/v_b again.
        let stored = Tensor::from_vec(kv_lora_with_rope.clone(), latent_shape.clone());
        let _dummy = Tensor::zeros(vec![seq_len, kv_total]); // same shape, value unused
        kv_cache.append(layer_idx, stored, _dummy);
    }

    // — Step 4: Reconstruct Q per head split into nope / rope.
    // q_full flat layout is [seq_len, num_heads, q_total_dim]
    let mut q_nope = vec![0.0f32; seq_len * params.num_heads * params.qk_nope_head_dim];
    let mut q_rope = vec![0.0f32; seq_len * params.num_heads * params.qk_rope_head_dim];
    for s in 0..seq_len {
        for h in 0..params.num_heads {
            for d in 0..params.qk_nope_head_dim {
                q_nope[s * params.num_heads * params.qk_nope_head_dim
                      + h * params.qk_nope_head_dim
                      + d] =
                    q_full.data[s * q_total + h * q_total_dim + d];
            }
            for d in 0..params.qk_rope_head_dim {
                q_rope[s * params.num_heads * params.qk_rope_head_dim
                      + h * params.qk_rope_head_dim
                      + d] =
                    q_full.data[s * q_total + h * q_total_dim + params.qk_nope_head_dim + d];
            }
        }
    }

    // — Step 5: Apply RoPE to q_rope (and to the absorbed k_rope as we walk it).
    // RoPE dim is qk_rope_head_dim.  Same convention as standard attention:
    // pair (x[0..rope/2], x[rope/2..rope]) via outer-half rotations.
    apply_rotary_3d(
        &mut q_rope,
        seq_len,
        params.num_heads,
        params.qk_rope_head_dim,
        params.rope_theta,
        position_offset,
    );

    // — Step 6: Per-token Reconstruct K (nope + rope) and V on demand.
    // Reconstruct on the fly from the cached latent + position-wise k_b/v_b.
    let k_b = weights
        .get("attn_k_b.weight")
        .expect("MLA requires attn_k_b.weight");
    let v_b = weights
        .get("attn_v_b.weight")
        .expect("MLA requires attn_v_b.weight");

    // Get full cached K latent sequence for this layer.
    let (k_cached, v_unused) = match kv_cache.get(layer_idx) {
        Some(t) => t,
        None => (
            Tensor::from_vec(vec![0.0f32; position_offset * kv_total], vec![position_offset, kv_total]),
            Tensor::zeros(vec![position_offset, kv_total]),
        ),
    };
    let cache_seq_len = k_cached.shape[0];
    let _ = v_unused; // we don't actually need v in this fallback — we reconstruct below
    let total_seq_len = cache_seq_len;
    assert_eq!(
        k_cached.shape[1], kv_total,
        "Cached K latent dim {} != expected {}",
        k_cached.shape[1], kv_total
    );

    // Per-head scale (MQA if num_kv_heads == 1).
    let head_group = params.num_heads / params.num_kv_heads;
    assert_eq!(
        params.num_heads % params.num_kv_heads,
        0,
        "num_heads must be a multiple of num_kv_heads"
    );

    // KV cache: total_seq_len * num_kv_heads * (qk_nope_head_dim + qk_rope_head_dim + v_head_dim)
    // We avoid a full materialization of K/V by iterating per head during attention,
    // see step 7.  Pre-cache the rope-vector threshold for the indexer.

    // — Step 7: Standard scaled dot-product attention per head (single-token
    //   one-token autoregressive path; batched multi-token is full prefill).
    //
    // Each head h maps to a kv_head index kv_h = h / head_group.  For each
    // kv_head we do one matmul of the cached `kv_lat` against the rows of k_b/v_b
    // corresponding to that head, building per-head K and V on the fly.
    //
    // For speed in this first implementation we materialise the per-head K and V
    // tensors once per forward per layer; that is still bounded by cache_seq_len
    // and is the cost a "non-MLA-MQA" engine pays too.
    let k_per_head_shape = vec![total_seq_len, params.qk_nope_head_dim + params.qk_rope_head_dim];
    let v_per_head_shape = vec![total_seq_len, params.v_head_dim];
    let mut per_head_k = Vec::with_capacity(params.num_kv_heads);
    let mut per_head_v = Vec::with_capacity(params.num_kv_heads);
    for kv_h in 0..params.num_kv_heads {
        // Slice one k_b row and one v_b row out.
        let k_row_start = kv_h * params.qk_nope_head_dim;
        let v_row_start = kv_h * params.v_head_dim;
        let k_row_len = params.qk_nope_head_dim;
        let v_row_len = params.v_head_dim;
        let k_b_row = &k_b.data[k_row_start * params.kv_lora_rank
                               ..(k_row_start + 1) * params.kv_lora_rank];
        let v_b_row = &v_b.data[v_row_start * params.kv_lora_rank
                               ..(v_row_start + 1) * params.kv_lora_rank];
        let k_b_t = Tensor::from_vec(k_b_row.to_vec(), vec![k_row_len, params.kv_lora_rank]);
        let v_b_t = Tensor::from_vec(v_b_row.to_vec(), vec![v_row_len, params.kv_lora_rank]);
        // k_cached_nope = (kv_lora_normed @ k_b_t.T) [cache_seq_len, qk_nope_head_dim]
        let k_for_head = kv_lat_norm.matmul(&k_b_t.transpose()); // [cache_seq_len, qk_nope_head_dim]
        // Concatenate the absorbed-rope chunk (already in latent) and the learned K.
        let mut k_full_head = vec![0.0f32; cache_seq_len * k_per_head_shape[1]];
        for s in 0..cache_seq_len {
            // qk_nope block
            for d in 0..params.qk_nope_head_dim {
                k_full_head[s * k_per_head_shape[1] + d] =
                    k_for_head.data[s * params.qk_nope_head_dim + d];
            }
            // absorbed-rope block (k_rope_raw[s, :])
            for d in 0..params.qk_rope_head_dim {
                k_full_head[s * k_per_head_shape[1] + params.qk_nope_head_dim + d] =
                    k_cached.data[s * kv_total + params.kv_lora_rank + d];
            }
        }
        // Apply RoPE to the rope portion of the cached K.  Position offset
        // depends on whether this layer is being hit during prefill (offset=0)
        // or single-token decode (offset set by caller).
        let mut k_full_head_rope = k_full_head; // moved
        // Reshape and rotate-in-place only the rope half of each row.
        for s in 0..cache_seq_len {
            for d in 0..params.qk_rope_head_dim / 2 {
                let freq = 1.0 / params.rope_theta.powf(2.0 * d as f32 / params.qk_rope_head_dim as f32);
                let pos = position_offset as f32 + 0.0; // prefill path; decode uses (position_offset + s)
                // For prefill (seq_len > 1, position_offset=0), RoPE position is s.
                let pos = if seq_len > 1 { s as f32 } else { position_offset as f32 + s as f32 };
                let angle = pos * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let base = s * k_per_head_shape[1] + params.qk_nope_head_dim;
                let x1 = k_full_head_rope[base + d];
                let x2 = k_full_head_rope[base + d + params.qk_rope_head_dim / 2];
                k_full_head_rope[base + d] = x1 * cos_a - x2 * sin_a;
                k_full_head_rope[base + d + params.qk_rope_head_dim / 2] = x1 * sin_a + x2 * cos_a;
            }
        }
        per_head_k.push(Tensor::from_vec(k_full_head_rope, k_per_head_shape.clone()));

        // V[t, :] = (kv_lat_norm @ v_b_t.T)[t, :]
        let v_for_head = kv_lat_norm.matmul(&v_b_t.transpose()); // [cache_seq_len, v_head_dim]
        per_head_v.push((v_for_head, v_per_head_shape.clone()));
    }

    // Reorder per-head_v to Vec<Tensor> for uniform access; tighten below.
    let per_head_v_tensors: Vec<Tensor> = per_head_v.into_iter().map(|(t, _s)| t).collect();

    // — Step 8: Per-head attention scoring.
    let output_dim_per_head = params.v_head_dim;
    let total_q_dim = params.qk_nope_head_dim + params.qk_rope_head_dim;
    let scale = 1.0 / (total_q_dim as f32).sqrt();

    let mut head_outputs: Vec<Vec<f32>> = (0..params.num_heads)
        .map(|h| {
            let kv_h = h / head_group;
            let k_per_head = per_head_k[kv_h].data.clone();
            let v_per_head = per_head_v_tensors[kv_h].data.clone();
            let mut head_out = vec![0.0f32; seq_len * output_dim_per_head];

            for s in 0..seq_len {
                let q_off_nope = s * params.num_heads * params.qk_nope_head_dim + h * params.qk_nope_head_dim;
                let q_off_rope = s * params.num_heads * params.qk_rope_head_dim + h * params.qk_rope_head_dim;

                let cache_len = total_seq_len - seq_len; // prefill path ⇒ 0

                let mut scores = vec![0.0f32; total_seq_len];
                for t in 0..total_seq_len {
                    if t > cache_len + s {
                        scores[t] = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0f32;
                        // nope
                        for d in 0..params.qk_nope_head_dim {
                            dot += q_nope[q_off_nope + d] * k_per_head[t * total_q_dim + d];
                        }
                        // rope
                        for d in 0..params.qk_rope_head_dim {
                            dot += q_rope[q_off_rope + d]
                                * k_per_head[t * total_q_dim + params.qk_nope_head_dim + d];
                        }
                        scores[t] = dot * scale;
                    }
                }

                // softmax
                let max_score = scores
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&x| (x - max_score).exp()).sum();
                for d in 0..output_dim_per_head {
                    let mut sum = 0.0f32;
                    for t in 0..total_seq_len {
                        let p = ((scores[t] - max_score).exp()) / exp_sum;
                        sum += p * v_per_head[t * output_dim_per_head + d];
                    }
                    head_out[s * output_dim_per_head + d] = sum;
                }
            }
            head_out
        })
        .collect();

    // — Step 9: Reassemble heads, output projection.
    let mut attn_output_flat = vec![0.0f32; seq_len * params.num_heads * output_dim_per_head];
    for h in 0..params.num_heads {
        for s in 0..seq_len {
            for d in 0..output_dim_per_head {
                attn_output_flat[s * params.num_heads * output_dim_per_head
                                 + h * output_dim_per_head
                                 + d] = head_outputs[h][s * output_dim_per_head + d];
            }
        }
    }
    // Replace head_outputs index 0..num_heads vec data with the reassembled buffer in-place.
    head_outputs.clear();
    head_outputs.push(attn_output_flat);

    let attn_tensor = Tensor::from_vec(
        head_outputs.remove(0),
        vec![seq_len, params.num_heads * output_dim_per_head],
    );

    let attn_output_w = weights
        .get("attn_output.weight")
        .expect("MLA requires attn_output.weight");
    attn_tensor.matmul(attn_output_w)
}

/// Apply RoPE in-place over a `[seq_len, num_heads, rope_dim]` flat tensor.
fn apply_rotary_3d(
    data: &mut Vec<f32>,
    seq_len: usize,
    num_heads: usize,
    rope_dim: usize,
    theta: f32,
    position_offset: usize,
) {
    let half = rope_dim / 2;
    for s in 0..seq_len {
        for h in 0..num_heads {
            let base = s * num_heads * rope_dim + h * rope_dim;
            for d in 0..half {
                let freq = 1.0 / theta.powf(2.0 * d as f32 / rope_dim as f32);
                let pos = (position_offset + s) as f32;
                let angle = pos * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x1 = data[base + d];
                let x2 = data[base + d + half];
                data[base + d] = x1 * cos_a - x2 * sin_a;
                data[base + d + half] = x1 * sin_a + x2 * cos_a;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_is_sensible() {
        let cfg = MlaParams::default();
        assert_eq!(cfg.num_heads, 64);
        assert_eq!(cfg.num_kv_heads, 1);
        assert_eq!(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim, 256);
        assert_eq!(cfg.q_lora_rank, 1536);
        assert_eq!(cfg.kv_lora_rank, 512);
    }

    #[test]
    fn num_heads_is_multiple_of_num_kv_heads() {
        let cfg = MlaParams::default();
        assert_eq!(cfg.num_heads % cfg.num_kv_heads, 0);
    }

    #[test]
    fn tensor_api_used_inside_mla() {
        // Smoke-test that Tensor has the methods we rely on this layer.
        let a = Tensor::zeros(vec![2, 3]);
        let b = Tensor::zeros(vec![3, 4]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 4]);

        let w = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let n = Tensor::zeros(vec![2, 3]).rms_norm(&w, 1e-5);
        assert_eq!(n.shape, vec![2, 3]);
    }
}
