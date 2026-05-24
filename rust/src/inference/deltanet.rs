//! Gated DeltaNet — Qwen3.5/3.6 linear attention
//!
//! Architecture (from llama.cpp source analysis):
//!   1. attn_qkv projects hidden → [Q, K, V] concatenated
//!   2. Causal Conv1d + SiLU on the full [Q, K, V]
//!   3. Extract Q, K, V from conv output, L2-normalize Q and K
//!   4. decay = exp( softplus(alpha + dt_bias) * ssm_a ) — per-head decay in (0,1)
//!   5. beta = sigmoid(hidden @ ssm_beta) — per-head update gate
//!   6. State: matrix [head_v_dim, head_k_dim] per head
//!   7. Core delta rule: S_t = decay_t * S_{t-1} + beta_t * (v_t ⊗ k_t)
//!   8. Output: o_t = S_t @ q_t  followed by ssm_out projection

use crate::model::tensor::Tensor;
use crate::cache::deltanet_state::DeltaNetStateCache;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DeltaNetParams {
    pub num_qk_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_dim: usize,
    pub conv_kernel: usize,
    pub state_size: usize,
}

impl Default for DeltaNetParams {
    fn default() -> Self {
        Self {
            num_qk_heads: 16,
            num_v_heads: 16,
            head_k_dim: 128,
            head_v_dim: 128,
            conv_dim: 8192,
            conv_kernel: 4,
            state_size: 128,
        }
    }
}

pub fn deltanet_forward(
    hidden_states: &Tensor,
    weights: &HashMap<String, Tensor>,
    params: &DeltaNetParams,
    state_cache: &mut DeltaNetStateCache,
    layer_idx: usize,
) -> Tensor {
    let seq_len = hidden_states.shape[0];
    let hidden_size = hidden_states.shape[1];

    // 1. attn_qkv projection: hidden → [Q, K, V]
    let qkv_weight = weights.get("self_attn.qkv_proj.weight")
        .or_else(|| weights.get("attn_qkv.weight"))
        .expect("DeltaNet requires attn_qkv.weight");
    let qkv_proj = hidden_states.matmul(qkv_weight);
    assert_eq!(
        qkv_proj.shape[1], params.conv_dim,
        "attn_qkv output dim {} != expected conv_dim {}",
        qkv_proj.shape[1], params.conv_dim
    );

    // 2. Causal Conv1d + SiLU on the full projection
    let conv_out = if let Some(conv_w) = weights.get("ssm_conv1d.weight") {
        let conv_state = state_cache.get_conv(layer_idx);
        let kernel = conv_w.shape[0];
        let (mut out, new_state) = causal_conv1d_cached(&qkv_proj, conv_w, kernel, &conv_state);
        state_cache.set_conv(layer_idx, new_state);
        // SiLU: x * sigmoid(x)
        for i in 0..out.data.len() {
            let x = out.data[i];
            out.data[i] = x * (1.0 / (1.0 + (-x).exp()));
        }
        out
    } else {
        qkv_proj.clone()
    };

    // 3. Split conv output into Q, K, V
    let q_total = params.num_qk_heads * params.head_k_dim;
    let k_total = params.num_qk_heads * params.head_k_dim;
    let v_total = params.num_v_heads * params.head_v_dim;

    let mut q_data = vec![0.0f32; seq_len * q_total];
    let mut k_data = vec![0.0f32; seq_len * k_total];
    let mut v_data = vec![0.0f32; seq_len * v_total];

    for s in 0..seq_len {
        let base = s * params.conv_dim;
        q_data[s * q_total..(s + 1) * q_total]
            .copy_from_slice(&conv_out.data[base..base + q_total]);
        k_data[s * k_total..(s + 1) * k_total]
            .copy_from_slice(&conv_out.data[base + q_total..base + q_total + k_total]);
        v_data[s * v_total..(s + 1) * v_total]
            .copy_from_slice(&conv_out.data[base + q_total + k_total..base + params.conv_dim]);
    }

    // 4. L2-normalize Q and K (per-head)
    let mut q = Tensor::from_vec(q_data, vec![seq_len, params.num_qk_heads, params.head_k_dim]);
    let mut k = Tensor::from_vec(k_data, vec![seq_len, params.num_qk_heads, params.head_k_dim]);
    let v = Tensor::from_vec(v_data, vec![seq_len, params.num_v_heads, params.head_v_dim]);

    l2_normalize_per_head(&mut q, seq_len, params.num_qk_heads, params.head_k_dim);
    l2_normalize_per_head(&mut k, seq_len, params.num_qk_heads, params.head_k_dim);

    // 5. Per-head decay rates: decay = exp( softplus(alpha + dt_bias) * ssm_a )
    let decay = compute_decay_rates(hidden_states, weights, params, seq_len);
    // decay shape: [seq_len, num_qk_heads]

    // 6. Beta gating: beta = sigmoid(hidden @ ssm_beta)
    let beta = compute_beta_gates(hidden_states, weights, params, seq_len);
    // beta shape: [seq_len, num_qk_heads]



    // 7. Delta rule state update + output
    let output_dim_per_token = params.num_v_heads * params.head_v_dim;
    let mut output = vec![0.0f32; seq_len * output_dim_per_token];

    if state_cache.get(layer_idx).is_none() {
        state_cache.init_layer(layer_idx, params.num_v_heads, params.head_v_dim, params.head_k_dim);
    }
    let state = state_cache.get_mut(layer_idx).unwrap();

    // Map QK heads to V heads.  Most Qwen3.5 variants have num_v_heads == num_qk_heads
    // or num_v_heads is a multiple of num_qk_heads.
    let v_heads_per_qk = if params.num_qk_heads > 0 {
        params.num_v_heads / params.num_qk_heads
    } else {
        1
    };

    for s in 0..seq_len {
        for h_qk in 0..params.num_qk_heads {
            let q_base = s * params.num_qk_heads * params.head_k_dim + h_qk * params.head_k_dim;
            let k_base = s * params.num_qk_heads * params.head_k_dim + h_qk * params.head_k_dim;
            let q_h = &q.data[q_base..q_base + params.head_k_dim];
            let k_h = &k.data[k_base..k_base + params.head_k_dim];

            for v_idx in 0..v_heads_per_qk.max(1) {
                let h_v = h_qk * v_heads_per_qk + v_idx;
                if h_v >= params.num_v_heads {
                    continue;
                }
                // Decay and beta are per-V-head (ssm_alpha/beta/a/dt have num_v_heads outputs)
                let decay_h = decay[s * params.num_v_heads + h_v];
                let beta_h = beta[s * params.num_v_heads + h_v];

                let state_stride = h_v * params.head_v_dim * params.head_k_dim;
                let v_base = s * params.num_v_heads * params.head_v_dim + h_v * params.head_v_dim;
                let v_h = &v.data[v_base..v_base + params.head_v_dim];

                // DeltaNet delta rule:
                // 1. Predict v from current state: v_pred = S @ k
                let mut v_pred = vec![0.0f32; params.head_v_dim];
                for i in 0..params.head_v_dim {
                    let mut sum = 0.0f32;
                    for j in 0..params.head_k_dim {
                        sum += state[state_stride + i * params.head_k_dim + j] * k_h[j];
                    }
                    v_pred[i] = sum;
                }

                // 2. State update: S = decay * S + beta * ((v - v_pred) outer k)
                let mut max_delta_v = 0.0f32;
                for i in 0..params.head_v_dim {
                    let delta_v = v_h[i] - v_pred[i];
                    max_delta_v = max_delta_v.max(delta_v.abs());
                    for j in 0..params.head_k_dim {
                        let idx = state_stride + i * params.head_k_dim + j;
                        state[idx] = decay_h * state[idx] + beta_h * delta_v * k_h[j];
                    }
                }



                // 3. Output: o = scale * S @ q
                let scale = 1.0f32 / (params.head_k_dim as f32).sqrt();
                let out_base = s * output_dim_per_token + h_v * params.head_v_dim;
                for i in 0..params.head_v_dim {
                    let mut sum = 0.0f32;
                    for j in 0..params.head_k_dim {
                        sum += state[state_stride + i * params.head_k_dim + j] * q_h[j];
                    }
                    output[out_base + i] = scale * sum;
                }
            }
        }
    }

    let mut output_tensor = Tensor::from_vec(output, vec![seq_len, output_dim_per_token]);

    // 8. Group norm — 128 groups over 4096 channels, 32 channels per group
    if let Some(norm_w) = weights.get("ssm_norm.weight") {
        let num_groups = norm_w.data.len();
        apply_group_norm(&mut output_tensor.data, seq_len, output_dim_per_token, num_groups, norm_w, 1e-5);
    }

    // 9. Output projection
    let mut output = if let Some(out_w) = weights.get("ssm_out.weight")
        .or_else(|| weights.get("ssm_out_proj.weight"))
    {
        output_tensor.matmul(out_w)
    } else {
        adaptive_project(&output_tensor, hidden_size)
    };

    output
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn l2_normalize_per_head(x: &mut Tensor, seq_len: usize, num_heads: usize, head_dim: usize) {
    for s in 0..seq_len {
        for h in 0..num_heads {
            let base = s * num_heads * head_dim + h * head_dim;
            let mut norm_sq = 0.0f32;
            for d in 0..head_dim {
                let v = x.data[base + d];
                norm_sq += v * v;
            }
            let norm = norm_sq.sqrt().max(1e-12);
            for d in 0..head_dim {
                x.data[base + d] /= norm;
            }
        }
    }
}

fn compute_decay_rates(
    hidden: &Tensor,
    weights: &HashMap<String, Tensor>,
    params: &DeltaNetParams,
    seq_len: usize,
) -> Vec<f32> {
    let num_heads = params.num_v_heads;

    let alpha_proj = weights.get("ssm_alpha.weight")
        .map(|w| hidden.matmul(w))
        .unwrap_or_else(|| Tensor::zeros(vec![seq_len, num_heads]));

    let dt_bias = weights.get("ssm_dt.bias")
        .map(|t| t.data.clone())
        .unwrap_or_else(|| vec![0.0f32; num_heads]);

    let a_vec = weights.get("ssm_a")
        .map(|t| t.data.clone())
        .unwrap_or_else(|| vec![-1.0f32; num_heads]);

    let mut decay = vec![0.0f32; seq_len * num_heads];
    for s in 0..seq_len {
        for h in 0..num_heads {
            let alpha_val = alpha_proj.data[s * num_heads + h];
            let dt_val = dt_bias.get(h).copied().unwrap_or(0.0);
            let a = a_vals(h, &a_vec);
            // ssm_a is already A = -exp(A_log) from GGUF conversion
            let dt = softplus(alpha_val + dt_val);
            decay[s * num_heads + h] = (dt * a).exp();
        }
    }
    decay
}

fn compute_beta_gates(
    hidden: &Tensor,
    weights: &HashMap<String, Tensor>,
    params: &DeltaNetParams,
    seq_len: usize,
) -> Vec<f32> {
    let num_heads = params.num_v_heads;

    if let Some(beta_w) = weights.get("ssm_beta.weight") {
        let beta_logits = hidden.matmul(beta_w);
        beta_logits.data.iter().map(|&v| sigmoid(v)).collect()
    } else {
        vec![1.0f32; seq_len * num_heads]
    }
}

fn a_vals(h: usize, a_vec: &[f32]) -> f32 {
    if a_vec.len() == 1 {
        a_vec[0]
    } else {
        a_vec.get(h).copied().unwrap_or(-1.0)
    }
}

fn softplus(x: f32) -> f32 {
    (1.0f32 + x.exp()).ln()
}

fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

fn max_abs(data: &[f32]) -> f32 {
    data.iter().map(|&v| v.abs()).fold(0.0f32, f32::max)
}

/// Group norm: normalize channels per-group.
fn apply_group_norm(data: &mut [f32], seq_len: usize, channels: usize, num_groups: usize, weight: &Tensor, eps: f32) {
    let channels_per_group = channels / num_groups;
    for s in 0..seq_len {
        for g in 0..num_groups {
            let base = s * channels + g * channels_per_group;
            let mut sq_sum = 0.0f32;
            for c in 0..channels_per_group {
                let v = data[base + c];
                sq_sum += v * v;
            }
            let rms = (sq_sum / channels_per_group as f32 + eps).sqrt();
            let w = weight.data[g % weight.data.len()];
            for c in 0..channels_per_group {
                data[base + c] = (data[base + c] / rms) * w;
            }
        }
    }
}

/// Causal conv1d with state caching for autoregressive generation.
/// `conv_state` holds the last (kernel_size - 1) inputs per channel.
/// Returns (output, updated_conv_state).
/// Apply per-head RMS norm.
/// Input shape: [seq_len, num_heads * head_dim]
/// norm_weight shape: [head_dim] — same weight for all heads
fn apply_per_head_rms_norm(
    x: &mut Tensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    norm_weight: &Tensor,
    eps: f32,
) {
    for s in 0..seq_len {
        for h in 0..num_heads {
            let base = s * num_heads * head_dim + h * head_dim;
            // Compute RMS
            let mut sq_sum = 0.0f32;
            for d in 0..head_dim {
                let v = x.data[base + d];
                sq_sum += v * v;
            }
            let rms = (sq_sum / head_dim as f32 + eps).sqrt();
            // Apply weight
            for d in 0..head_dim {
                x.data[base + d] = (x.data[base + d] / rms) * norm_weight.data[d % norm_weight.data.len()];
            }
        }
    }
}

fn causal_conv1d_cached(
    x: &Tensor,
    weight: &Tensor,
    kernel_size: usize,
    conv_state: &[f32],
) -> (Tensor, Vec<f32>) {
    let seq_len = x.shape[0];
    let channels = x.shape[1];
    let state_len = if !conv_state.is_empty() && channels > 0 {
        conv_state.len() / channels
    } else {
        0
    };

    // Build full input: [cached_steps, ..., current_steps]
    let full_seq_len = state_len + seq_len;
    let mut full_input = vec![0.0f32; full_seq_len * channels];
    for c in 0..channels {
        for s in 0..state_len {
            full_input[s * channels + c] = conv_state[s * channels + c];
        }
        for t in 0..seq_len {
            full_input[(state_len + t) * channels + c] = x.data[t * channels + c];
        }
    }

    let mut out = vec![0.0f32; seq_len * channels];
    for c in 0..channels {
        for t in 0..seq_len {
            let mut sum = 0.0f32;
            let global_t = state_len + t;
            for k in 0..kernel_size.min(global_t + 1) {
                let x_val = full_input[(global_t - k) * channels + c];
                // PyTorch Conv1d convention: w[k-1] is the coefficient for the CURRENT input.
                // Reverse the kernel index to match exported weights.
                let w_idx = kernel_size - 1 - k;
                let w_val = if weight.shape.len() == 1 {
                    weight.data[w_idx]
                } else {
                    weight.data[w_idx * weight.shape[1] + c]
                };
                sum += x_val * w_val;
            }
            out[t * channels + c] = sum;
        }
    }

    // Update cache: keep last (kernel_size - 1) inputs
    let keep = (kernel_size - 1).min(full_seq_len);
    let mut new_state = vec![0.0f32; keep * channels];
    for c in 0..channels {
        for s in 0..keep {
            new_state[s * channels + c] = full_input[(full_seq_len - keep + s) * channels + c];
        }
    }

    (Tensor::from_vec(out, x.shape.clone()), new_state)
}

fn adaptive_project(x: &Tensor, target: usize) -> Tensor {
    let seq_len = x.shape[0];
    let src = x.shape[1];
    if src == target {
        return x.clone();
    }
    let mut out = vec![0.0f32; seq_len * target];
    for s in 0..seq_len {
        for d in 0..target {
            out[s * target + d] = x.data[s * src + (d % src)];
        }
    }
    Tensor::from_vec(out, vec![seq_len, target])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deltanet_shapes() {
        let mut weights = HashMap::new();
        let hidden = 16;
        let conv_dim = 48; // 2*qk_heads*head_k + v_heads*head_v = 2*2*8 + 2*8 = 48

        weights.insert("attn_qkv.weight".to_string(), Tensor::from_vec(vec![0.01; hidden * conv_dim], vec![hidden, conv_dim]));
        weights.insert("ssm_out.weight".to_string(), Tensor::from_vec(vec![0.01; 16 * hidden], vec![16, hidden]));
        weights.insert("ssm_conv1d.weight".to_string(), Tensor::from_vec(vec![1.0; 4 * conv_dim], vec![4, conv_dim]));
        weights.insert("ssm_beta.weight".to_string(), Tensor::from_vec(vec![0.5; hidden * 2], vec![hidden, 2]));
        weights.insert("ssm_alpha.weight".to_string(), Tensor::from_vec(vec![1.0; hidden * 2], vec![hidden, 2]));
        weights.insert("ssm_dt.bias".to_string(), Tensor::from_vec(vec![0.1; 2], vec![2]));
        weights.insert("ssm_a".to_string(), Tensor::from_vec(vec![-1.0; 2], vec![2]));
        weights.insert("ssm_norm.weight".to_string(), Tensor::from_vec(vec![1.0; 8], vec![8]));

        let hidden_states = Tensor::from_vec(vec![0.1; 2 * hidden], vec![2, hidden]);
        let params = DeltaNetParams {
            num_qk_heads: 2,
            num_v_heads: 2,
            head_k_dim: 8,
            head_v_dim: 8,
            conv_dim,
            conv_kernel: 4,
            state_size: 8,
        };

        let mut cache = DeltaNetStateCache::new();
        let out = deltanet_forward(&hidden_states, &weights, &params, &mut cache, 0);
        assert_eq!(out.shape, vec![2, hidden]);
        assert!(out.data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_l2_normalize() {
        let mut t = Tensor::from_vec(vec![3.0, 4.0, 0.0, 5.0], vec![1, 2, 2]);
        l2_normalize_per_head(&mut t, 1, 2, 2);
        // Head 0: [3, 4] -> norm=5 -> [0.6, 0.8]
        assert!((t.data[0] - 0.6).abs() < 1e-6);
        assert!((t.data[1] - 0.8).abs() < 1e-6);
        // Head 1: [0, 5] -> norm=5 -> [0.0, 1.0]
        assert!((t.data[2] - 0.0).abs() < 1e-6);
        assert!((t.data[3] - 1.0).abs() < 1e-6);
    }
}
