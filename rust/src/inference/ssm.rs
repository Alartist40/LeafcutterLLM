//! State Space Model (Mamba) layer — Qwen3.5 adaptive implementation
//!
//! Qwen3.5 uses a non-standard SSM tensor layout:
//!   - attn_qkv.weight   [hidden, 3*hidden] → fused in-projection
//!   - attn_gate.weight  [hidden, hidden]   → gating / output
//!   - ssm_a             [state_size]       → state matrix A
//!   - ssm_alpha.weight  [hidden, state_size] → C projection
//!   - ssm_beta.weight   [hidden, state_size] → B projection
//!   - ssm_conv1d.weight [kernel, 3*hidden]  → causal conv on qkv output
//!   - ssm_dt.bias       [state_size]        → delta bias
//!   - ssm_norm.weight   [group_size]        → group norm
//!   - ssm_out.weight    [hidden, hidden]    → final output projection

use crate::model::tensor::Tensor;
use std::collections::HashMap;

pub struct SSMConfig {
    pub state_size: usize,
    pub inner_size: usize,
    pub time_step_rank: usize,
    pub conv_kernel: usize,
    pub group_count: usize,
}

impl Default for SSMConfig {
    fn default() -> Self {
        Self {
            state_size: 128,
            inner_size: 4096,
            time_step_rank: 32,
            conv_kernel: 4,
            group_count: 16,
        }
    }
}

pub fn ssm_forward(
    hidden_states: &Tensor,
    weights: &HashMap<String, Tensor>,
    _config: &SSMConfig,
    ssm_cache: &mut crate::cache::ssm_state::SSMStateCache,
    layer_idx: usize,
) -> Tensor {
    let seq_len = hidden_states.shape[0];
    let hidden_size = hidden_states.shape[1];

    // 1. In-projection: hidden → [X, z_gate]
    let x_inner = if let Some(in_proj) = weights.get("ssm_in_proj.weight")
        .or_else(|| weights.get("self_attn.qkv_proj.weight"))
        .or_else(|| weights.get("attn_qkv.weight")) {
        hidden_states.matmul(in_proj)
    } else {
        hidden_states.clone()
    };
    let inner_total = x_inner.shape[1];

    // 2. Causal conv1d on FULL in-proj output (conv weight has 8192 channels)
    let x_conv = if let Some(conv_w) = weights.get("ssm_conv1d.weight") {
        let conv_state = ssm_cache.get_conv(layer_idx);
        let kernel = conv_w.shape[0];
        let (out, new_state) = causal_conv1d_cached(&x_inner, conv_w, kernel, &conv_state);
        ssm_cache.set_conv(layer_idx, new_state);
        out
    } else {
        x_inner.clone()
    };


    // 3. Split conv output into X (first half) and z_gate (second half)
    let half = hidden_size;
    let mut x_data = vec![0.0f32; seq_len * half];
    let mut z_data = vec![0.0f32; seq_len * half];
    for s in 0..seq_len {
        let base = s * inner_total;
        x_data[s * half..(s + 1) * half].copy_from_slice(&x_conv.data[base..base + half]);
        z_data[s * half..(s + 1) * half].copy_from_slice(&x_conv.data[base + half..base + inner_total]);
    }
    let mut x = Tensor::from_vec(x_data, vec![seq_len, half]);
    let z_gate = Tensor::from_vec(z_data, vec![seq_len, half]);

    // 4. SiLU on X (the selective-scan input, not z_gate)
    for i in 0..x.data.len() {
        let v = x.data[i];
        x.data[i] = v * (1.0 / (1.0 + (-v).exp()));
    }

    // 5. B and C projections from x (conv output + SiLU) — Mamba standard
    let state_size = _config.state_size;
    let b_raw = if let Some(b_w) = weights.get("ssm_beta.weight")
        .or_else(|| weights.get("ssm_B.weight")) {
        x.matmul(b_w)
    } else {
        Tensor::zeros(vec![seq_len, state_size])
    };
    let c_raw = if let Some(c_w) = weights.get("ssm_alpha.weight")
        .or_else(|| weights.get("ssm_C.weight")) {
        x.matmul(c_w)
    } else {
        Tensor::zeros(vec![seq_len, state_size])
    };

    // 6. Delta (dt) projection + bias
    let dt_raw = if let Some(dt_w) = weights.get("ssm_dt.weight") {
        let dt_proj = hidden_states.matmul(dt_w);
        if let Some(dt_b) = weights.get("ssm_dt.bias") {
            let mut dt = dt_proj;
            for s in 0..seq_len {
                for i in 0..state_size {
                    dt.data[s * state_size + i] += dt_b.data[i % dt_b.data.len()];
                }
            }
            dt
        } else {
            dt_proj
        }
    } else if let Some(dt_b) = weights.get("ssm_dt.bias") {
        let mut dt_data = vec![0.0f32; seq_len * state_size];
        for s in 0..seq_len {
            for i in 0..state_size {
                dt_data[s * state_size + i] = dt_b.data[i % dt_b.data.len()];
            }
        }
        Tensor::from_vec(dt_data, vec![seq_len, state_size])
    } else {
        Tensor::zeros(vec![seq_len, state_size])
    };

    // 7. Broadcast B, C, delta to match X channels (half = hidden_size)
    let conv_channels = x.shape[1];
    let b_proj = broadcast_to_dim(b_raw, conv_channels);
    let c_proj = broadcast_to_dim(c_raw, conv_channels);
    let dt_proj = broadcast_to_dim(dt_raw, conv_channels);

    // 8. A_log → A discretization (Mamba convention)
    let a_vec = weights.get("ssm_a")
        .map(|t| t.data.iter().map(|&a_log| -a_log.exp()).collect::<Vec<f32>>())
        .unwrap_or_else(|| vec![-1.0f32; state_size]);

    // 9. Selective scan on x (4096 channels), not the full conv output
    let state_len = conv_channels * state_size;
    let initial_state = ssm_cache.get(layer_idx, state_len);
    let _ssm_t0 = std::time::Instant::now();
    let (y, final_state) = selective_scan(
        &x, &b_proj, &c_proj, &dt_proj, &a_vec,
        conv_channels, Some(&initial_state),
    );
    let _ssm_elapsed = _ssm_t0.elapsed();
    if std::env::var("LEAFCUTTER_PROFILE").is_ok() {
        eprintln!("[PROFILE] ssm_selective_scan               {:>8.2}ms",
            _ssm_elapsed.as_secs_f32() * 1000.0);
    }
    ssm_cache.set(layer_idx, final_state);

    // 10. Group norm (ssm_norm.weight = [state_size/num_groups])
    let mut y_normed = y.clone();
    if let Some(norm_w) = weights.get("ssm_norm.weight") {
        let num_groups = norm_w.data.len();
        apply_group_norm(&mut y_normed, seq_len, conv_channels, num_groups, norm_w, 1e-5);
    }

    // 11. Gating: y * silu(z_gate)
    let activated = z_gate.silu();
    let mut gated = vec![0.0f32; y_normed.data.len()];
    for i in 0..gated.len() {
        gated[i] = y_normed.data[i] * activated.data[i % activated.data.len()];
    }

    let y_gated = Tensor::from_vec(gated, y_normed.shape.clone());

    // 12. Output projection
    let output = if let Some(out_proj) = weights.get("ssm_out.weight")
        .or_else(|| weights.get("ssm_out_proj.weight")) {
        y_gated.matmul(out_proj)
    } else {
        adaptive_fallback(y_gated, hidden_size)
    };

    // 13. Output gate (Gated DeltaNet): o = ssm_out * sigmoid(hidden @ attn_gate)
    if let Some(gate_w) = weights.get("self_attn.gate_proj.weight") {
        let gate_logits = hidden_states.matmul(gate_w);
        let mut gated = vec![0.0f32; output.data.len()];
        for i in 0..gated.len() {
            let sigmoid = 1.0f32 / (1.0f32 + (-gate_logits.data[i]).exp());
            gated[i] = output.data[i] * sigmoid;
        }
        Tensor::from_vec(gated, output.shape.clone())
    } else {
        output
    }
}

/// Broadcast a tensor's last dimension to target_dim by repetition.
fn broadcast_to_dim(t: Tensor, target_dim: usize) -> Tensor {
    if t.shape[1] == target_dim {
        return t;
    }
    let seq_len = t.shape[0];
    let src_dim = t.shape[1];
    let mut out = vec![0.0f32; seq_len * target_dim];
    for i in 0..seq_len {
        for j in 0..target_dim {
            out[i * target_dim + j] = t.data[i * src_dim + (j % src_dim)];
        }
    }
    Tensor::from_vec(out, vec![seq_len, target_dim])
}

/// Group norm: normalize channels per-group.
fn apply_group_norm(x: &mut Tensor, seq_len: usize, channels: usize, num_groups: usize, weight: &Tensor, eps: f32) {
    let channels_per_group = channels / num_groups;
    for s in 0..seq_len {
        for g in 0..num_groups {
            let base = s * channels + g * channels_per_group;
            let mut sq_sum = 0.0f32;
            for c in 0..channels_per_group {
                let v = x.data[base + c];
                sq_sum += v * v;
            }
            let rms = (sq_sum / channels_per_group as f32 + eps).sqrt();
            let w = weight.data[g % weight.data.len()];
            for c in 0..channels_per_group {
                x.data[base + c] = (x.data[base + c] / rms) * w;
            }
        }
    }
}

/// Adaptive matmul when inner dims don't match.
/// Maps source channels to target channels via simple pooling.
fn adaptive_matmul(x: &Tensor, w: &Tensor, target_hidden: usize) -> Tensor {
    let seq_len = x.shape[0];
    let src_dim = x.shape[1];
    let w_in = w.shape[0];
    let _w_out = w.shape[1];

    // If src_dim > w_in: pool src down to w_in
    // If src_dim < w_in: pad src up to w_in
    let mut pooled = vec![0.0f32; seq_len * w_in];
    for i in 0..seq_len {
        for j in 0..w_in {
            if src_dim == w_in {
                pooled[i * w_in + j] = x.data[i * src_dim + j];
            } else if src_dim > w_in {
                // Average pool
                let start = j * src_dim / w_in;
                let end = ((j + 1) * src_dim / w_in).max(start + 1);
                let sum: f32 = x.data[i * src_dim + start..i * src_dim + end].iter().sum();
                pooled[i * w_in + j] = sum / (end - start) as f32;
            } else {
                // Repeat
                pooled[i * w_in + j] = x.data[i * src_dim + (j % src_dim)];
            }
        }
    }
    let pooled_t = Tensor::from_vec(pooled, vec![seq_len, w_in]);
    let out = pooled_t.matmul(w);

    // Ensure output matches target_hidden
    if out.shape[1] == target_hidden {
        out
    } else {
        adaptive_fallback(out, target_hidden)
    }
}

fn adaptive_fallback(t: Tensor, target_hidden: usize) -> Tensor {
    let seq_len = t.shape[0];
    let src_dim = t.shape[1];
    if src_dim == target_hidden {
        return t;
    }
    let mut out = vec![0.0f32; seq_len * target_hidden];
    for i in 0..seq_len {
        for j in 0..target_hidden {
            out[i * target_hidden + j] = t.data[i * src_dim + (j % src_dim)];
        }
    }
    Tensor::from_vec(out, vec![seq_len, target_hidden])
}

fn causal_conv1d(x: &Tensor, weight: &Tensor, kernel_size: usize) -> Tensor {
    let seq_len = x.shape[0];
    let channels = x.shape[1];
    let mut out = vec![0.0f32; seq_len * channels];

    for c in 0..channels {
        for t in 0..seq_len {
            let mut sum = 0.0f32;
            for k in 0..kernel_size.min(t + 1) {
                let x_val = x.data[(t - k) * channels + c];
                let w_val = if weight.shape.len() == 1 {
                    weight.data[k]
                } else {
                    weight.data[k * weight.shape[1] + c]
                };
                sum += x_val * w_val;
            }
            out[t * channels + c] = sum;
        }
    }
    Tensor::from_vec(out, x.shape.clone())
}

/// Causal conv1d with state caching for autoregressive generation.
/// `conv_state` holds the last (kernel_size - 1) inputs per channel.
/// Returns (output, updated_conv_state).
pub fn causal_conv1d_cached(
    x: &Tensor,
    weight: &Tensor,
    kernel_size: usize,
    conv_state: &[f32],
) -> (Tensor, Vec<f32>) {
    let seq_len = x.shape[0];
    let channels = x.shape[1];
    let state_len = conv_state.len() / channels; // number of cached steps per channel

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

pub fn selective_scan(
    x: &Tensor,
    b: &Tensor,
    c: &Tensor,
    delta: &Tensor,
    a_vec: &[f32],
    inner: usize,
    initial_state: Option<&[f32]>,
) -> (Tensor, Vec<f32>) {
    let seq_len = x.shape[0];
    let state_size = a_vec.len();
    let mut y = vec![0.0f32; seq_len * inner];
    let state_len = inner * state_size;
    let mut h = initial_state
        .map(|s| s.to_vec())
        .unwrap_or_else(|| vec![0.0f32; state_len]);

    for t in 0..seq_len {
        for i in 0..inner {
            let dt_raw = delta.data[t * delta.shape[1] + i];
            let dt = softplus(dt_raw).max(0.001);
            let x_t = x.data[t * inner + i];
            let mut y_ti = 0.0f32;

            for d in 0..state_size {
                let a_d = a_vec[d];
                let b_t = b.data[t * b.shape[1] + (d % b.shape[1])];
                let c_t = c.data[t * c.shape[1] + (d % c.shape[1])];

                let decay = (dt * a_d).exp();
                let b_bar = dt * b_t;

                let idx = i * state_size + d;
                h[idx] = decay * h[idx] + b_bar * x_t;
                y_ti += c_t * h[idx];
            }
            y[t * inner + i] = y_ti;
        }
    }

    (Tensor::from_vec(y, vec![seq_len, inner]), h)
}

#[inline]
fn softplus(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_conv1d() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let w = Tensor::from_vec(vec![1.0, 0.5], vec![2, 1]);
        let out = causal_conv1d(&x, &w, 2);
        assert_eq!(out.data, vec![1.0, 2.5, 4.0]);
    }

    #[test]
    fn test_adaptive_matmul() {
        let x = Tensor::from_vec(vec![1.0; 2 * 6144], vec![2, 6144]);
        let w = Tensor::from_vec(vec![0.5; 2048 * 2048], vec![2048, 2048]);
        let out = adaptive_matmul(&x, &w, 2048);
        assert_eq!(out.shape, vec![2, 2048]);
    }

    #[test]
    fn test_ssm_forward_qwen_shapes() {
        let mut weights = HashMap::new();
        let hidden = 16;
        let inner = 32; // 2 * hidden — conv output split into x and z_gate

        weights.insert("attn_qkv.weight".to_string(), Tensor::from_vec(vec![0.01; hidden * inner], vec![hidden, inner]));
        weights.insert("ssm_out.weight".to_string(), Tensor::from_vec(vec![0.01; hidden * hidden], vec![hidden, hidden]));
        weights.insert("ssm_conv1d.weight".to_string(), Tensor::from_vec(vec![1.0; 4 * inner], vec![4, inner]));
        weights.insert("ssm_beta.weight".to_string(), Tensor::from_vec(vec![0.5; hidden * 4], vec![hidden, 4]));
        weights.insert("ssm_alpha.weight".to_string(), Tensor::from_vec(vec![1.0; hidden * 4], vec![hidden, 4]));
        weights.insert("ssm_dt.bias".to_string(), Tensor::from_vec(vec![0.1; 4], vec![4]));
        weights.insert("ssm_a".to_string(), Tensor::from_vec(vec![-1.0; 4], vec![4]));

        let hidden_states = Tensor::from_vec(vec![0.1; 2 * hidden], vec![2, hidden]);
        let config = SSMConfig { state_size: 4, inner_size: inner, time_step_rank: 4, conv_kernel: 4, group_count: 1 };

        let mut cache = crate::cache::ssm_state::SSMStateCache::new();
        let out = ssm_forward(&hidden_states, &weights, &config, &mut cache, 0);
        assert_eq!(out.shape, vec![2, hidden]);
        assert!(out.data.iter().all(|&v| v.is_finite()));
    }
}
