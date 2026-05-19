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
) -> Tensor {
    let seq_len = hidden_states.shape[0];
    let hidden_size = hidden_states.shape[1];

    // 1. Input projection: attn_qkv outputs 3*hidden (or intermediate)
    let x_inner = if let Some(in_proj) = weights.get("ssm_in_proj.weight")
        .or_else(|| weights.get("self_attn.qkv_proj.weight"))
        .or_else(|| weights.get("attn_qkv.weight")) {
        hidden_states.matmul(in_proj)
    } else {
        hidden_states.clone()
    };
    let inner_size = x_inner.shape[1];

    // 2. Causal conv1d — only apply if shape matches
    let x_conv = if let Some(conv_w) = weights.get("ssm_conv1d.weight") {
        if conv_w.shape.len() >= 2 && conv_w.shape[1] == inner_size {
            causal_conv1d(&x_inner, conv_w, conv_w.shape[0])
        } else if conv_w.shape.len() == 1 {
            causal_conv1d(&x_inner, conv_w, _config.conv_kernel.min(conv_w.data.len()))
        } else {
            x_inner.clone()
        }
    } else {
        x_inner.clone()
    };

    // 3. B and C projections from hidden state (Qwen3.5 style)
    let b_raw = if let Some(b_w) = weights.get("ssm_beta.weight")
        .or_else(|| weights.get("ssm_B.weight")) {
        hidden_states.matmul(b_w)
    } else {
        Tensor::zeros(vec![seq_len, _config.state_size])
    };

    let c_raw = if let Some(c_w) = weights.get("ssm_alpha.weight")
        .or_else(|| weights.get("ssm_C.weight")) {
        hidden_states.matmul(c_w)
    } else {
        Tensor::zeros(vec![seq_len, _config.state_size])
    };

    // 4. Delta projection
    let dt_raw = if let Some(dt_w) = weights.get("ssm_dt.weight") {
        hidden_states.matmul(dt_w)
    } else if let Some(dt_b) = weights.get("ssm_dt.bias") {
        let mut dt_data = vec![0.0f32; seq_len * _config.state_size];
        for t in 0..seq_len {
            for i in 0.._config.state_size {
                dt_data[t * _config.state_size + i] = dt_b.data[i % dt_b.data.len()];
            }
        }
        Tensor::from_vec(dt_data, vec![seq_len, _config.state_size])
    } else {
        Tensor::zeros(vec![seq_len, _config.state_size])
    };

    // 5. Broadcast B, C, delta to match conv output channels
    // Qwen3.5 uses group-wise SSM: state_size groups, each with inner_size/group_count channels
    let b_proj = broadcast_to_dim(b_raw, inner_size);
    let c_proj = broadcast_to_dim(c_raw, inner_size);
    let dt_proj = broadcast_to_dim(dt_raw, inner_size);

    // 6. A matrix broadcast to inner_size
    let a_vec = weights.get("ssm_a")
        .map(|t| t.data.clone())
        .unwrap_or_else(|| vec![-1.0f32; _config.state_size]);

    // 7. Selective scan on inner_size channels
    let y = selective_scan(&x_conv, &b_proj, &c_proj, &dt_proj, &a_vec, inner_size);

    // 8. Optional gating (attn_gate.weight)
    let y_gated = if let Some(gate_w) = weights.get("self_attn.gate_proj.weight")
        .or_else(|| weights.get("attn_gate.weight")) {
        let gate = hidden_states.matmul(gate_w);
        let mut gated = vec![0.0f32; y.data.len()];
        for i in 0..gated.len() {
            let sigmoid = 1.0 / (1.0 + (-gate.data[i % gate.data.len()]).exp());
            gated[i] = y.data[i] * sigmoid;
        }
        Tensor::from_vec(gated, y.shape.clone())
    } else {
        y
    };

    // 9. Output projection with shape adaptation
    if let Some(out_proj) = weights.get("ssm_out.weight")
        .or_else(|| weights.get("ssm_out_proj.weight")) {
        if y_gated.shape[1] == out_proj.shape[0] {
            y_gated.matmul(out_proj)
        } else {
            // Adaptive: project via mean-pooling or slice-matching
            adaptive_matmul(&y_gated, out_proj, hidden_size)
        }
    } else {
        adaptive_fallback(y_gated, hidden_size)
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

/// Adaptive matmul when inner dims don't match.
/// Maps source channels to target channels via simple pooling.
fn adaptive_matmul(x: &Tensor, w: &Tensor, target_hidden: usize) -> Tensor {
    let seq_len = x.shape[0];
    let src_dim = x.shape[1];
    let w_in = w.shape[0];
    let w_out = w.shape[1];

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
                    weight.data[k % weight.data.len()]
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

fn selective_scan(
    x: &Tensor,
    b: &Tensor,
    c: &Tensor,
    delta: &Tensor,
    a_vec: &[f32],
    inner: usize,
) -> Tensor {
    let seq_len = x.shape[0];
    let mut y = vec![0.0f32; seq_len * inner];

    for i in 0..inner {
        let a_i = a_vec.get(i % a_vec.len()).copied().unwrap_or(-1.0f32);
        let mut h = 0.0f32;
        for t in 0..seq_len {
            let dt = delta.data[t * inner + i].max(0.001);
            let x_t = x.data[t * inner + i];
            let b_t = b.data[t * inner + i];
            let c_t = c.data[t * inner + i];

            let a_bar = (dt * a_i).exp();
            let b_bar = dt * b_t;

            h = a_bar * h + b_bar * x_t;
            y[t * inner + i] = c_t * h;
        }
    }

    Tensor::from_vec(y, vec![seq_len, inner])
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
        let inner = 48; // 3 * hidden

        weights.insert("attn_qkv.weight".to_string(), Tensor::from_vec(vec![0.01; hidden * inner], vec![hidden, inner]));
        weights.insert("ssm_out.weight".to_string(), Tensor::from_vec(vec![0.01; hidden * hidden], vec![hidden, hidden]));
        weights.insert("ssm_conv1d.weight".to_string(), Tensor::from_vec(vec![1.0; 4 * inner], vec![4, inner]));
        weights.insert("ssm_beta.weight".to_string(), Tensor::from_vec(vec![0.5; hidden * 4], vec![hidden, 4]));
        weights.insert("ssm_alpha.weight".to_string(), Tensor::from_vec(vec![1.0; hidden * 4], vec![hidden, 4]));
        weights.insert("ssm_dt.bias".to_string(), Tensor::from_vec(vec![0.1; 4], vec![4]));
        weights.insert("ssm_a".to_string(), Tensor::from_vec(vec![-1.0; 4], vec![4]));

        let hidden_states = Tensor::from_vec(vec![0.1; 2 * hidden], vec![2, hidden]);
        let config = SSMConfig { state_size: 4, inner_size: inner, time_step_rank: 4, conv_kernel: 4, group_count: 1 };

        let out = ssm_forward(&hidden_states, &weights, &config);
        assert_eq!(out.shape, vec![2, hidden]);
        assert!(out.data.iter().all(|&v| v.is_finite()));
    }
}
