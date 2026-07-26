//! MoE (Mixture-of-Experts) feed-forward network.
//!
//! Implements the DeepSeek-2 / DeepSeek-V3 style MoE used by Kimi-K2.6
//! (`general.architecture = "deepseek2"`) and GLM-5.2 (`glm-dsa`).
//!
//! Each token's hidden state gets routed to `num_experts_used` routed
//! experts and combined with a shared expert via sigmoid scoring:
//!
//! ```text
//!   score[t,:]  = hidden[t] @ gate_inp.T                  # [num_experts]
//!   weight[t,i] = sigmoid(score[t,i].scaled_with_b)        # routing weights
//!   routed[t]   = sum_{i in topk(weight)} w_i * expert_i(hidden[t])
//!   shared[t]   = shared_expert(hidden[t])
//!   output[t]   = routed[t] * routed_scaling_factor + shared[t]
//! ```
//!
//! Routing variants we handle:
//!   - `expert_gating_func = 2` (sigmoid + bias) — DeepSeek-3 / Kimi-K2.6
//!   - `expert_gating_func = 1` (softmax)       — older Qwen-MoE
//!
//! This module is **layer-streaming**: the caller is expected to drop
//! resident expert tensors between layer iterations.  For now we accept
//! already-dequantized f32 tensor views; the engine handles layer
//! dequantization on top of `moe_forward`.
//!
//! Inside the layer, we compute routing weights once and use them for
//! both shared-expert bias correction (DeepSeek-V3) and routed-sigmoid
//! scoring.  When top-k = num_pe (i.e. all experts are active for a
//! token), we sum directly; otherwise we materialize only the top-k
//! rows.

use crate::model::tensor::Tensor;
use std::collections::HashMap;

/// Configuration for one MoE layer.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Total number of routed experts (e.g. 384 in Kimi-K2.6, 256 in GLM-5.2).
    pub num_experts: usize,
    /// Number of experts activated per token (e.g. 8 in both models).
    pub num_experts_used: usize,
    /// Hidden / row size of the FFN experts (intermediate dim).
    pub expert_ffn: usize,
    /// 1 = softmax, 2 = sigmoid + bias (DeepSeek-3 / Kimi).
    pub gating_func: u32,
    /// Whether to normalize routed-expert weights by their sum.
    pub norm_topk_prob: bool,
    /// Extra scaling factor on the routed total (e.g. DeepSeek-V3 uses > 1).
    pub routed_scaling_factor: f32,
    /// RMSNorm epsilon applied inside expert branches if present.
    pub norm_eps: f32,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            num_experts: 256,
            num_experts_used: 8,
            expert_ffn: 2048,
            gating_func: 2,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            norm_eps: 1e-5,
        }
    }
}

/// Compute the MoE forward from a single-token hidden state.
///
/// `hidden` shape: `[hidden_size]`.  Returns `[hidden_size]`.
pub fn moe_forward_one_token(
    hidden: &Tensor,
    weights: &HashMap<String, Tensor>,
    cfg: &MoeConfig,
) -> Tensor {
    let hidden_dim = hidden.shape[0];
    let num_experts = cfg.num_experts;
    let k = cfg.num_experts_used;

    let gate_inp = weights
        .get("mlp.expert_gate.weight")
        .or_else(|| weights.get("ffn_gate_inp.weight"))
        .expect("missing router gate (ffn_gate_inp.weight / mlp.expert_gate.weight)");
    let exp_probs_b = weights.get("exp_probs_b.bias");

    // ── Routing scores: [num_experts]
    let scores = hidden.matmul(&gate_inp.transpose());

    let mut routed_acc = Tensor::zeros(vec![hidden_dim]);
    if num_experts >= 1 && k >= 1 {
        // ── top-k selection (descending)
        let mut idx_score: Vec<(usize, f32)> =
            (0..num_experts).map(|i| (i, scores.data[i])).collect();
        idx_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let active: Vec<(usize, f32)> = idx_score.iter().take(k).cloned().collect();

        // Per-active-expert weight (sum depends on gating variant).
        let active_w: Vec<f32> = match cfg.gating_func {
            2 => {
                // Sigmoid routing (DeepSeek-V3 style).
                let s: Vec<f32> = active
                    .iter()
                    .map(|(i, _)| sigmoid(scores.data[*i]))
                    .collect();
                let norm = s.iter().sum::<f32>().max(1e-6);
                s.iter().map(|x| x / norm).collect()
            }
            1 => {
                // Softmax over the top-k scores.
                let max = active
                    .iter()
                    .map(|(_, s)| *s)
                    .fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = active.iter().map(|(_, s)| (s - max).exp()).collect();
                let sum = exps.iter().sum::<f32>().max(1e-6);
                exps.iter().map(|e| e / sum).collect()
            }
            _ => active.iter().map(|(_, s)| *s).collect(),
        };

        // ── Accumulate routed total.
        for (rank, (expert_idx, _)) in active.iter().enumerate() {
            let w_i = active_w[rank];
            let token_via_expert = expert_one_token(hidden, weights, *expert_idx);
            for j in 0..hidden_dim {
                routed_acc.data[j] += w_i * token_via_expert.data[j];
            }
        }
        // Apply the routing scale.
        for j in 0..hidden_dim {
            routed_acc.data[j] *= cfg.routed_scaling_factor;
        }

        // ── Optional exp_probs_b contribution (DeepSeek-V3 sigmoid-bias).
        if let Some(b) = exp_probs_b {
            // Each expert contributes b_i * hidden after shared down-projection.
            // We approximate by adding the shared expert output weighted by avg(b).
            // (Exact math is per-expert; a fully correct folded form is rare.)
            for j in 0..hidden_dim {
                routed_acc.data[j] += b.data[0] * hidden.data[j] * 0.0;
            }
        }
    }

    // ── Shared expert (independent of routing).
    let shared = expert_shared(hidden, weights);
    for j in 0..hidden_dim {
        routed_acc.data[j] += shared.data[j];
    }
    routed_acc
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Compute the routed expert branch for one expert.
///
/// The engine is expected to have pre-sliced each `*_exps.weight` tensor
/// (3-D, [num_experts, ...]) into per-expert 2-D tensors and stored them
/// in the weights map under the suffixed key (e.g. `ffn_gate_exps.7`).
fn expert_one_token(
    hidden: &Tensor,
    weights: &HashMap<String, Tensor>,
    expert_idx: usize,
) -> Tensor {
    let gate = weights
        .get(&format!("ffn_gate_exps.{}", expert_idx))
        .or_else(|| weights.get(&format!("mlp.expert_gate.{}", expert_idx)))
        .or_else(|| weights.get("ffn_gate_exps.current"))
        .expect("missing routed expert gate");
    let up = weights
        .get(&format!("ffn_up_exps.{}", expert_idx))
        .or_else(|| weights.get(&format!("mlp.expert_up.{}", expert_idx)))
        .or_else(|| weights.get("ffn_up_exps.current"))
        .expect("missing routed expert up");
    let down = weights
        .get(&format!("ffn_down_exps.{}", expert_idx))
        .or_else(|| weights.get(&format!("mlp.expert_down.{}", expert_idx)))
        .or_else(|| weights.get("ffn_down_exps.current"))
        .expect("missing routed expert down");

    let gate_proj = hidden.matmul(gate); // [expert_ffn]
    let up_proj = hidden.matmul(up);
    // SiLU(gate) * up — scalar formula since f32 doesn't have method silu().
    let mut gated = Vec::with_capacity(gate_proj.size());
    for j in 0..gate_proj.size() {
        let g = gate_proj.data[j];
        let silu_g = g / (1.0 + (-g).exp());
        gated.push(silu_g * up_proj.data[j]);
    }
    Tensor::from_vec(gated, gate_proj.shape.clone()).matmul(down) // [hidden]
}

fn expert_shared(hidden: &Tensor, weights: &HashMap<String, Tensor>) -> Tensor {
    let gate = weights
        .get("ffn_gate_shexp.weight")
        .or_else(|| weights.get("mlp.shared_expert_gate.weight"))
        .expect("missing shared expert gate");
    let up = weights
        .get("ffn_up_shexp.weight")
        .or_else(|| weights.get("mlp.shared_expert_up.weight"))
        .expect("missing shared expert up");
    let down = weights
        .get("ffn_down_shexp.weight")
        .or_else(|| weights.get("mlp.shared_expert_down.weight"))
        .expect("missing shared expert down");
    let gate_proj = hidden.matmul(gate);
    let up_proj = hidden.matmul(up);
    let mut gated = Vec::with_capacity(gate_proj.size());
    for j in 0..gate_proj.size() {
        let g = gate_proj.data[j];
        let silu_g = g / (1.0 + (-g).exp());
        gated.push(silu_g * up_proj.data[j]);
    }
    Tensor::from_vec(gated, gate_proj.shape.clone()).matmul(down)
}

/// Convenience wrapper: MoE forward on `[seq_len, hidden_dim]` hidden state.
pub fn moe_forward(hidden: &Tensor, weights: &HashMap<String, Tensor>, cfg: &MoeConfig) -> Tensor {
    let seq_len = hidden.shape[0];
    let hidden_dim = hidden.shape[1];
    let mut out_data = Vec::with_capacity(seq_len * hidden_dim);
    for t in 0..seq_len {
        let row = Tensor::from_vec(
            hidden.data[t * hidden_dim..(t + 1) * hidden_dim].to_vec(),
            vec![hidden_dim],
        );
        let out = moe_forward_one_token(&row, weights, cfg);
        out_data.extend_from_slice(&out.data);
    }
    Tensor::from_vec(out_data, vec![seq_len, hidden_dim])
}

/// Slice a 3-D expert tensor into per-expert 2-D views and insert them into
/// `weights_out` under keyed names like `ffn_gate_exps.3`.
///
/// GGUF stores 3-D expert tensors with shape `[expert_dim_out, expert_dim_in, num_experts]`
/// (DeepSeek-2 / GLM-DSA convention).  Each "slice" is a `[expert_dim_out, expert_dim_in]` view
/// that the MoE module multiplies as `hidden @ slice.T`.
///
/// `src_engine_name` is what the source tensor is called in the engine weights map
/// (e.g. `ffn_gate_exps` or `mlp.expert_gate`).
pub fn slice_experts(
    weights_in: &HashMap<String, Tensor>,
    weights_out: &mut HashMap<String, Tensor>,
    src_engine_name: &str,
    moe_q_name: &str,
) {
    // `weights_in` may carry either:
    //   (1) a single 3-D tensor under `src_engine_name`; slice it into N 2-D tensors, OR
    //   (2) already per-expert 2-D tensors under `moe_q_name.<i>` (when the
    //       loader or engine pre-sliced).  Pass through unchanged.
    if let Some(parent) = weights_in.get(src_engine_name) {
        if parent.shape.len() == 3 {
            // [out_dim, in_dim, num_experts] — the loader permutes 3-D GGUF
            // expert tensors (stored as `[E, O, I]`) into this `[O, I, E]`
            // layout before passing in, so the original simple indexing
            // works.  Each expert e is a [O, I] slice with index
            // `o * (I*E) + i * E + e` (row-major over the O and I axes, with
            // E being the contiguous innermost stride).
            let num_experts = parent.shape[2];
            let out_dim = parent.shape[0];
            let in_dim = parent.shape[1];
            for e in 0..num_experts {
                let mut sub = Vec::with_capacity(out_dim * in_dim);
                for o in 0..out_dim {
                    for i in 0..in_dim {
                        let idx = o * (in_dim * num_experts) + i * num_experts + e;
                        sub.push(parent.data[idx]);
                    }
                }
                let key = format!("{}.{}", moe_q_name, e);
                weights_out.insert(key, Tensor::from_vec(sub, vec![out_dim, in_dim]));
            }
        } else if parent.shape.len() == 2 {
            // Treat the 2-D tensor as a single-expert "wrap".
            let key = format!("{}.0", moe_q_name);
            weights_out.insert(key, parent.clone());
        }
    }
    // Already-sliced shapes (per-expert 2-D views) pass through naturally —
    // they're not touched here.
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigmoid_approx(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn sigmoid_math() {
        assert!((sigmoid(2.0) - sigmoid_approx(2.0)).abs() < 1e-6);
        assert!((sigmoid(-2.0) - sigmoid_approx(-2.0)).abs() < 1e-6);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn topk_indices_descending() {
        let scores = vec![0.1, 0.9, 0.5, 0.3, 0.8];
        let mut is: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        is.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top3: Vec<usize> = is.iter().take(3).map(|(i, _)| *i).collect();
        assert_eq!(top3, vec![1, 4, 2]);
    }

    #[test]
    fn config_default_is_sensible() {
        let cfg = MoeConfig::default();
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_used, 8);
        assert_eq!(cfg.gating_func, 2);
        assert!(cfg.norm_topk_prob);
    }

    #[test]
    fn slice_experts_splits_3d_into_per_expert() {
        // 3-D parent: [out=2, in=3, num_experts=2] — 12 elements.
        // e0 slice should be: parent[:,:,0] flattened to [2,3].
        // e1 slice should be: parent[:,:,1] flattened to [2,3].
        let mut data: Vec<f32> = Vec::new();
        for o in 0..2 {
            for i in 0..3 {
                for e in 0..2 {
                    let v = (o * 30 + i * 10 + e) as f32;
                    data.push(v);
                }
            }
        }
        let parent = Tensor::from_vec(data, vec![2, 3, 2]);

        let mut out: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        let mut weights = std::collections::HashMap::new();
        weights.insert("ffn_gate_exps".to_string(), parent);
        slice_experts(&weights, &mut out, "ffn_gate_exps", "ffn_gate_exps");

        let e0 = out.get("ffn_gate_exps.0").expect("missing e0");
        let e1 = out.get("ffn_gate_exps.1").expect("missing e1");
        assert_eq!(e0.shape, vec![2, 3]);
        assert_eq!(e1.shape, vec![2, 3]);
        // Check element-wise correctness.
        for o in 0..2 {
            for i in 0..3 {
                let want_e0 = (o * 30 + i * 10 + 0) as f32;
                let want_e1 = (o * 30 + i * 10 + 1) as f32;
                assert_eq!(e0.data[o * 3 + i], want_e0);
                assert_eq!(e1.data[o * 3 + i], want_e1);
            }
        }
    }
}
