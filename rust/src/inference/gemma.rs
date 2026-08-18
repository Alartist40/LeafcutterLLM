//! Gemma-family RMSNorm + per-layer routing.
//!
//! Gemma's variant of transformer block differs from Llama in three places
//! that matter here:
//!
//! 1. **RMSNorm scaling**: Gemma RMSNorm applies `(1 + weight)` instead of
//!    just `weight`.  
//! 2. **Per-head RMSNorm on Q and K**: from Gemma 2 onward, codified in
//!    Gemma 3/4. Handled inside `attention::attention_forward`.
//! 3. **Alternating attention layers**: from Gemma 3 onward, alternating
//!    global vs sliding-window layers per metadata pattern. Read from
//!    `gemma4.attention.sliding_window_pattern` etc.
//!
//! S ("sliding") layers also fold `attn_v` into the second half of
//! `attn_q`, so the GGUF tensor subset is just `[attn_q, attn_k]` (no
//! `attn_v`).  We reconstruct the fused Q+V tensor at runtime before
//! dispatching to `attention::attention_forward` in fused-QKV mode.
//!
//! `gemma_layer_forward` is the single entry point the engine calls.

use super::attention::{attention_forward, AttentionParams};
use crate::cache::KVCache;
use crate::model::tensor::Tensor;
use std::collections::HashMap;

/// Helper: remove a weight from the map, materialize its f32 data (if it
/// was loaded as quantized-only), and re-insert it.  This is the cleanest
/// way to drop the borrow on the map before doing further lookups.
fn materialize_in_place(weights: &mut HashMap<String, Tensor>, key: &str) {
    if let Some(mut t) = weights.remove(key) {
        t.materialize_data();
        weights.insert(key.to_string(), t);
    }
}

/// Per-layer Gemma config, derived from `gemma4.attention.*` metadata.
#[derive(Debug, Clone)]
pub struct GemmaLayerParams {
    pub num_kv_heads: usize,
    pub q_head_dim: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub is_global: bool,
    pub rope_theta: f32,
}

impl Default for GemmaLayerParams {
    fn default() -> Self {
        Self {
            num_kv_heads: 8,
            q_head_dim: 256,
            k_head_dim: 256,
            v_head_dim: 256,
            is_global: true,
            rope_theta: 1_000_000.0,
        }
    }
}

/// Gemma RMSNorm.
///
/// **Gemma 4 (this model):** `y = x * rsqrt(mean(x²) + eps) * w` — the weight
/// is applied **directly**, with NO `+1` shift. This matches HF
/// `Gemma4RMSNorm.forward` (modeling_gemma4.py:207-211), which intentionally
/// differs from Gemma 2/3:
///   - Gemma 2/3 (`Gemma{2,3}RMSNorm`): `output * (1.0 + self.weight)` — `+1`.
///   - Gemma 4 (`Gemma4RMSNorm`):      `output * self.weight`       — direct.
///
/// The on-disk GGUF weight is the trained `γ` (initialized to `ones`, applied
/// directly); the GGUF converter does not bake in a `+1`. Applying `(w + 1)`
/// here therefore inflates every activation by a factor of up to ~2x per norm,
/// which compounds across 48 layers × 4 norms = 192 applications and is the
/// leading cause of the "logits 10–20× too large, degenerate generation"
/// symptom (Gemma 4 investigation, Finding 1).
///
/// `with_scale=False` norms (e.g. Gemma 4 `v_norm`) pass an all-ones weight
/// and reduce to pure RMS — handled by the same code path.
pub fn gemma_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    let n = x.shape.last().copied().unwrap_or(x.data.len());
    let seq = x.data.len() / n.max(1);
    let mut out = Vec::with_capacity(seq * n);
    if seq == 0 || n == 0 {
        return Tensor::zeros(x.shape.clone());
    }
    let inv_n = 1.0 / n as f32;
    for s in 0..seq {
        let base = s * n;
        let sum_sq: f32 = x.data[base..base + n].iter().map(|&v| v * v).sum();
        let rms = (sum_sq * inv_n + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for d in 0..n {
            // Match HF Gemma4: `y = x * inv_rms * w` (direct, no +1).
            let w = weight.data[d];
            out.push(x.data[base + d] * inv_rms * w);
        }
    }
    Tensor::from_vec(out, x.shape.clone())
}

/// GeGLU FFN: `down × (GeLU(gate × x) ⊙ (up × x))`.
pub fn gemma_ffn_forward(x: &Tensor, weights: &HashMap<String, Tensor>) -> Result<Tensor, String> {
    let gate = weights
        .get("mlp.gate_proj.weight")
        .ok_or_else(|| "gemma_ffn: missing mlp.gate_proj.weight".to_string())?;
    let up = weights
        .get("mlp.up_proj.weight")
        .ok_or_else(|| "gemma_ffn: missing mlp.up_proj.weight".to_string())?;
    let down = weights
        .get("mlp.down_proj.weight")
        .ok_or_else(|| "gemma_ffn: missing mlp.down_proj.weight".to_string())?;
    let gate_proj = x.matmul(gate);
    let up_proj = x.matmul(up);
    let inv_sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let mut fused = vec![0.0f32; gate_proj.data.len()];
    for i in 0..gate_proj.data.len() {
        let gv = gate_proj.data[i];
        let gelu = 0.5 * gv * (1.0 + (inv_sqrt_2_pi * (gv + 0.044715 * gv * gv * gv)).tanh());
        fused[i] = gelu * up_proj.data[i];
    }
    Ok(Tensor::from_vec(fused, gate_proj.shape.clone()).matmul(down))
}

/// Build a synthetic fused QKV tensor for a Gemma-style attention layer.
/// On G ("global") layers, K and V are independent tensors.  On S
/// ("sliding") layers, V is baked into the second half of the projected
/// Q tensor (Gemma 3+ convention).
/// Build a synthetic fused QKV weight matrix for a Gemma-style attention layer.
/// Output shape is `[hidden_size, total_out_dim]` so it can be `matmul`'d as
/// `hidden @ qkv_weight` to produce `[seq_len, total_out_dim]`.
///
/// On G ("global") layers, K and V are independent tensors.  On S
/// ("sliding") layers, V is baked into the second half of the projected
/// Q tensor — q_proj is twice the size of K_proj.
//
// Refactor: instead of taking matmul of hidden with each projection, then
// concatenating result tensors, we *build a single weight matrix* by
// column-stacking the gguf weights.  Because GGUF weights are stored as
// [in, out] row-major, those are already in the form `[in_dim, out_dim]`,
// so we can just stick them side-by-side along the output axis.
pub fn gemma_fused_qkv(_hidden: &Tensor, layer_weights: &mut HashMap<String, Tensor>) -> Result<Tensor, String> {
    // Locate + materialize f32 data from the quantized weight (loader stores
    // Q4_K/Q6_K tensors with empty `data` and populated `q_data`).
    materialize_in_place(layer_weights, "self_attn.q_proj.weight");
    materialize_in_place(layer_weights, "self_attn.k_proj.weight");
    materialize_in_place(layer_weights, "self_attn.v_proj.weight");
    let q_w = layer_weights
        .get("self_attn.q_proj.weight")
        .or_else(|| layer_weights.get("attn_q.weight"))
        .ok_or_else(|| "gemma_fused_qkv: missing attn_q/attn_q_proj.weight".to_string())?;
    let k_w = layer_weights
        .get("self_attn.k_proj.weight")
        .or_else(|| layer_weights.get("attn_k.weight"))
        .ok_or_else(|| "gemma_fused_qkv: missing attn_k/attn_k_proj.weight".to_string())?;
    assert_eq!(q_w.shape.len(), 2, "weight must be 2-D, got {:?}", q_w.shape);
    assert_eq!(k_w.shape.len(), 2, "weight must be 2-D, got {:?}", k_w.shape);

    // Try a separate V projection first.
    let v_w_opt = layer_weights
        .get("self_attn.v_proj.weight")
        .or_else(|| layer_weights.get("attn_v.weight"));

    Ok(if let Some(v_w) = v_w_opt {
        // Global (G) layer: K and V are independent projections.
        assert_eq!(v_w.shape.len(), 2, "weight must be 2-D, got {:?}", v_w.shape);
        // Stack Q, K, V along the output axis (column-wise concat).
        // All have same in_dim (hidden_size).
        let in_dim = q_w.shape[0];
        let q_out = q_w.shape[1];
        let k_out = k_w.shape[1];
        let v_out = v_w.shape[1];
        let total_out = q_out + k_out + v_out;
        // GGUF row-major GGUF, so each weight is contiguous in dim 1 then 0.
        // Stacking along output axis: out[i] is from q, k, or v based on column.
        let mut data = vec![0.0f32; in_dim * total_out];
        for r in 0..in_dim {
            let q_row = &q_w.data[r * q_out..(r + 1) * q_out];
            let k_row = &k_w.data[r * k_out..(r + 1) * k_out];
            let v_row = &v_w.data[r * v_out..(r + 1) * v_out];
            let mut dst = &mut data[r * total_out..r * total_out + q_out];
            dst.copy_from_slice(q_row);
            let mut dst = &mut data[r * total_out + q_out..r * total_out + q_out + k_out];
            dst.copy_from_slice(k_row);
            let mut dst = &mut data[r * total_out + q_out + k_out..r * total_out + total_out];
            dst.copy_from_slice(v_row);
        }
        Tensor::from_vec(data, vec![in_dim, total_out])
    } else {
        // Gemma 4 GLOBAL layer with single KV head: no V projection in GGUF.
        // Reference impl: V = K (llama.cpp gemma4.cpp line 247-248 — when V is
        // absent, Vcur is aliased to Kcur).  We build [Q_full | K | K_clone]
        // so attention_forward splits it into Q, K, V where V == K's values.
        // Per llama.cpp the engine check attention_forward's V tensor uses
        // `v.data` which is now K's contents — attention math is correct.
        let in_dim = q_w.shape[0];
        let q_total = q_w.shape[1];
        let k_out = k_w.shape[1];
        let v_out = k_out; // V = K shape
        let total_out = q_total + k_out + v_out;
        let mut data = vec![0.0f32; in_dim * total_out];
        for r in 0..in_dim {
            let q_row = &q_w.data[r * q_total..(r + 1) * q_total];
            let k_row = &k_w.data[r * k_out..(r + 1) * k_out];
            let dst_start = r * total_out;
            data[dst_start..dst_start + q_total].copy_from_slice(q_row);
            data[dst_start + q_total..dst_start + q_total + k_out].copy_from_slice(k_row);
            data[dst_start + q_total + k_out..dst_start + total_out].copy_from_slice(k_row);
        }
        Tensor::from_vec(data, vec![in_dim, total_out])
    })
}


/// Run a single Gemma-4 transformer block for one layer.
///
/// Mirrors HF `Gemma4TextDecoderLayer.forward` (modeling_gemma4.py:1392-1439)
/// exactly:
/// ```text
///   residual = x
///   h = input_layernorm(x)                # attn_norm
///   h = attention(h)
///   h = post_attention_layernorm(h)       # post_attention_norm
///   x = residual + h                      # ★ NO layer_output_scale here
///
///   residual = x
///   h = pre_feedforward_layernorm(x)      # ffn_norm
///   h = mlp(h)                            # GeGLU
///   h = post_feedforward_layernorm(h)     # post_ffw_norm
///   x = residual + h                      # ★ NO layer_output_scale here
/// ```
///
/// `layer_scalar` / `layer_output_scale`: HF declares this as a fixed buffer
/// (`register_buffer("layer_scalar", torch.ones(1))`, init.ones_), i.e. **always
/// 1.0** in real checkpoints. We deliberately IGNORE the `layer_output_scale.weight`
/// tensor because the Q4_K_M GGUF we target contains garbage in those slots
/// (`7e+32`, `nan`, …) — a known Gemma-4 conversion artifact (Gemma 4
/// investigation, Finding 4). Multiplying by it destroyed the residual
/// stream from layer 0 onward.
pub fn gemma_layer_forward(
    hidden: &Tensor,
    layer_weights: &mut HashMap<String, Tensor>,
    layer_cfg: &GemmaLayerParams,
    global_cfg: &AttentionParams,
    kv_cache: &mut KVCache,
    layer_idx: usize,
    position_offset: usize,
    rms_eps: f32,
) -> Result<Tensor, String> {
    // ── Attention sub-block ──
    // Gemma's `attn_norm.weight` is engine-mapped to `input_layernorm.weight`
    // (per arch.rs tensor-mapping table). Fall back to either name.
    let attn_norm_w = layer_weights
        .get("attn_norm.weight")
        .or_else(|| layer_weights.get("input_layernorm.weight"))
        .ok_or_else(|| format!("layer {}: missing attn_norm / input_layernorm", layer_idx))?;
    if std::env::var("LEAFCUTTER_DEBUG_NORMS").is_ok() && layer_idx < 2 {
        let n = attn_norm_w.data.len();
        let l2: f32 = attn_norm_w.data.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let mx = attn_norm_w.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mn = attn_norm_w.data.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!(
            "[NORM-DIAG] L{}_attn_norm_w  n={} l2={:.3} min={:.4} max={:.4}  first5=[{:.4},{:.4},{:.4},{:.4},{:.4}]  keys={:?}",
            layer_idx, n, l2, mn, mx,
            attn_norm_w.data[0], attn_norm_w.data[1],
            attn_norm_w.data.get(2).copied().unwrap_or(0.0),
            attn_norm_w.data.get(3).copied().unwrap_or(0.0),
            attn_norm_w.data.get(4).copied().unwrap_or(0.0),
            &layer_weights.keys().collect::<Vec<_>>()[..8.min(layer_weights.len())]
        );
    }
    let pre_attn = gemma_rms_norm(hidden, attn_norm_w, rms_eps);
    debug_norm(&format!("L{layer_idx}_pre_attn_rmsnorm"), &pre_attn);
    let qkv = gemma_fused_qkv(&pre_attn, layer_weights)?;
    let mut weights_with_qkv = layer_weights.clone();
    weights_with_qkv.insert("attn_qkv.weight".to_string(), qkv);

    // Build per-layer AttentionParams from ACTUAL layer tensor shapes, not
    // from infer_gemma_layouts metadata — sliding-window layers can have
    // q_proj_out doubled (V baked into Q), and we need head_dim to match
    // total q_proj_out / num_heads so attention_forward shapes line up.
    let q_proj = layer_weights
        .get("self_attn.q_proj.weight")
        .or_else(|| layer_weights.get("attn_q.weight"));
    let k_proj = layer_weights
        .get("self_attn.k_proj.weight")
        .or_else(|| layer_weights.get("attn_k.weight"));
    let v_proj = layer_weights
        .get("self_attn.v_proj.weight")
        .or_else(|| layer_weights.get("attn_v.weight"));
    // Default values from layer_cfg (which itself comes from GGUF metadata).
    let mut num_heads: usize = 16;
    let mut num_kv_heads = layer_cfg.num_kv_heads;
    let mut head_dim = layer_cfg.q_head_dim;
    let mut kv_head_dim = layer_cfg.k_head_dim;
    // True if V exists as a separate projection (G-layer); otherwise V is
    // baked into the second half of Q's projection (S-layer) and head_dim is
    // actually q_proj_out / num_heads (no doubling of head_dim).
    let has_separate_v = v_proj.is_some();
    if let Some(q) = q_proj {
        if let Some(k) = k_proj {
            let q_out = q.shape[1];
            let k_out = k.shape[1];
            if has_separate_v {
                // G layer: q_out = num_heads * head_dim, k_out = num_kv_heads * kv_head_dim.
                head_dim = if num_heads > 0 { q_out / num_heads } else { head_dim };
                kv_head_dim = if num_kv_heads > 0 { k_out / num_kv_heads } else { kv_head_dim };
            } else {
                // S layer: q_out = num_heads * (head_dim + kv_head_dim) IF
                // each Q-head has a paired V-head, OR q_out = num_heads *
                // head_dim where each head_dim already includes Q+V halves.
                // We pick head_dim such that attention_forward sees:
                //   q_dim_total = q_out (the whole Q projection), because
                // the engine's matmul shape expects q_dim = num_heads * head_dim
                // and total_fused = q_dim + 2*kv_dim where the V lives in the
                // second half of Q's projection.
                head_dim = if num_heads > 0 { q_out / num_heads } else { head_dim };
                // Decide kv_head_dim: assume num_kv_heads stays at
                // layer_cfg.num_kv_heads (per metadata).
                if num_kv_heads > 0 {
                    kv_head_dim = (k_out / num_kv_heads).max(1);
                } else {
                    num_kv_heads = 1;
                    kv_head_dim = k_out;
                }
            }
        }
    }
    let per_layer = AttentionParams {
        num_heads,
        num_kv_heads,
        head_dim,
        kv_head_dim,
        rope_theta: layer_cfg.rope_theta,
        rope_dim: global_cfg.rope_dim,
        use_fused_qkv: true,
        use_gate: false,
        // sliding layers use window_size=1024; global layers use 0 (no mask)
        window_size: if layer_cfg.is_global { 0 } else { 1024 },
        yarn: None,
        temp_scale: 0.0,
        temp_floor_scale: 0,
        rope_pair_norm: false,
    };
    let attn_out = attention_forward(
        &pre_attn,
        &weights_with_qkv,
        &per_layer,
        kv_cache,
        layer_idx,
        position_offset,
    );
    debug_norm(&format!("L{layer_idx}_attn_out"), &attn_out);

    // post-attention norm on attn_out (Gemma norm sits BETWEEN sublayer output
    // and the residual add — see HF Gemma4TextDecoderLayer.forward lines 1405-1406).
    let post_attn_w = layer_weights
        .get("post_attention_norm.weight")
        .ok_or_else(|| format!("layer {}: missing post_attention_norm.weight", layer_idx))?;
    let attn_normed = gemma_rms_norm(&attn_out, post_attn_w, rms_eps);
    debug_norm(&format!("L{layer_idx}_post_attn_norm"), &attn_normed);

    // Residual add — NO layer_output_scale (HF uses layer_scalar=1.0 always;
    // the GGUF slot is garbage, see doc comment above).
    let mut x = hidden.add(&attn_normed);
    debug_norm(&format!("L{layer_idx}_after_attn_residual"), &x);

    // ── FFN sub-block ──
    // `ffn_norm.weight` is engine-mapped to `pre_feedforward_layernorm.weight`
    // (a.k.a. the legacy `post_attention_layernorm.weight` Llama name).
    let ffn_norm_w = layer_weights
        .get("ffn_norm.weight")
        .or_else(|| layer_weights.get("post_attention_layernorm.weight"))
        .ok_or_else(|| format!("layer {}: missing ffn_norm / post_attention_layernorm", layer_idx))?;
    let ffn_in = gemma_rms_norm(&x, ffn_norm_w, rms_eps);
    let ffn_out = gemma_ffn_forward(&ffn_in, layer_weights)?;
    let post_ffw_w = layer_weights
        .get("post_ffw_norm.weight")
        .ok_or_else(|| format!("layer {}: missing post_ffw_norm.weight", layer_idx))?;
    let ffn_normed = gemma_rms_norm(&ffn_out, post_ffw_w, rms_eps);
    // Residual add — NO layer_output_scale.
    x = x.add(&ffn_normed);
    debug_norm(&format!("L{layer_idx}_after_ffn_residual"), &x);
    Ok(x)
}

/// Print L2/min/max stats for a Tensor label if LEAFCUTTER_DEBUG_NORMS is set.
fn debug_norm(label: &str, t: &Tensor) {
    if std::env::var("LEAFCUTTER_DEBUG_NORMS").is_err() {
        return;
    }
    let n = t.data.len();
    if n == 0 {
        eprintln!("[NORM] {}  [EMPTY]", label);
        return;
    }
    let l2 = t.data.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let max = t.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = t.data.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!(
        "[NORM] label={:<24}  n={:>6}  l2={:>10.3}  min={:>12.4}  max={:>12.4}",
        label, n, l2, min, max
    );
}

/// Apply Gemma logit soft-capping: clamp out-of-range logits via `tanh`.
pub fn apply_logit_softcap(logits: &mut [f32], cap: f32) {
    if cap <= 0.0 {
        return;
    }
    for v in logits.iter_mut() {
        *v = cap * ((*v) / cap).tanh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_layer_params_default_is_global_8_kv_heads() {
        let l = GemmaLayerParams::default();
        assert!(l.is_global);
        assert_eq!(l.num_kv_heads, 8);
        assert_eq!(l.q_head_dim, 256);
    }

    #[test]
    fn gemma_rms_norm_multiplies_weight_directly() {
        // Reference: HF `Gemma4RMSNorm.forward` (modeling_gemma4.py:207-211)
        // does `y = x * rsqrt(mean(x²) + eps) * w` — weight applied DIRECTLY,
        // no `+1`. Gemma 4 differs from Gemma 2/3 (which use `(1 + w)`).
        // (Gemma 2/3 store `(1 + γ₀)` semantics; Gemma 4 reverted to direct.)
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // Row 0: x = [1, 2]; mean(x²) = (1 + 4)/2 = 2.5; inv_rms = 1/√2.5 ≈ 0.6325
        //   y[0] = 1 * 0.6325 * w[0] = 0.6325 / 2 = 0.31625
        //   y[1] = 2 * 0.6325 * w[1] = 1.265 * 0  = 0
        let w = Tensor::from_vec(vec![0.5, 0.0], vec![2]);
        let y = gemma_rms_norm(&x, &w, 1e-6);
        assert!(
            (0.30..0.34).contains(&y.data[0]),
            "y[0] = {} (expected ≈0.316)",
            y.data[0]
        );
        assert!(y.data[1].abs() < 1e-5, "y[1] = {}", y.data[1]);
        // Row 1: x = [3, 4]; mean(x²) = (9 + 16)/2 = 12.5; inv_rms = 1/√12.5 ≈ 0.2828
        //   y[2] = 3 * 0.2828 * w[0] = 0.8485 * 0.5 = 0.4243
        //   y[3] = 4 * 0.2828 * w[1] = 1.1314 * 0 = 0
        assert!(
            (0.40..0.45).contains(&y.data[2]),
            "y[2] = {} (expected ≈0.424)",
            y.data[2]
        );
        assert!(y.data[3].abs() < 1e-5, "y[3] = {}", y.data[3]);
    }

    #[test]
    fn apply_logit_softcap_clips_when_cap_set() {
        let mut logits = vec![100.0, 0.0, -100.0, 60.0];
        apply_logit_softcap(&mut logits, 30.0);
        // |x|=100 is far past cap, so |result| < cap (≈ 30).
        assert!(logits[0].abs() <= 30.0 + 1e-3, "logits[0]={}", logits[0]);
        assert!((logits[1]).abs() < 1e-6);
        // tanh(60/30) = tanh(2) ≈ 0.964
        let p = 30.0_f32 * (60.0_f32 / 30.0).tanh();
        assert!((logits[3] - p).abs() < 1e-3, "logits[3]={} want {}", logits[3], p);
    }

    #[test]
    fn apply_logit_softcap_passthrough_when_cap_zero() {
        let mut logits = vec![100.0, 0.0];
        apply_logit_softcap(&mut logits, 0.0);
        assert_eq!(logits[0], 100.0);
        assert_eq!(logits[1], 0.0);
    }
}
