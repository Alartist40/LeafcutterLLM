//! Key aliasing for safetensors → existing engine code.
//!
//! The existing leafcutter inference code (deltanet.rs, attention.rs, mlp.rs)
//! was written for GGUF tensor names. Our safetensors use different names.
//! This module maps between them by creating Tensor entries under both
//! naming conventions when reading from safetensors.

use crate::model::tensor::Tensor;
use crate::safetensor_tensors::SafetensorTensors;
use std::collections::HashMap;

/// Map an Ornith safetensors tensor name to the keys the existing
/// engine code looks up.
pub fn safetensor_to_engine_key(name: &str) -> Option<&'static str> {
    // linear_attn → self_attn.*  (the existing DeltaNet code uses self_attn names)
    if let Some(rest) = name.strip_prefix("model.language_model.layers.") {
        // rest looks like "0.linear_attn.in_proj_qkv.weight"
        let parts: Vec<&str> = rest.splitn(2, '.').collect();
        if parts.len() != 2 {
            return None;
        }
        let layer_str = parts[0];
        let body = parts[1];

        // Handle linear_attn (DeltaNet)
        if let Some(sub) = body.strip_prefix("linear_attn.") {
            let sub = sub.trim_end_matches(".weight");
            match sub {
                "in_proj_qkv" => return Some("self_attn.qkv_proj.weight"),
                "conv1d" => return Some("ssm_conv1d.weight"),
                "A_log" => return Some("ssm_a.weight"), // existing engine uses 'weight'
                "dt_bias" => return Some("ssm_dt.bias"),
                "in_proj_a" => return Some("ssm_alpha.weight"),
                "in_proj_b" => return Some("ssm_beta.weight"),
                "in_proj_z" => return Some("attn_gate.weight"),
                "norm" => return Some("ssm_norm.weight"),
                "out_proj" => return Some("ssm_out.weight"),
                _ => return None,
            }
        }

        // Handle self_attn (full_attention)
        if let Some(sub) = body.strip_prefix("self_attn.") {
            let sub = sub.trim_end_matches(".weight");
            match sub {
                "q_proj" => return Some("self_attn.q_proj.weight"),
                "k_proj" => return Some("self_attn.k_proj.weight"),
                "v_proj" => return Some("self_attn.v_proj.weight"),
                "o_proj" => return Some("self_attn.o_proj.weight"),
                "q_norm" => return Some("self_attn.q_norm.weight"),
                "k_norm" => return Some("self_attn.k_norm.weight"),
                _ => return None,
            }
        }

        // Layer norms
        if body == "input_layernorm.weight" {
            return Some("attn_norm.weight"); // existing convention
        }
        if body == "post_attention_layernorm.weight" {
            return Some("ffn_norm.weight"); // existing convention
        }

        // MLP (same names in both)
        if let Some(sub) = body.strip_prefix("mlp.") {
            match sub.trim_end_matches(".weight") {
                "gate_proj" => return Some("mlp.gate_proj.weight"),
                "up_proj" => return Some("mlp.up_proj.weight"),
                "down_proj" => return Some("mlp.down_proj.weight"),
                _ => return None,
            }
        }

        let _ = layer_str;
    }
    None
}

/// Read a safetensor and return it under BOTH safetensor and engine key names.
/// Returns (engine_key, tensor).
pub fn load_engine_keyed(
    src: &SafetensorTensors,
    safetensor_name: &str,
) -> Option<(String, Tensor)> {
    let t = src.get(safetensor_name)?;
    let engine_key = safetensor_to_engine_key(safetensor_name);
    engine_key.map(|k| (k.to_string(), t))
}

/// Pre-load all weights for one layer under engine key names.
/// Returns a HashMap keyed by engine tensor names.
pub fn load_layer_weights(
    src: &SafetensorTensors,
    layer: usize,
    layer_type: &str,
) -> HashMap<String, Tensor> {
    let mut out = HashMap::new();
    let prefix = format!("model.language_model.layers.{layer}.");
    let candidates: Vec<&str> = match layer_type {
        "linear_attention" => vec![
            "input_layernorm.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.conv1d.weight",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ],
        "full_attention" => vec![
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ],
        _ => return out,
    };
    for cand in candidates {
        let full = format!("{}{}", prefix, cand);
        if let Some((key, t)) = load_engine_keyed(src, &full) {
            out.insert(key, t);
        }
    }
    out
}
