//! GGUF weight provider bridge
//!
//! Reads weights from a GGUF file and presents them as
//! `HashMap<String, Vec<f32>>` with the short-name keys that
//! `streaming_ornith.rs` expects.
//!
//! Also provides `GGUFWeightProvider` — a `WeightProvider` implementation
//! that wraps `GGUFile` for use directly in the streaming engine.
//!
//! Handles:
//!   - Name mapping  (blk.{i}.ssm_alpha.weight → linear_attn.in_proj_a.weight)
//!   - A_log inversion (GGUF stores -exp(A_log), recovers raw A_log)
//!   - Conv1d dimension transposition (GGUF [kernel, conv_dim] → [conv_dim, kernel])

use std::collections::HashMap;
use std::path::Path;

use crate::model::gguf::{GGUFile, GGUError};
use crate::ornith_config::OrnithConfig;
use crate::safetensors_loader::WeightProvider;

/// GGUF suffix → streaming engine short name (linear attention layers)
const LINEAR_ATTN_MAP: &[(&str, &str)] = &[
    ("attn_qkv.weight",       "linear_attn.in_proj_qkv.weight"),
    ("attn_gate.weight",      "linear_attn.in_proj_z.weight"),
    ("ssm_alpha.weight",      "linear_attn.in_proj_a.weight"),
    ("ssm_beta.weight",       "linear_attn.in_proj_b.weight"),
    ("ssm_conv1d.weight",     "linear_attn.conv1d.weight"),
    ("ssm_a",                 "linear_attn.A_log"),
    ("ssm_dt.bias",           "linear_attn.dt_bias"),
    ("ssm_norm.weight",       "linear_attn.norm.weight"),
    ("ssm_out.weight",        "linear_attn.out_proj.weight"),
    ("attn_norm.weight",      "input_layernorm.weight"),
    ("post_attention_norm.weight", "post_attention_layernorm.weight"),
    ("ffn_gate.weight",       "mlp.gate_proj.weight"),
    ("ffn_up.weight",         "mlp.up_proj.weight"),
    ("ffn_down.weight",       "mlp.down_proj.weight"),
];

/// GGUF suffix → streaming engine short name (full attention layers)
const FULL_ATTN_MAP: &[(&str, &str)] = &[
    ("attn_q.weight",         "self_attn.q_proj.weight"),
    ("attn_k.weight",         "self_attn.k_proj.weight"),
    ("attn_v.weight",         "self_attn.v_proj.weight"),
    ("attn_output.weight",    "self_attn.o_proj.weight"),
    ("attn_q_norm.weight",    "self_attn.q_norm.weight"),
    ("attn_k_norm.weight",    "self_attn.k_norm.weight"),
    ("attn_norm.weight",      "input_layernorm.weight"),
    ("post_attention_norm.weight", "post_attention_layernorm.weight"),
    ("ffn_gate.weight",       "mlp.gate_proj.weight"),
    ("ffn_up.weight",         "mlp.up_proj.weight"),
    ("ffn_down.weight",       "mlp.down_proj.weight"),
];

/// GGUF name → streaming engine name for non-layer tensors
const NON_LAYER_MAP: &[(&str, &str)] = &[
    ("token_embd.weight",     "model.language_model.embed_tokens.weight"),
    ("output_norm.weight",    "model.language_model.norm.weight"),
    ("output.weight",         "lm_head.weight"),
];

// ── WeightProvider implementation ─────────────────────────────────────

/// A `WeightProvider` that reads from a GGUF file.
pub struct GGUFWeightProvider {
    gguf: GGUFile,
    /// Cached non-layer weights (embed, norm, lm_head)
    non_layer: HashMap<String, Vec<f32>>,
}

impl GGUFWeightProvider {
    pub fn open(path: &str) -> Result<Self, GGUError> {
        let gguf = GGUFile::open(path)?;
        let non_layer = load_gguf_non_layer_weights(&gguf)
            .map_err(|e| GGUError::InvalidTensorType(0))?;
        Ok(Self { gguf, non_layer })
    }

    /// Wrap an already-opened GGUFile.
    pub fn from_gguf(gguf: GGUFile) -> Result<Self, GGUError> {
        let non_layer = load_gguf_non_layer_weights(&gguf)
            .map_err(|e| GGUError::InvalidTensorType(0))?;
        Ok(Self { gguf, non_layer })
    }

    /// Access the underlying GGUFile (for config extraction, etc.)
    pub fn gguf_file(&self) -> &GGUFile {
        &self.gguf
    }
}

impl WeightProvider for GGUFWeightProvider {
    fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        // Check cached non-layer weights first
        if let Some(data) = self.non_layer.get(name) {
            return Ok(data.clone());
        }
        // Otherwise read from GGUF
        read_gguf_tensor_f32(&self.gguf, name)
            .map_err(|e| format!("{name}: {e}"))
    }

    fn read_tensor_slice_f32(&self, name: &str, offset: usize, count: usize) -> Result<Vec<f32>, String> {
        // Check cached non-layer weights first
        if let Some(data) = self.non_layer.get(name) {
            if offset + count <= data.len() {
                return Ok(data[offset..offset + count].to_vec());
            }
            return Err(format!("{name}: slice out of bounds (offset={offset} count={count}, len={})", data.len()));
        }
        // Read subsections of the tensor via row-based access
        let info = self.gguf.get_tensor_info(name)
            .ok_or_else(|| format!("{name}: tensor not found"))?;
        let cols = info.dimensions[0] as usize;
        let row_start = offset / cols;
        let row_end = (offset + count + cols - 1) / cols;
        let mut out = Vec::with_capacity(count);
        for row in row_start..row_end {
            let row_data = self.gguf.get_tensor_row_f32(name, row)
                .ok_or_else(|| format!("{name}: row {row} read failed"))?;
            let chunk_off = if row == row_start { offset % cols } else { 0 };
            let chunk_end = if row == row_end - 1 {
                let rem = (offset + count) % cols;
                if rem == 0 { cols } else { rem }
            } else {
                cols
            };
            out.extend_from_slice(&row_data[chunk_off..chunk_end]);
        }
        Ok(out)
    }

    fn load_layer_weights(
        &self,
        layer_idx: usize,
        layer_type: &str,
        _layer_names: &[&str],
        _prefix: &str,
    ) -> Result<HashMap<String, Vec<f32>>, String> {
        load_gguf_layer_weights(&self.gguf, layer_idx, layer_type)
    }
}

// ── Layer weight loading ─────────────────────────────────────────────

/// Load weights for one transformer layer from a GGUF file.
pub fn load_gguf_layer_weights(
    gguf: &GGUFile,
    layer_idx: usize,
    layer_type: &str,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let prefix = format!("blk.{}", layer_idx);
    let mapping = match layer_type {
        "linear_attention" => LINEAR_ATTN_MAP,
        "full_attention" => FULL_ATTN_MAP,
        other => return Err(format!("unknown layer type: {other}")),
    };

    let mut w = HashMap::new();

    for (gguf_suffix, engine_name) in mapping {
        let gguf_name = format!("{prefix}.{gguf_suffix}");

        // Special handling for A_log (ssm_a)
        if *gguf_suffix == "ssm_a" {
            let a_vals = read_gguf_tensor_f32(gguf, &gguf_name)
                .map_err(|e| format!("{gguf_name}: {e}"))?;
            let a_log: Vec<f32> = a_vals
                .iter()
                .map(|&v| {
                    let neg = -v;
                    if neg <= 0.0 {
                        f32::NEG_INFINITY
                    } else {
                        neg.ln()
                    }
                })
                .collect();
            w.insert(engine_name.to_string(), a_log);
            continue;
        }

        // Special handling for conv1d: GGUF stores dims=[kernel, conv_dim] with
        // dims[0]=kernel contiguous per channel, i.e. flat data is channel-major
        // [c*conv_k + k].  That is already the engine layout, so no transpose.
        if *gguf_suffix == "ssm_conv1d.weight" {
            let conv_vals = read_gguf_tensor_f32(gguf, &gguf_name)
                .map_err(|e| format!("{gguf_name}: {e}"))?;
            w.insert(engine_name.to_string(), conv_vals);
            continue;
        }

        // General case: read tensor as flat f32
        let data = read_gguf_tensor_f32(gguf, &gguf_name)
            .map_err(|e| format!("{gguf_name}: {e}"))?;
        w.insert(engine_name.to_string(), data);
    }

    Ok(w)
}

/// Load non-layer weights (embed_tokens, norm, lm_head) from GGUF.
pub fn load_gguf_non_layer_weights(
    gguf: &GGUFile,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut w = HashMap::new();
    for (gguf_name, engine_name) in NON_LAYER_MAP {
        let data = read_gguf_tensor_f32(gguf, gguf_name)
            .map_err(|e| format!("{gguf_name}: {e}"))?;
        w.insert(engine_name.to_string(), data);
    }
    Ok(w)
}

// ── Config extraction from GGUF metadata ─────────────────────────────

/// Build an OrnithConfig from GGUF metadata + tensor shapes.
///
/// This avoids needing a separate config.json when loading from GGUF.
/// Uses metadata keys from llama.cpp's conversion script.
pub fn extract_ornith_config(gguf: &GGUFile) -> Result<OrnithConfig, String> {
    let meta = |key: &str| gguf.get_metadata_int(key);

    let prefix_meta = |keys: &[&str]| -> Option<i64> {
        for k in keys {
            if let Some(v) = meta(k) {
                return Some(v);
            }
        }
        None
    };

    let hidden_size = prefix_meta(&[
        "ornith.embedding_length", "qwen35.embedding_length",
        "llama.embedding_length",
    ]).unwrap_or(5120) as usize;

    let num_hidden_layers = prefix_meta(&[
        "ornith.block_count", "qwen35.block_count",
        "llama.block_count",
    ]).unwrap_or(32) as usize;

    let num_attention_heads = prefix_meta(&[
        "ornith.attention.head_count", "qwen35.attention.head_count",
        "llama.attention.head_count",
    ]).unwrap_or(40) as usize;

    let num_key_value_heads = prefix_meta(&[
        "ornith.attention.head_count_kv", "qwen35.attention.head_count_kv",
        "llama.attention.head_count_kv",
    ]).unwrap_or(num_attention_heads as i64) as usize;

    let intermediate_size = prefix_meta(&[
        "ornith.feed_forward_length", "qwen35.feed_forward_length",
        "llama.feed_forward_length",
    ]).unwrap_or(16384) as usize;

    let max_position_embeddings = prefix_meta(&[
        "ornith.context_length", "qwen35.context_length",
        "llama.context_length",
    ]).unwrap_or(32768) as usize;

    let vocab_size = prefix_meta(&["ornith.vocab_size", "qwen35.vocab_size"])
        .or_else(|| {
            // Read tokenizer.ggml.tokens array length directly
            gguf.metadata.get("tokenizer.ggml.tokens")
                .and_then(|v| {
                    if let crate::model::gguf::GGUFValue::Array(arr) = v {
                        Some(arr.len() as i64)
                    } else {
                        None
                    }
                })
        })
        .unwrap_or(248320) as usize;

    let rope_theta = gguf.get_metadata_f32("ornith.rope.freq_base")
        .or_else(|| gguf.get_metadata_f32("qwen35.rope.freq_base"))
        .or_else(|| gguf.get_metadata_f32("llama.rope.freq_base"))
        .unwrap_or(10000.0);

    let rms_norm_eps = gguf.get_metadata_f32("ornith.attention.layer_norm_rms_epsilon")
        .or_else(|| gguf.get_metadata_f32("qwen35.attention.layer_norm_rms_epsilon"))
        .or_else(|| gguf.get_metadata_f32("llama.attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-6);

    let head_dim = prefix_meta(&["ornith.attention.head_dim", "qwen35.attention.head_dim"])
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            // Infer from attn_k weight if available (K is pure projection, not fused with gate)
            let inferred = (0..num_hidden_layers).find_map(|i| {
                let name = format!("blk.{i}.attn_k.weight");
                gguf.get_tensor_info(&name)
                    .map(|t| t.dimensions[1] as usize / num_key_value_heads)
            });
            inferred.unwrap_or(hidden_size / num_attention_heads)
        });

    let rope_dim = prefix_meta(&[
        "ornith.rope.dimension_count", "qwen35.rope.dimension_count",
    ]).unwrap_or(head_dim as i64) as usize;

    let attention_interval = prefix_meta(&[
        "ornith.full_attention_interval", "qwen35.full_attention_interval",
    ]).unwrap_or(4) as usize;

    // ── Qwen35 SSM-specific parameters ────────────────────────────
    // llama.cpp stores these under qwen35.ssm.*.  Use them directly;
    // fall back to tensor-shape inference if absent.
    let ssm_meta = |suffix: &str| -> Option<i64> {
        prefix_meta(&[
            &format!("ornith.{}", suffix),
            &format!("qwen35.{}", suffix),
        ])
    };

    let state_size       = ssm_meta("ssm.state_size").unwrap_or(128) as usize;
    let conv_kernel      = ssm_meta("ssm.conv_kernel").unwrap_or(4) as usize;
    let group_count      = ssm_meta("ssm.group_count").unwrap_or(16) as usize;
    let _time_step_rank   = ssm_meta("ssm.time_step_rank").unwrap_or(32) as usize;
    let ssm_inner_size   = ssm_meta("ssm.inner_size").unwrap_or(hidden_size as i64) as usize;

    // Map llama.cpp SSM params → OrnithConfig fields
    let linear_key_head_dim   = state_size;
    let linear_conv_kernel_dim = conv_kernel;
    let linear_num_key_heads  = group_count;
    // ssm_inner_size = n_v * d_v, so n_v = ssm_inner_size / state_size
    let linear_num_value_heads = ssm_inner_size / state_size;
    let linear_value_head_dim  = state_size;

    // Build layer_types array from tensor presence (hybrid detection)
    let layer_types: Vec<String> = (0..num_hidden_layers)
        .map(|i| {
            let has_ssm = gguf.get_tensor_info(&format!("blk.{i}.ssm_alpha.weight")).is_some()
                || gguf.get_tensor_info(&format!("blk.{i}.ssm_out.weight")).is_some();
            if has_ssm {
                "linear_attention".to_string()
            } else {
                "full_attention".to_string()
            }
        })
        .collect();

    let mtp_num_hidden_layers = prefix_meta(&["ornith.nextn_predict_layers", "qwen35.nextn_predict_layers"])
        .unwrap_or(0) as usize;

    Ok(OrnithConfig {
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        intermediate_size,
        rms_norm_eps,
        vocab_size,
        rope_theta,
        partial_rotary_factor: rope_dim as f32 / head_dim as f32,
        linear_conv_kernel_dim,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
        layer_types,
        max_position_embeddings,
        mtp_num_hidden_layers,
        tie_word_embeddings: false,
    })
}

// ── Low-level tensor reading ─────────────────────────────────────────

pub fn read_gguf_tensor_f32(gguf: &GGUFile, name: &str) -> Result<Vec<f32>, GGUError> {
    let info = gguf
        .get_tensor_info(name)
        .ok_or_else(|| GGUError::MissingTensor(name.to_string()))?;

    let ndims = info.dimensions.len();
    let cols = info.dimensions[0] as usize;
    let rows = info.dimensions.get(1).copied().unwrap_or(1) as usize;

    if ndims > 2 {
        return Err(GGUError::UnsupportedQuantType(
            format!(">2D tensor: {name} with {ndims} dims"),
            info.typ,
        ));
    }

    let mut out = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        match gguf.get_tensor_row_f32(name, row) {
            Some(row_data) => out.extend_from_slice(&row_data),
            None => return Err(GGUError::MissingTensor(name.to_string())),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_a_log_inversion() {
        let raw_a_logs = vec![-2.656, -1.0, 0.0, -5.5, -10.0];
        let expected: Vec<f32> = raw_a_logs.clone();
        let ssm_a: Vec<f32> = raw_a_logs.iter().map(|&v| -v.exp()).collect();
        let recovered: Vec<f32> = ssm_a.iter().map(|&v| (-v).ln()).collect();
        for (i, (e, r)) in expected.iter().zip(&recovered).enumerate() {
            let diff = (e - r).abs();
            assert!(diff < 1e-4, "Mismatch at {i}: expected {e}, got {r}, diff {diff}");
        }
    }
}
