//! Ornith model config — parsed from HuggingFace config.json.
//!
//! Ornith-1.0-9B is a hybrid model with linear_attention (DeltaNet) and
//! full_attention (standard) layers mixed.  Layer types are specified
//! per-layer in config.json's `layer_types` array.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct OrnithConfig {
    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default)]
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(default)]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub vocab_size: usize,
    #[serde(default)]
    pub rope_theta: f32,
    #[serde(default)]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub linear_conv_kernel_dim: usize,
    #[serde(default)]
    pub linear_num_key_heads: usize,
    #[serde(default)]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub linear_key_head_dim: usize,
    #[serde(default)]
    pub linear_value_head_dim: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl OrnithConfig {
    /// Load from a HuggingFace config.json path.
    pub fn load(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("read {path}: {e}"))?;
        let root: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("parse json: {e}"))?;
        // Ornith nests most settings under .text_config
        let val = root
            .get("text_config")
            .cloned()
            .unwrap_or(root);
        let cfg: OrnithConfig = serde_json::from_value(val)
            .map_err(|e| format!("parse config: {e}"))?;
        // Sane defaults
        let mut c = cfg;
        if c.head_dim == 0 {
            c.head_dim = c.hidden_size / c.num_attention_heads;
        }
        if c.rms_norm_eps == 0.0 {
            c.rms_norm_eps = 1e-6;
        }
        if c.rope_theta == 0.0 {
            c.rope_theta = 10000.0;
        }
        Ok(c)
    }
}
