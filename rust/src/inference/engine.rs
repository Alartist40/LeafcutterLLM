//! Unified inference engine with native Qwen3.5 hybrid support
//!
//! Implements hybrid Transformer-Mamba forward pass:
//!   - Most layers: SSM (Mamba-style state space)
//!   - Every Nth layer (attention_interval): standard attention
//!   - Final layer: optional speculative decoding heads
//!
//! For standard architectures (Llama, Qwen2, Mistral), all layers use attention.

use crate::model::loader::{GGUFModel, ModelConfig};
use crate::model::tensor::Tensor;
use crate::cache::KVCache;
use crate::inference::attention::{attention_forward, AttentionParams};
use crate::inference::sampler::sample_top_p;
use crate::inference::ssm::{ssm_forward, SSMConfig};
use crate::inference::speculative::SpeculativeHead;
use std::collections::HashMap;

pub struct Engine {
    pub model: GGUFModel,
    pub config: ModelConfig,
    pub kv_cache: KVCache,
    pub special_weights: HashMap<String, Tensor>,
    pub attn_params: AttentionParams,
    pub ssm_config: SSMConfig,
    pub speculative_head: Option<SpeculativeHead>,
}

impl Engine {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model = GGUFModel::load(path)?;

        // Run corruption scan
        let corruption = crate::model::loader::scan_for_corruption(&model.file);
        if !corruption.is_clean() {
            eprintln!("\n{}", corruption.print());
        }

        // Run pre-flight capability report
        let report = model.capability_report();
        if !report.can_run {
            eprintln!("\n{}", report.print());
            return Err(format!(
                "Model cannot run: architecture={} unsupported_quant={} missing_tensors={}",
                report.architecture.name(),
                report.quant_summary.unsupported.len(),
                report.missing_tensors.len()
            ).into());
        }

        let config = model.config.clone();
        let special_weights = model.load_special()?;
        let kv_cache = KVCache::new(config.num_hidden_layers);

        // Build attention params with fused QKV / compressed KV support
        let attn_params = AttentionParams {
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            kv_head_dim: config.kv_head_dim,
            rope_theta: config.rope_theta,
            use_fused_qkv: report.uses_fused_qkv,
            use_gate: report.uses_ssm,
            ..Default::default()
        };

        // Build SSM config for hybrid architectures from model metadata
        let ssm_config = if report.uses_ssm {
            let get_meta = |keys: &[&str]| -> Option<i64> {
                for key in keys {
                    if let Some(v) = model.file.get_metadata_int(key) {
                        return Some(v);
                    }
                }
                None
            };
            SSMConfig {
                state_size: get_meta(&["qwen35.ssm.state_size", "ssm.state_size"]).map(|v| v as usize).unwrap_or(128),
                inner_size: get_meta(&["qwen35.ssm.inner_size", "ssm.inner_size"]).map(|v| v as usize).unwrap_or(config.intermediate_size),
                time_step_rank: get_meta(&["qwen35.ssm.time_step_rank", "ssm.time_step_rank"]).map(|v| v as usize).unwrap_or(32),
                conv_kernel: get_meta(&["qwen35.ssm.conv_kernel", "ssm.conv_kernel"]).map(|v| v as usize).unwrap_or(4),
                group_count: get_meta(&["qwen35.ssm.group_count", "ssm.group_count"]).map(|v| v as usize).unwrap_or(16),
            }
        } else {
            SSMConfig::default()
        };

        // Load speculative decoding head if present
        let speculative_head = if report.architecture == crate::model::arch::ModelArchitecture::Qwen35 {
            let mut draft_weights = HashMap::new();
            for tensor in &model.file.tensors {
                if tensor.name.starts_with("nextn") {
                    if let Some(raw) = model.file.get_tensor_raw(&tensor.name) {
                        if let Some(info) = model.file.get_tensor_info(&tensor.name) {
                            let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).rev().collect();
                            if let Ok(t) = crate::model::loader::GGUFModel::dequantize(raw, info.typ, shape) {
                                draft_weights.insert(tensor.name.clone(), t);
                            }
                        }
                    }
                }
            }
            SpeculativeHead::from_weights(&draft_weights, 4)
        } else {
            None
        };

        Ok(Self {
            model,
            config,
            kv_cache,
            special_weights,
            attn_params,
            ssm_config,
            speculative_head,
        })
    }

    pub fn generate(&mut self, tokens: &[usize], max_tokens: usize, temperature: f32, top_p: f32) -> Vec<usize> {
        self.kv_cache.clear();

        // Prefill
        let mut logits = self.forward(tokens);
        let mut next_token = sample_top_p(&logits, temperature, top_p);
        let mut generated = vec![next_token];

        if next_token == 2 {
            return generated;
        }

        // Decode loop
        for _ in 0..max_tokens - 1 {
            logits = self.forward(&[next_token]);
            next_token = sample_top_p(&logits, temperature, top_p);
            generated.push(next_token);

            if next_token == 2 {
                break;
            }
        }

        generated
    }

    /// Hybrid forward pass supporting both standard transformers and SSM/Transformer hybrids.
    pub fn forward(&mut self, tokens: &[usize]) -> Vec<f32> {
        let seq_len = tokens.len();

        // Embedding lookup
        let embed = self.special_weights.get("model.embed_tokens.weight")
            .expect("Missing embed_tokens");
        let mut hidden = self.embed_lookup(tokens, embed);

        // Transformer / hybrid layers
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer_weights = self.model.load_layer(layer_idx)
                .expect("Failed to load layer");

            // Detect layer type from actual tensor contents (most robust)
            let has_standard_attn = layer_weights.contains_key("self_attn.q_proj.weight");
            let has_ssm = layer_weights.contains_key("ssm_out.weight")
                || layer_weights.contains_key("ssm_alpha.weight");

            // Pre-norm
            let pre_norm_weight = layer_weights.get("input_layernorm.weight")
                .or_else(|| layer_weights.get("attn_norm.weight"))
                .expect("Missing pre-norm");
            let normed = hidden.rms_norm(pre_norm_weight, 1e-5);

            if has_standard_attn {
                // Standard attention layer
                let attn_out = attention_forward(&normed, &layer_weights, &self.attn_params, &mut self.kv_cache, layer_idx);
                hidden = hidden.add(&attn_out);
            } else if has_ssm {
                // SSM layer (Mamba-style)
                let ssm_out = ssm_forward(&normed, &layer_weights, &self.ssm_config);
                hidden = hidden.add(&ssm_out);
            }

            // Post-attention/SSM norm + FFN
            let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
                .or_else(|| layer_weights.get("ffn_norm.weight"))
                .expect("Missing post-norm");
            let normed = hidden.rms_norm(post_norm_weight, 1e-5);
            let ffn_out = self.ffn_forward(&normed, &layer_weights);
            hidden = hidden.add(&ffn_out);
        }

        // Final norm
        let final_norm = self.special_weights.get("model.norm.weight")
            .expect("Missing final norm");
        hidden = hidden.rms_norm(final_norm, 1e-5);

        // LM head
        let lm_head = self.special_weights.get("lm_head.weight")
            .expect("Missing lm_head");
        let logits = hidden.matmul(lm_head);

        // Return last token's logits
        let vocab_size = logits.shape[1];
        let start = (seq_len - 1) * vocab_size;
        logits.data[start..start + vocab_size].to_vec()
    }

    pub fn embed_lookup(&self, tokens: &[usize], embed: &Tensor) -> Tensor {
        let hidden_size = embed.shape[1];
        let mut data = vec![0.0f32; tokens.len() * hidden_size];
        for (i, &token) in tokens.iter().enumerate() {
            if token < embed.shape[0] {
                data[i * hidden_size..(i + 1) * hidden_size]
                    .copy_from_slice(&embed.data[token * hidden_size..(token + 1) * hidden_size]);
            }
        }
        Tensor::from_vec(data, vec![tokens.len(), hidden_size])
    }

    fn ffn_forward(&self, x: &Tensor, weights: &HashMap<String, Tensor>) -> Tensor {
        let gate = weights.get("mlp.gate_proj.weight").expect("Missing gate");
        let up = weights.get("mlp.up_proj.weight").expect("Missing up");
        let down = weights.get("mlp.down_proj.weight").expect("Missing down");

        let gate_proj = x.matmul(gate);
        let up_proj = x.matmul(up);
        let activated = gate_proj.silu();

        let mut fused = vec![0.0f32; activated.size()];
        for i in 0..activated.size() {
            fused[i] = activated.data[i] * up_proj.data[i];
        }
        let fused_tensor = Tensor::from_vec(fused, activated.shape.clone());

        fused_tensor.matmul(down)
    }

    /// Get engine info for diagnostics
    pub fn info(&self) -> EngineInfo {
        EngineInfo {
            architecture: self.model.architecture.name().to_string(),
            total_layers: self.config.num_hidden_layers,
            hidden_size: self.config.hidden_size,
            kv_cache_tokens: self.kv_cache.total_seq_len(),
            kv_cache_bytes: self.kv_cache.memory_bytes(),
            use_ssm: self.model.architecture.uses_ssm(),
            use_fused_qkv: self.attn_params.use_fused_qkv,
            use_compressed_kv: self.config.kv_head_dim != self.config.head_dim,
            use_speculative: self.speculative_head.is_some(),
        }
    }
}

#[derive(Debug)]
pub struct EngineInfo {
    pub architecture: String,
    pub total_layers: usize,
    pub hidden_size: usize,
    pub kv_cache_tokens: usize,
    pub kv_cache_bytes: usize,
    pub use_ssm: bool,
    pub use_fused_qkv: bool,
    pub use_compressed_kv: bool,
    pub use_speculative: bool,
}
