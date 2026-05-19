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
use crate::cache::{KVCache, ssm_state::SSMStateCache};
use crate::inference::attention::{attention_forward, AttentionParams};
use crate::inference::sampler::sample_top_p;
use crate::inference::ssm::{ssm_forward, SSMConfig};
use crate::inference::speculative::SpeculativeHead;
use rayon::prelude::*;
use std::collections::HashMap;

pub struct Engine {
    pub model: GGUFModel,
    pub config: ModelConfig,
    pub kv_cache: KVCache,
    pub special_weights: HashMap<String, Tensor>,
    pub attn_params: AttentionParams,
    pub ssm_config: SSMConfig,
    pub speculative_head: Option<SpeculativeHead>,
    /// Whether lm_head is tied to token embeddings (no separate output.weight tensor).
    lm_head_tied: bool,
    /// Layer weight cache: layer_idx -> weights HashMap.
    /// Avoids reloading layers from disk on every forward pass.
    pub layer_cache: HashMap<usize, HashMap<String, Tensor>>,
    /// SSM state cache: persistent hidden state for Mamba-style layers.
    pub ssm_cache: SSMStateCache,
    /// Current sequence position offset for RoPE. Tracks total tokens processed
    /// across forward calls within a generation session.
    seq_offset: usize,
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
        let mut special_weights = model.load_special()?;
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

        // Don't keep embed or lm_head in RAM — use mmap per-row lookup instead.
        special_weights.remove("model.embed_tokens.weight");
        special_weights.remove("lm_head.weight");
        let lm_head_tied = !model.file.get_tensor_info("output.weight").is_some();

        Ok(Self {
            model,
            config,
            kv_cache,
            special_weights,
            attn_params,
            ssm_config,
            speculative_head,
            lm_head_tied,
            layer_cache: HashMap::new(),
            ssm_cache: SSMStateCache::new(),
            seq_offset: 0,
        })
    }

    pub fn generate(&mut self, tokens: &[usize], max_tokens: usize, temperature: f32, top_p: f32) -> Vec<usize> {
        self.kv_cache.clear();
        self.ssm_cache.clear();
        self.seq_offset = 0;

        // Prefill
        let mut logits = self.forward(tokens);
        self.seq_offset = tokens.len();
        let mut next_token = sample_top_p(&logits, temperature, top_p);
        let mut generated = vec![next_token];

        if next_token == 2 {
            return generated;
        }

        // Decode loop
        for _ in 0..max_tokens - 1 {
            logits = self.forward(&[next_token]);
            self.seq_offset += 1;
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

        // Embedding lookup via mmap (avoids loading full embed matrix into RAM)
        let mut hidden = self.embed_lookup_mmap(tokens);

        // Transformer / hybrid layers
        for layer_idx in 0..self.config.num_hidden_layers {
            // Check layer cache first; load from disk only on miss
            if !self.layer_cache.contains_key(&layer_idx) {
                let weights = self.model.load_layer(layer_idx)
                    .expect("Failed to load layer");
                self.layer_cache.insert(layer_idx, weights);
            }
            let layer_weights = self.layer_cache.get(&layer_idx).unwrap();

            // Detect layer type from actual tensor contents (most robust)
            // Qwen3.5 uses attn_q.weight (unfused) or self_attn.q_proj.weight (Llama-style)
            let has_standard_attn = layer_weights.contains_key("self_attn.q_proj.weight")
                || layer_weights.contains_key("attn_q.weight");
            let has_ssm = layer_weights.contains_key("ssm_out.weight")
                || layer_weights.contains_key("ssm_alpha.weight");

            // Pre-norm
            let pre_norm_weight = layer_weights.get("input_layernorm.weight")
                .or_else(|| layer_weights.get("attn_norm.weight"))
                .expect("Missing pre-norm");
            let normed = hidden.rms_norm(pre_norm_weight, 1e-5);

            if has_standard_attn {
                // Standard attention layer
                let attn_out = attention_forward(&normed, layer_weights, &self.attn_params, &mut self.kv_cache, layer_idx, self.seq_offset);
                hidden = hidden.add(&attn_out);
            } else if has_ssm {
                // SSM layer (Mamba-style)
                let ssm_out = ssm_forward(&normed, layer_weights, &self.ssm_config, &mut self.ssm_cache, layer_idx);
                hidden = hidden.add(&ssm_out);
            }

            // Post-attention/SSM norm + FFN
            let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
                .or_else(|| layer_weights.get("ffn_norm.weight"))
                .expect("Missing post-norm");
            let normed = hidden.rms_norm(post_norm_weight, 1e-5);
            let ffn_out = Self::ffn_forward(&normed, layer_weights);
            hidden = hidden.add(&ffn_out);
        }

        // Final norm
        let final_norm = self.special_weights.get("model.norm.weight")
            .expect("Missing final norm");
        hidden = hidden.rms_norm(final_norm, 1e-5);

        // LM head — computed via outer-product over rows from mmap (no full matrix in RAM)
        if self.lm_head_tied {
            self.lm_head_tied_forward(&hidden, seq_len)
        } else {
            self.lm_head_separate_forward(&hidden, seq_len)
        }
    }

    /// Legacy embed lookup from a fully-materialized tensor (for tests / compatibility).
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

    /// Lookup embeddings via memory-mapped file (per-token, no full matrix in RAM).
    pub fn embed_lookup_mmap(&self, tokens: &[usize]) -> Tensor {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let mut data = vec![0.0f32; tokens.len() * hidden_size];
        for (i, &token) in tokens.iter().enumerate() {
            let idx = token.min(vocab_size - 1);
            let row = self.model.file.get_tensor_row_f32("token_embd.weight", idx)
                .expect("Failed to read embedding row");
            data[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(&row);
        }
        Tensor::from_vec(data, vec![tokens.len(), hidden_size])
    }

    /// Compute logits when lm_head is tied to embeddings.
    /// Parallelized over the vocabulary using rayon.
    fn lm_head_tied_forward(&self, hidden: &Tensor, seq_len: usize) -> Vec<f32> {
        self.lm_head_projection(&hidden.data[(seq_len - 1) * self.config.hidden_size..seq_len * self.config.hidden_size],
            "token_embd.weight", self.config.hidden_size, self.config.vocab_size)
    }

    /// Compute logits when lm_head is a separate tensor (output.weight).
    /// The raw GGUF data is [vocab, hidden] (before transpose), so each row
    /// corresponds to one vocab token's weights.  We compute dot(hidden, row_j)
    /// for each token j — same pattern as tied embeddings.
    fn lm_head_separate_forward(&self, hidden: &Tensor, seq_len: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let hidden_last = &hidden.data[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        self.lm_head_projection(hidden_last, "output.weight", hidden_size, vocab_size)
    }

    /// Generic lm_head projection: dot(hidden_last, embed_row) for each token.
    fn lm_head_projection(&self, hidden_last: &[f32], tensor_name: &str, _hidden_size: usize, vocab_size: usize) -> Vec<f32> {
        let file = &self.model.file;
        (0..vocab_size).into_par_iter().map(|token_id| {
            let row = file.get_tensor_row_f32(tensor_name, token_id)
                .expect("lm_head row");
            hidden_last.iter().zip(row.iter()).map(|(a, b)| a * b).sum::<f32>()
        }).collect()
    }

    fn ffn_forward(x: &Tensor, weights: &HashMap<String, Tensor>) -> Tensor {
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
