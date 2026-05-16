//! Inference engine — autoregressive token generation

use crate::model::loader::{GGUFModel, ModelConfig};
use crate::model::tensor::Tensor;
use crate::cache::KVCache;
use crate::inference::attention::{attention_forward, AttentionParams};
use crate::inference::sampler::sample_top_p;
use std::collections::HashMap;

pub struct Engine {
    pub model: GGUFModel,
    pub config: ModelConfig,
    pub kv_cache: KVCache,
    pub special_weights: HashMap<String, Tensor>,
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

        Ok(Self {
            model,
            config,
            kv_cache,
            special_weights,
        })
    }

    pub fn generate(&mut self, tokens: &[usize], max_tokens: usize, temperature: f32, top_p: f32) -> Vec<usize> {
        self.kv_cache.clear();

        // Prefill
        let mut logits = self.forward(tokens);
        let mut next_token = sample_top_p(&logits, temperature, top_p);
        let mut generated = vec![next_token];

        if next_token == 2 {
            // EOS
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

    pub fn forward(&mut self, tokens: &[usize]) -> Vec<f32> {
        let seq_len = tokens.len();
        let _hidden_size = self.config.hidden_size;

        // Embedding lookup
        let embed = self.special_weights.get("model.embed_tokens.weight")
            .expect("Missing embed_tokens");
        let mut hidden = self.embed_lookup(tokens, embed);

        let attn_params = AttentionParams {
            num_heads: self.config.num_attention_heads,
            num_kv_heads: self.config.num_key_value_heads,
            head_dim: self.config.hidden_size / self.config.num_attention_heads,
            rope_theta: self.config.rope_theta,
        };

        // Transformer layers
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer_weights = self.model.load_layer(layer_idx)
                .expect("Failed to load layer");

            // Pre-attention RMSNorm
            let pre_norm_weight = layer_weights.get("input_layernorm.weight")
                .expect("Missing pre-norm");
            let normed = hidden.rms_norm(pre_norm_weight, 1e-5);

            // Attention
            let attn_out = attention_forward(&normed, &layer_weights, &attn_params, &mut self.kv_cache, layer_idx);
            hidden = hidden.add(&attn_out);

            // Post-attention RMSNorm
            let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
                .expect("Missing post-norm");
            let normed = hidden.rms_norm(post_norm_weight, 1e-5);

            // FFN
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

        // SiLU(gate) * up
        let gate_proj = x.matmul(gate);
        let up_proj = x.matmul(up);
        let activated = gate_proj.silu();

        // Element-wise multiply
        let mut fused = vec![0.0f32; activated.size()];
        for i in 0..activated.size() {
            fused[i] = activated.data[i] * up_proj.data[i];
        }
        let fused_tensor = Tensor::from_vec(fused, activated.shape.clone());

        fused_tensor.matmul(down)
    }
}
