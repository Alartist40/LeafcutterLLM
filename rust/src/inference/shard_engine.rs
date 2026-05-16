//! ShardEngine — layer-by-layer inference with disk offloading
//!
//! Loads **one transformer layer at a time** from per-layer shard files,
//! computes, then immediately drops the layer weights.  Peak RAM is
//! ~1 layer + KV cache + activations + special weights.
//!
//! This engine sits alongside the original `Engine` (GGUF-native).
//! Use `Engine` for small models, `ShardEngine` for 70B+ models.

use crate::model::loader::ModelConfig;
use crate::model::tensor::Tensor;
use crate::cache::KVCache;
use crate::inference::attention::{attention_forward, AttentionParams};
use crate::inference::sampler::sample_top_p;
use crate::shard::{Manifest, ShardLoader};
use std::collections::HashMap;

pub struct ShardEngine {
    pub config: ModelConfig,
    pub kv_cache: KVCache,
    pub special_weights: HashMap<String, Tensor>,
    loader: ShardLoader,
}

impl ShardEngine {
    /// Load a model from a shard manifest.
    pub fn load(manifest_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_with_cache(manifest_path, 0)
    }

    /// Load with a layer cache.  `cache_slots` = number of recently-used
    /// layers to keep in RAM.  Set to `num_layers` to cache everything
    /// (fastest decode, highest RAM usage).
    pub fn load_with_cache(manifest_path: &str, cache_slots: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let manifest = Manifest::load(manifest_path)?;
        let loader = ShardLoader::from_manifest(manifest.clone())
            .with_cache_capacity(cache_slots);

        // Load special weights (kept permanently in RAM)
        let special_weights = loader.load_special()?;

        let config = ModelConfig {
            hidden_size: manifest.hidden_size,
            num_hidden_layers: manifest.num_layers,
            num_attention_heads: manifest.num_attention_heads,
            num_key_value_heads: manifest.num_key_value_heads,
            intermediate_size: manifest.intermediate_size,
            max_seq_len: manifest.max_seq_len,
            vocab_size: manifest.vocab_size,
            rope_theta: manifest.rope_theta,
        };

        let kv_cache = KVCache::new(config.num_hidden_layers);

        Ok(Self {
            config,
            kv_cache,
            special_weights,
            loader,
        })
    }

    /// Autoregressive generation.
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

    /// Forward pass through all layers, loading each from disk.
    pub fn forward(&mut self, tokens: &[usize]) -> Vec<f32> {
        let seq_len = tokens.len();

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

        // Transformer layers — load one at a time, compute, evict
        for layer_idx in 0..self.config.num_hidden_layers {
            // Try to use a prefetched layer first
            let layer_weights = if let Some(prefetched) = self.loader.take_prefetch() {
                prefetched
            } else {
                self.loader.load_layer(layer_idx)
                    .expect(&format!("Failed to load layer {}", layer_idx))
            };

            // Prefetch next layer while we compute this one
            if layer_idx + 1 < self.config.num_hidden_layers {
                self.loader.prefetch_layer(layer_idx + 1);
            }

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

            // Optionally cache layer for reuse (e.g., during decode)
            self.loader.cache_layer(layer_idx, layer_weights);
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

        let mut fused = vec![0.0f32; activated.size()];
        for i in 0..activated.size() {
            fused[i] = activated.data[i] * up_proj.data[i];
        }
        let fused_tensor = Tensor::from_vec(fused, activated.shape.clone());

        fused_tensor.matmul(down)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard::writer::ShardWriter;
    use crate::shard::loader::{Manifest, LayerManifest, SpecialManifest};
    use crate::model::loader::ModelConfig;
    use std::collections::HashMap;

    fn create_identity_weights(hidden: usize, intermediate: usize) -> HashMap<String, Tensor> {
        let mut w = HashMap::new();

        // Attention: identity projections (Q=K=V=O=I) → attention becomes a no-op
        let eye_h = identity_matrix(hidden, hidden);
        w.insert("self_attn.q_proj.weight".to_string(), eye_h.clone());
        w.insert("self_attn.k_proj.weight".to_string(), eye_h.clone());
        w.insert("self_attn.v_proj.weight".to_string(), eye_h.clone());
        w.insert("self_attn.o_proj.weight".to_string(), eye_h.clone());

        // FFN: gate=0, up=0, down=0 → FFN outputs zero
        // Shapes are engine-ready (post-transpose from GGUF):
        // gate/up: [hidden, intermediate], down: [intermediate, hidden]
        w.insert("mlp.gate_proj.weight".to_string(), zeros_matrix(hidden, intermediate));
        w.insert("mlp.up_proj.weight".to_string(), zeros_matrix(hidden, intermediate));
        w.insert("mlp.down_proj.weight".to_string(), zeros_matrix(intermediate, hidden));

        // Norms: weight=1
        w.insert("input_layernorm.weight".to_string(), ones_vector(hidden));
        w.insert("post_attention_layernorm.weight".to_string(), ones_vector(hidden));

        w
    }

    fn identity_matrix(rows: usize, cols: usize) -> Tensor {
        let mut data = vec![0.0f32; rows * cols];
        let n = rows.min(cols);
        for i in 0..n {
            data[i * cols + i] = 1.0;
        }
        Tensor::from_vec(data, vec![rows, cols])
    }

    fn zeros_matrix(rows: usize, cols: usize) -> Tensor {
        Tensor::from_vec(vec![0.0f32; rows * cols], vec![rows, cols])
    }

    fn ones_vector(len: usize) -> Tensor {
        Tensor::from_vec(vec![1.0f32; len], vec![len])
    }

    #[test]
    fn test_shard_engine_forward() {
        let dir = tempfile::tempdir().unwrap();
        let output_dir = dir.path().to_str().unwrap();

        let hidden = 8;
        let intermediate = 16;
        let vocab = 10;
        let num_layers = 2;

        let config = ModelConfig {
            hidden_size: hidden,
            num_hidden_layers: num_layers,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            intermediate_size: intermediate,
            max_seq_len: 128,
            vocab_size: vocab,
            rope_theta: 10000.0,
        };

        let writer = ShardWriter::new(config.clone(), output_dir);

        // Write layer shards
        let mut layer_files = Vec::new();
        for i in 0..num_layers {
            let weights = create_identity_weights(hidden, intermediate);
            let (filename, size) = writer.write_layer_shard(i, &weights).unwrap();
            layer_files.push(LayerManifest { idx: i, file: filename, size });
        }

        // Write special shards
        let mut embed = HashMap::new();
        // Simple embedding: token i = one-hot-ish vector with i in first position
        let mut embed_data = vec![0.0f32; vocab * hidden];
        for i in 0..vocab {
            embed_data[i * hidden] = i as f32;
        }
        embed.insert("model.embed_tokens.weight".to_string(), Tensor::from_vec(embed_data, vec![vocab, hidden]));
        let (embed_file, embed_size) = writer.write_special_shard("embed", &embed).unwrap();

        let mut norm = HashMap::new();
        norm.insert("model.norm.weight".to_string(), ones_vector(hidden));
        let (norm_file, norm_size) = writer.write_special_shard("norm", &norm).unwrap();

        let mut lm_head = HashMap::new();
        // LM head: map hidden back to vocab (just read first element)
        let mut lm_data = vec![0.0f32; hidden * vocab];
        for i in 0..vocab.min(hidden) {
            lm_data[i * vocab + i] = 1.0;
        }
        lm_head.insert("lm_head.weight".to_string(), Tensor::from_vec(lm_data, vec![hidden, vocab]));
        let (lm_head_file, lm_head_size) = writer.write_special_shard("lm_head", &lm_head).unwrap();

        // Write manifest
        let manifest = Manifest {
            model: "test".to_string(),
            num_layers,
            hidden_size: hidden,
            vocab_size: vocab,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            intermediate_size: intermediate,
            max_seq_len: 128,
            rope_theta: 10000.0,
            shard_dir: output_dir.to_string(),
            layers: layer_files,
            special: SpecialManifest {
                embed: embed_file,
                embed_size,
                norm: norm_file,
                norm_size,
                lm_head: lm_head_file,
                lm_head_size,
            },
        };

        let manifest_path = dir.path().join("manifest.json");
        std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

        // Load engine and run forward
        let mut engine = ShardEngine::load(manifest_path.to_str().unwrap()).unwrap();
        let logits = engine.forward(&[3]); // token 3

        assert_eq!(logits.len(), vocab);
        // Forward pass completed successfully — all values should be finite
        assert!(logits.iter().all(|v| v.is_finite()), "Non-finite values in logits: {:?}", logits);
    }

    /// Same as test_shard_engine_forward but with Q8_0 quantized shards.
    /// Dimensions must be multiples of 32 for native INT8 GEMM.
    #[test]
    fn test_shard_engine_forward_q8_0() {
        let dir = tempfile::tempdir().unwrap();
        let output_dir = dir.path().to_str().unwrap();

        let hidden = 32;
        let intermediate = 64;
        let vocab = 32;
        let num_layers = 2;

        let config = ModelConfig {
            hidden_size: hidden,
            num_hidden_layers: num_layers,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            intermediate_size: intermediate,
            max_seq_len: 128,
            vocab_size: vocab,
            rope_theta: 10000.0,
        };

        let writer = ShardWriter::with_quant(config.clone(), output_dir, crate::shard::QuantFormat::Q8_0);

        // Write layer shards
        let mut layer_files = Vec::new();
        for i in 0..num_layers {
            let weights = create_identity_weights(hidden, intermediate);
            let (filename, size) = writer.write_layer_shard(i, &weights).unwrap();
            layer_files.push(LayerManifest { idx: i, file: filename, size });
        }

        // Write special shards
        let mut embed = HashMap::new();
        let mut embed_data = vec![0.0f32; vocab * hidden];
        for i in 0..vocab {
            embed_data[i * hidden] = i as f32;
        }
        embed.insert("model.embed_tokens.weight".to_string(), Tensor::from_vec(embed_data, vec![vocab, hidden]));
        let (embed_file, embed_size) = writer.write_special_shard("embed", &embed).unwrap();

        let mut norm = HashMap::new();
        norm.insert("model.norm.weight".to_string(), ones_vector(hidden));
        let (norm_file, norm_size) = writer.write_special_shard("norm", &norm).unwrap();

        let mut lm_head = HashMap::new();
        let mut lm_data = vec![0.0f32; hidden * vocab];
        for i in 0..vocab.min(hidden) {
            lm_data[i * vocab + i] = 1.0;
        }
        lm_head.insert("lm_head.weight".to_string(), Tensor::from_vec(lm_data, vec![hidden, vocab]));
        let (lm_head_file, lm_head_size) = writer.write_special_shard("lm_head", &lm_head).unwrap();

        let manifest = Manifest {
            model: "test-q8".to_string(),
            num_layers,
            hidden_size: hidden,
            vocab_size: vocab,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            intermediate_size: intermediate,
            max_seq_len: 128,
            rope_theta: 10000.0,
            shard_dir: output_dir.to_string(),
            layers: layer_files,
            special: SpecialManifest {
                embed: embed_file,
                embed_size,
                norm: norm_file,
                norm_size,
                lm_head: lm_head_file,
                lm_head_size,
            },
        };

        let manifest_path = dir.path().join("manifest.json");
        std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

        // Load engine and run forward
        let mut engine = ShardEngine::load(manifest_path.to_str().unwrap()).unwrap();
        let logits = engine.forward(&[3]);

        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|v| v.is_finite()), "Non-finite values in Q8_0 logits: {:?}", logits);

        // Verify that loaded weight tensors actually carry Q8_0 metadata
        let layer0 = engine.loader.load_layer(0).unwrap();
        let q_proj = layer0.get("self_attn.q_proj.weight").unwrap();
        assert!(q_proj.has_q8_data(), "Q8_0 shard should load weights with Q8_0 metadata");
    }
}
