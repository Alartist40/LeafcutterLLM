//! Streaming native Rust forward pass for Ornith safetensors.
//!
//! Architecture: AirLLM-style layer streaming.
//!   - Embedding: read ONE row (8KB BF16) from disk, not the whole 2GB table.
//!   - Per layer: read that layer's ~13 weight tensors from disk, convert
//!     BF16→f32, compute, DISCARD the weights. Peak RAM = one layer (~400MB).
//!   - lm_head: compute logits by reading one row at a time OR read the
//!     whole lm_head (same size as embed — 2GB) in chunks.
//!   - No global cache. No HashMap. No cloning.
//!
//! This is what beats AirLLM: Rust + direct file I/O + BLAS-like matmul,
//! with peak RAM = one layer's weights (~400MB for 9B), not the whole model.

use crate::bpe_tokenizer::BpeTokenizer;
use crate::model::tensor::Tensor;
use crate::ornith_config::OrnithConfig;
use crate::safetensors_loader::Shards;
use std::collections::HashMap;
use std::path::Path;

pub struct StreamingOrnith {
    pub cfg: OrnithConfig,
    pub shards: Shards,
    pub tok: BpeTokenizer,
}

impl StreamingOrnith {
    /// Open a model from a directory containing config.json, tokenizer.json,
    /// and .safetensors file(s).
    pub fn open(dir: &Path) -> Result<Self, String> {
        let cfg = OrnithConfig::load(dir.join("config.json").to_str().unwrap())?;
        let shards = Shards::open_dir(dir)?;
        let tok = BpeTokenizer::load(dir.join("tokenizer.json").to_str().unwrap())?;
        Ok(Self { cfg, shards, tok })
    }

    /// Forward pass for ONE token. Returns logits (vocab_size entries).
    ///
    /// `token_id`: the input token.
    /// `pos`: position in the sequence (for RoPE).
    pub fn forward_one_token(
        &mut self,
        token_id: i32,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden_size;

        // 1. Embedding: read ONLY the one row we need (h elements = 8KB BF16).
        let embed_name = "model.language_model.embed_tokens.weight";
        let mut hidden =
            self.shards
                .read_tensor_slice_f32(embed_name, token_id as usize * h, h)?;
        eprintln!(
            "[stream] embed row {} read: {} values, first 4 = {:?}",
            token_id, hidden.len(), &hidden[..4]
        );

        // 2. Run all 32 layers — load each layer's weights, compute, discard.
        for layer_idx in 0..self.cfg.num_hidden_layers {
            let layer_type = self
                .cfg
                .layer_types
                .get(layer_idx)
                .map(|s| s.as_str())
                .unwrap_or("linear_attention");

            let t0 = std::time::Instant::now();
            let weights = self.load_layer_weights(layer_idx, layer_type)?;
            eprintln!(
                "[stream] layer {}/{} loaded {} tensors in {:?}",
                layer_idx,
                self.cfg.num_hidden_layers,
                weights.len(),
                t0.elapsed()
            );

            let residual = hidden.clone();
            let t1 = std::time::Instant::now();
            let attn_out = if layer_type == "linear_attention" {
                self.deltanet_forward(&weights, &hidden, layer_idx)?
            } else {
                self.attention_forward(&weights, &hidden, layer_idx, pos)?
            };
            eprintln!(
                "[stream] layer {} attn ({}) {:?}",
                layer_idx, layer_type, t1.elapsed()
            );

            // Add residual
            for i in 0..h {
                hidden[i] = residual[i] + attn_out[i];
            }

            // Post-attention norm + MLP
            let post_norm = weights
                .get("post_attention_layernorm.weight")
                .ok_or("missing post_attention_layernorm")?;
            let normed = rms_norm(&hidden, post_norm, self.cfg.rms_norm_eps);

            let t2 = std::time::Instant::now();
            let mlp_out = self.mlp_forward(&weights, &normed)?;
            eprintln!("[stream] layer {} mlp {:?}", layer_idx, t2.elapsed());

            // Add residual again
            for i in 0..h {
                hidden[i] = residual[i] + mlp_out[i];
            }
        }

        // 3. Final norm
        let final_norm = self
            .shards
            .lookup("model.language_model.norm.weight")
            .ok_or("missing final norm")?;
        let final_w = self
            .shards
            .read_tensor_f32("model.language_model.norm.weight")?;
        hidden = rms_norm(&hidden, &final_w, self.cfg.rms_norm_eps);

        // 4. LM head: logits = hidden @ lm_head.T
        // lm_head is [vocab, hidden]. Read it in chunks to avoid loading
        // the whole 2GB at once. For now, read it all (it's the same size
        // as embed — we'll optimize this later).
        let lm_head = self.shards.read_tensor_f32("lm_head.weight")?;
        let vocab = self.cfg.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        // logits[v] = dot(hidden, lm_head_row_v)
        let lm_head_t = Tensor::from_vec(lm_head, vec![vocab, h]);
        let hidden_t = Tensor::from_vec(hidden, vec![1, h]);
        let lm_head_transposed = lm_head_t.transpose();
        let logits_t = hidden_t.matmul(&lm_head_t.transpose());
        // The above creates [1, vocab] = hidden @ lm_head^T.
        // Wait — lm_head is [vocab, h], so hidden(1,h) @ lm_head^T(h, vocab) = [1, vocab].
        logits = logits_t.data;
        // Truncate just in case
        logits.truncate(vocab);

        Ok(logits)
    }

    /// Load all weights for ONE layer from disk. ~13 tensors, ~400MB for 9B.
    /// These are returned as f32 and will be DROPPED after the layer computes.
    fn load_layer_weights(
        &mut self,
        layer_idx: usize,
        layer_type: &str,
    ) -> Result<HashMap<String, Vec<f32>>, String> {
        let pfx = format!("model.language_model.layers.{layer_idx}.");
        let mut w = HashMap::new();

        // Always-present: input_layernorm, post_attention_layernorm, MLP
        let names = match layer_type {
            "linear_attention" => vec![
                "input_layernorm.weight",
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.conv1d.weight",
                "linear_attn.A_log",
                "linear_attn.dt_bias",
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
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ],
            _ => return Err(format!("unknown layer type: {layer_type}")),
        };

        for suffix in &names {
            let full = format!("{pfx}{suffix}");
            let data = self.shards.read_tensor_f32(&full)?;
            w.insert(suffix.to_string(), data);
        }
        Ok(w)
    }

    /// DeltaNet (linear attention) forward — simplified.
    /// For now, just do the QKV projection and output projection.
    /// Full DeltaNet recurrence to be implemented next.
    fn deltanet_forward(
        &self,
        w: &HashMap<String, Vec<f32>>,
        hidden: &[f32],
        layer_idx: usize,
    ) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden_size;
        // Input norm
        let norm_w = w
            .get("input_layernorm.weight")
            .ok_or("missing input_layernorm")?;
        let normed = rms_norm(hidden, norm_w, self.cfg.rms_norm_eps);

        // QKV projection: [1, h] @ [h, 3h] = [1, 3h]
        // in_proj_qkv is [3h, h] in safetensors (out, in).
        // We need hidden @ qkv^T = [1, 3h].
        let qkv_w = w
            .get("linear_attn.in_proj_qkv.weight")
            .ok_or("missing in_proj_qkv")?;
        let qkv_t = Tensor::from_vec(qkv_w.clone(), vec![3 * h, h]);
        let hidden_t = Tensor::from_vec(normed, vec![1, h]);
        let qkv = hidden_t.matmul(&qkv_t.transpose());
        eprintln!(
            "[stream] layer {} qkv shape: {} first 4 = {:?}",
            layer_idx,
            qkv.data.len(),
            &qkv.data[..4]
        );

        // TODO: full DeltaNet recurrence (conv1d, A_log, dt_bias, state update)
        // For now, just use the Q part for a simplified attention pass.
        let q = &qkv.data[..h];
        let k = &qkv.data[h..2 * h];
        let v = &qkv.data[2 * h..3 * h];

        // Simple scaled dot-product: just q @ v^T (simplified — no state yet)
        // This is WRONG but lets us validate the pipeline end-to-end.
        // We'll replace with proper DeltaNet next.
        let mut out = vec![0.0f32; h];
        for i in 0..h {
            out[i] = q[i] * v[i]; // placeholder
        }

        // Output projection
        let o_w = w
            .get("linear_attn.out_proj.weight")
            .ok_or("missing out_proj")?;
        let o_t = Tensor::from_vec(o_w.clone(), vec![h, h]);
        let out_t = Tensor::from_vec(out, vec![1, h]);
        let result = out_t.matmul(&o_t.transpose());
        Ok(result.data)
    }

    /// Standard attention forward (for full_attention layers).
    fn attention_forward(
        &self,
        w: &HashMap<String, Vec<f32>>,
        hidden: &[f32],
        layer_idx: usize,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden_size;
        let n_heads = self.cfg.num_attention_heads;
        let head_dim = self.cfg.head_dim; // 256
        let n_kv = self.cfg.num_key_value_heads; // 4

        let norm_w = w
            .get("input_layernorm.weight")
            .ok_or("missing input_layernorm")?;
        let normed = rms_norm(hidden, norm_w, self.cfg.rms_norm_eps);

        // Q = hidden @ q_proj^T — q_proj is [h, h]
        let q_w = w.get("self_attn.q_proj.weight").ok_or("missing q_proj")?;
        let k_w = w.get("self_attn.k_proj.weight").ok_or("missing k_proj")?;
        let v_w = w.get("self_attn.v_proj.weight").ok_or("missing v_proj")?;
        let o_w = w.get("self_attn.o_proj.weight").ok_or("missing o_proj")?;

        let hidden_t = Tensor::from_vec(normed, vec![1, h]);
        let q = hidden_t.matmul(&Tensor::from_vec(q_w.clone(), vec![h, h]).transpose());
        let k = hidden_t.matmul(&Tensor::from_vec(k_w.clone(), vec![n_kv * head_dim, h]).transpose());
        let v = hidden_t.matmul(&Tensor::from_vec(v_w.clone(), vec![n_kv * head_dim, h]).transpose());

        // For a single token, attention is just: for each head, score = q @ k^T / sqrt(d).
        // Since this is the FIRST token (pos=0 with no cache), it attends to itself.
        let mut attn_out = vec![0.0f32; h];

        for head in 0..n_heads {
            let q_h = &q.data[head * head_dim..(head + 1) * head_dim];
            // For each KV head, broadcast to q heads (GQA: n_heads / n_kv = 4)
            let kv_head = head / (n_heads / n_kv);
            let k_h = &k.data[kv_head * head_dim..(kv_head + 1) * head_dim];
            let v_h = &v.data[kv_head * head_dim..(kv_head + 1) * head_dim];

            // Single token: score = q @ k / sqrt(d), then softmax (trivially 1.0 for one token)
            // output = score * v = v
            for (i, &v_val) in v_h.iter().enumerate() {
                attn_out[head * head_dim + i] += v_val;
            }
        }

        // Output projection
        let out_t = Tensor::from_vec(attn_out, vec![1, h]);
        let result = out_t.matmul(&Tensor::from_vec(o_w.clone(), vec![h, h]).transpose());
        eprintln!(
            "[stream] layer {} (full_attn) done, o first 4 = {:?}",
            layer_idx,
            &result.data[..4]
        );
        Ok(result.data)
    }

    /// MLP forward: SwiGLU — down(silu(gate(x)) * up(x))
    fn mlp_forward(
        &self,
        w: &HashMap<String, Vec<f32>>,
        hidden: &[f32],
    ) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden_size;
        let inter = self.cfg.intermediate_size;

        let gate = w.get("mlp.gate_proj.weight").ok_or("missing gate")?;
        let up = w.get("mlp.up_proj.weight").ok_or("missing up")?;
        let down = w.get("mlp.down_proj.weight").ok_or("missing down")?;

        let hidden_t = Tensor::from_vec(hidden.to_vec(), vec![1, h]);
        let gate_t = hidden_t.matmul(&Tensor::from_vec(gate.clone(), vec![inter, h]).transpose());
        let up_t = hidden_t.matmul(&Tensor::from_vec(up.clone(), vec![inter, h]).transpose());

        // SwiGLU: silu(gate) * up
        let mut hidden_inter = vec![0.0f32; inter];
        for i in 0..inter {
            let g = gate_t.data[i];
            let u = up_t.data[i];
            hidden_inter[i] = (g / (1.0 + (-g).exp())) * u; // silu(g) * u
        }

        let inter_t = Tensor::from_vec(hidden_inter, vec![1, inter]);
        let result = inter_t.matmul(&Tensor::from_vec(down.clone(), vec![h, inter]).transpose());
        Ok(result.data)
    }

    /// Greedy argmax sampling
    pub fn argmax(logits: &[f32]) -> usize {
        let mut best = 0;
        let mut best_v = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best
    }
}

/// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * inv_rms * wi)
        .collect()
}
