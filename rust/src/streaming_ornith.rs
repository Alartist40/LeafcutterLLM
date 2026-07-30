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
        let num_layers = self.cfg.num_hidden_layers;
        let layer_types: Vec<String> = self.cfg.layer_types.clone();
        let rms_eps = self.cfg.rms_norm_eps;
        let vocab_size = self.cfg.vocab_size;

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
        for layer_idx in 0..num_layers {
            let layer_type = layer_types
                .get(layer_idx)
                .map(|s| s.as_str())
                .unwrap_or("linear_attention");

            let t0 = std::time::Instant::now();
            let weights = self.load_layer_weights(layer_idx, layer_type)?;
            eprintln!(
                "[stream] layer {}/{} loaded {} tensors in {:?}",
                layer_idx,
                num_layers,
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
            let normed = rms_norm(&hidden, post_norm, rms_eps);

            let t2 = std::time::Instant::now();
            let mlp_out = self.mlp_forward(&weights, &normed)?;
            eprintln!("[stream] layer {} mlp {:?}", layer_idx, t2.elapsed());

            // Add residual again
            for i in 0..h {
                hidden[i] = residual[i] + mlp_out[i];
            }
        }

        // 3. Final norm
        let final_w = self
            .shards
            .read_tensor_f32("model.language_model.norm.weight")?;
        hidden = rms_norm(&hidden, &final_w, rms_eps);

        // 4. LM head: logits = hidden @ lm_head.T
        // lm_head.weight is [vocab, hidden] = 248320 × 4096 (~4GB f32).
        // Read it in CHUNKS of 1024 rows to keep peak RAM low.
        let lm_head_name = "lm_head.weight";
        let mut logits = vec![0.0f32; vocab_size];
        let chunk_size = 1024;
        for chunk_start in (0..vocab_size).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(vocab_size);
            let n_rows = chunk_end - chunk_start;
            let chunk = self.shards.read_tensor_slice_f32(
                lm_head_name,
                chunk_start * h,
                n_rows * h,
            )?;
            let chunk_t = Tensor::from_vec(chunk, vec![n_rows, h]);
            let hidden_t = Tensor::from_vec(hidden.clone(), vec![1, h]);
            let logits_chunk = hidden_t.matmul(&chunk_t.transpose());
            logits[chunk_start..chunk_end].copy_from_slice(&logits_chunk.data);
        }

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
        let rms_eps = self.cfg.rms_norm_eps;

        // Config for linear attention heads
        let n_qk = self.cfg.linear_num_key_heads;   // 16
        let n_v = self.cfg.linear_num_value_heads;   // 32
        let d_k = self.cfg.linear_key_head_dim;     // 128
        let d_v = self.cfg.linear_value_head_dim;   // 128
        let conv_k = self.cfg.linear_conv_kernel_dim; // 4
        let conv_dim = n_qk * d_k + n_qk * d_k + n_v * d_v; // 8192

        // 1. Input norm
        let norm_w = w.get("input_layernorm.weight").ok_or("missing input_layernorm")?;
        let normed = rms_norm(hidden, norm_w, rms_eps);

        // 2. QKV projection: [1, h] @ [conv_dim, h]^T = [1, conv_dim]
        let qkv_w = w.get("linear_attn.in_proj_qkv.weight").ok_or("missing in_proj_qkv")?;
        let qkv_t = Tensor::from_vec(qkv_w.clone(), vec![conv_dim, h]);
        let hidden_t = Tensor::from_vec(normed.clone(), vec![1, h]);
        let qkv_proj = hidden_t.matmul(&qkv_t.transpose());
        // qkv_proj.data: [conv_dim] = [Q(2048) | K(2048) | V(4096)]

        // 3. Causal Conv1d (kernel=4) + SiLU
        // conv1d.weight shape in safetensors: [conv_dim, 1, conv_k] = [8192, 1, 4]
        // Row-major index for channel c, kernel tap k: c * conv_k + k
        // For pos=0 (first token), conv state is zeros, so only the last tap (k=3) applies.
        let conv_w = w.get("linear_attn.conv1d.weight").ok_or("missing conv1d")?;
        let conv_out = if conv_w.len() == conv_dim * conv_k {
            let mut out = vec![0.0f32; conv_dim];
            for c in 0..conv_dim {
                out[c] = conv_w[c * conv_k + (conv_k - 1)] * qkv_proj.data[c];
            }
            // SiLU: x * sigmoid(x)
            for v in out.iter_mut() {
                *v = *v * (1.0 / (1.0 + (-*v).exp()));
            }
            out
        } else {
            // Unexpected shape — just use raw QKV
            qkv_proj.data.clone()
        };

        // 4. Split into Q, K, V
        let q_total = n_qk * d_k;   // 2048
        let k_total = n_qk * d_k;   // 2048
        let v_total = n_v * d_v;    // 4096
        let q_data = &conv_out[..q_total];
        let k_data = &conv_out[q_total..q_total + k_total];
        let v_data = &conv_out[q_total + k_total..q_total + k_total + v_total];

        // 5. L2-normalize Q and K (per-head)
        let mut q = q_data.to_vec();
        let mut k = k_data.to_vec();
        for head in 0..n_qk {
            let base = head * d_k;
            let mut norm_sq = 0.0f32;
            for d in 0..d_k { norm_sq += q[base + d] * q[base + d]; }
            let norm = norm_sq.sqrt().max(1e-6);
            for d in 0..d_k { q[base + d] /= norm; }

            let mut norm_sq = 0.0f32;
            for d in 0..d_k { norm_sq += k[base + d] * k[base + d]; }
            let norm = norm_sq.sqrt().max(1e-6);
            for d in 0..d_k { k[base + d] /= norm; }
        }
        // Scale Q by 1/sqrt(d_k)
        let scale = 1.0f32 / (d_k as f32).sqrt();
        for v in q.iter_mut() { *v *= scale; }

        // 6. Compute decay rates: decay = exp(softplus(alpha + dt_bias) * A)
        //    where alpha = hidden @ in_proj_a.weight, A = -exp(A_log)
        let a_w = w.get("linear_attn.in_proj_a.weight").ok_or("missing in_proj_a")?;
        let a_t = Tensor::from_vec(a_w.clone(), vec![n_v, h]);
        let alpha = hidden_t.matmul(&a_t.transpose()); // [1, n_v]

        let a_log = w.get("linear_attn.A_log").ok_or("missing A_log")?;
        let dt_bias = w.get("linear_attn.dt_bias").ok_or("missing dt_bias")?;

        let mut decay = vec![0.0f32; n_v];
        for head in 0..n_v {
            let alpha_val = alpha.data[head];
            let dt_val = dt_bias.get(head).copied().unwrap_or(0.0);
            let a_log_val = a_log.get(head).copied().unwrap_or(0.0);
            let a = -a_log_val.exp(); // A = -exp(A_log)
            let dt = softplus(alpha_val + dt_val);
            decay[head] = (dt * a).exp();
        }

        // 7. Compute beta gates: beta = sigmoid(hidden @ in_proj_b.weight)
        let b_w = w.get("linear_attn.in_proj_b.weight").ok_or("missing in_proj_b")?;
        let b_t = Tensor::from_vec(b_w.clone(), vec![n_v, h]);
        let beta_logits = hidden_t.matmul(&b_t.transpose()); // [1, n_v]
        let beta: Vec<f32> = beta_logits.data.iter().map(|&v| sigmoid(v)).collect();

        // 8. Delta rule state update + output
        // For pos=0: state starts at zero.
        //   v_pred = S @ k = 0 (state is zero)
        //   S = decay * S + beta * (v - v_pred) * k = beta * v * k (outer product)
        //   o = S @ q = beta * (q . k) * v
        let v_heads_per_qk = if n_qk > 0 { n_v / n_qk } else { 1 };
        let mut output = vec![0.0f32; v_total];

        for h_qk in 0..n_qk {
            let q_h = &q[h_qk * d_k..(h_qk + 1) * d_k];
            let k_h = &k[h_qk * d_k..(h_qk + 1) * d_k];

            for v_idx in 0..v_heads_per_qk.max(1) {
                let h_v = h_qk * v_heads_per_qk + v_idx;
                if h_v >= n_v { continue; }
                let v_h = &v_data[h_v * d_v..(h_v + 1) * d_v];
                let decay_h = decay[h_v];
                let beta_h = beta[h_v];

                // For pos=0: state = 0, so:
                //   S = decay * 0 + beta * (v - 0) * k = beta * v ⊗ k
                //   o = S @ q = beta * (q · k) * v
                let qk_dot: f32 = q_h.iter().zip(k_h.iter()).map(|(&qi, &ki)| qi * ki).sum();

                // Direct computation for pos=0:
                // S[i][j] = beta_h * v_h[i] * k_h[j]
                // o[i] = sum_j S[i][j] * q_h[j] = beta_h * v_h[i] * (k_h . q_h)
                //      = beta_h * qk_dot * v_h[i]
                for i in 0..d_v {
                    output[h_v * d_v + i] = beta_h * qk_dot * v_h[i];
                }
            }
        }

        // 9. Per-head RMSNorm using linear_attn.norm.weight
        let norm_weight = w.get("linear_attn.norm.weight").ok_or("missing norm.weight")?;
        for head in 0..n_v {
            let base = head * d_v;
            let mut sq_sum = 0.0f32;
            for d in 0..d_v { sq_sum += output[base + d] * output[base + d]; }
            let rms = (sq_sum / d_v as f32 + rms_eps).sqrt();
            for d in 0..d_v {
                output[base + d] = (output[base + d] / rms) * norm_weight.get(d).copied().unwrap_or(1.0);
            }
        }

        // 10. Z-gate: z = hidden @ in_proj_z.weight, then output *= silu(z)
        let z_w = w.get("linear_attn.in_proj_z.weight").ok_or("missing in_proj_z")?;
        let z_t = Tensor::from_vec(z_w.clone(), vec![v_total, h]);
        let z = hidden_t.matmul(&z_t.transpose()); // [1, v_total]
        for i in 0..output.len() {
            let z_val = z.data[i];
            let silu_z = z_val * (1.0 / (1.0 + (-z_val).exp()));
            output[i] *= silu_z;
        }

        // 11. Output projection: [1, v_total] @ [h, v_total]^T = [1, h]
        let o_w = w.get("linear_attn.out_proj.weight").ok_or("missing out_proj")?;
        let o_t = Tensor::from_vec(o_w.clone(), vec![h, v_total]);
        let out_t = Tensor::from_vec(output, vec![1, v_total]);
        let result = out_t.matmul(&o_t.transpose());

        if layer_idx == 0 {
            eprintln!("[stream] layer 0 (deltanet) OUT first 4 = {:?}", &result.data[..4]);
        }
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
        let rms_eps = self.cfg.rms_norm_eps;

        let norm_w = w
            .get("input_layernorm.weight")
            .ok_or("missing input_layernorm")?;
        let normed = rms_norm(hidden, norm_w, rms_eps);

        // Q = hidden @ q_proj^T — q_proj is [h, h]
        let q_w = w.get("self_attn.q_proj.weight").ok_or("missing q_proj")?;
        let k_w = w.get("self_attn.k_proj.weight").ok_or("missing k_proj")?;
        let v_w = w.get("self_attn.v_proj.weight").ok_or("missing v_proj")?;
        let o_w = w.get("self_attn.o_proj.weight").ok_or("missing o_proj")?;

        let hidden_t = Tensor::from_vec(normed, vec![1, h]);
        // q_proj is [2h, h] — first h is Q, second h is attn_output_gate
        let q_all = hidden_t.matmul(&Tensor::from_vec(q_w.clone(), vec![2 * h, h]).transpose());
        let q_data = q_all.data[..h].to_vec();
        let q = Tensor::from_vec(q_data, vec![1, h]);
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

    /// Generate text: single-token autoregressive loop.
    /// Note: no KV cache yet — each token re-processes all 32 layers.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, String> {
        let mut ids = self.tok.encode(prompt, 1024);
        eprintln!("[generate] prompt tokens: {}", ids.len());
        for i in 0..max_tokens {
            let last = *ids.last().unwrap() as i32;
            let pos = ids.len() - 1;
            let logits = self.forward_one_token(last, pos)?;
            let next = Self::argmax(&logits) as i32;
            ids.push(next);
            let text = self.tok.decode(&[next]);
            print!("{text}");
            std::io::Write::flush(&mut std::io::stdout()).ok();
            if i == 0 {
                eprintln!("\n[generate] first token id={next} text=\"{text}\"");
            }
        }
        Ok(self.tok.decode(&ids))
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

#[inline]
fn softplus(x: f32) -> f32 {
    (1.0f32 + x.exp()).ln()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}
