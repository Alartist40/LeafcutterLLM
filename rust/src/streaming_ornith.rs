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
use crate::cache::deltanet_state::DeltaNetStateCache;
use crate::model::tensor::Tensor;
use crate::ornith_config::OrnithConfig;
use crate::safetensors_loader::{Shards, WeightProvider};
use crate::gguf_provider::{self, GGUFWeightProvider};
use std::collections::HashMap;
use std::path::Path;

pub struct StreamingOrnith {
    pub cfg: OrnithConfig,
    pub weights: Box<dyn WeightProvider>,
    pub tok: BpeTokenizer,
    /// Per-layer DeltaNet state matrices (accumulated across prompt tokens)
    pub deltanet_cache: DeltaNetStateCache,
    /// Per-layer KV cache for full_attention layers: layer_idx -> (keys, values)
    /// keys/values: flat Vec<f32> of shape [seq_len, n_kv * head_dim] each
    pub kv_cache: HashMap<usize, (Vec<f32>, Vec<f32>)>,
}

impl StreamingOrnith {
    /// Open a model from a directory containing config.json, tokenizer.json,
    /// and .safetensors file(s).
    pub fn open(dir: &Path) -> Result<Self, String> {
        let cfg = OrnithConfig::load(dir.join("config.json").to_str().unwrap())?;
        let shards = Shards::open_dir(dir)?;
        let tok = BpeTokenizer::load(dir.join("tokenizer.json").to_str().unwrap())?;
        Ok(Self { cfg, weights: Box::new(shards), tok, deltanet_cache: DeltaNetStateCache::new(), kv_cache: HashMap::new() })
    }

    /// Open a model from a GGUF file.
    ///
    /// `gguf_path` — path to the .gguf file.
    /// `tokenizer_path` — path to tokenizer.json (can be in the same directory).
    pub fn open_gguf(gguf_path: &str, tokenizer_path: &str) -> Result<Self, String> {
        use crate::model::gguf::GGUFile;
        let gguf = GGUFile::open(gguf_path)
            .map_err(|e| format!("open GGUF: {e}"))?;
        let cfg = gguf_provider::extract_ornith_config(&gguf)?;
        let provider = GGUFWeightProvider::from_gguf(gguf)
            .map_err(|e| format!("GGUF provider: {e}"))?;
        let tok = BpeTokenizer::load(tokenizer_path)?;
        Ok(Self { cfg, weights: Box::new(provider), tok, deltanet_cache: DeltaNetStateCache::new(), kv_cache: HashMap::new() })
    }

    /// Forward pass for ONE token. Returns logits (vocab_size entries).
    ///
    /// `token_id`: the input token.
    /// Forward a multi-token sequence, processing all tokens layer by layer.
    /// Each layer's weights are loaded ONCE from disk and shared across all tokens.
    /// This avoids re-loading weights for every token.
    pub fn forward_sequence(&mut self, tokens: &[i32]) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden_size;
        let num_layers = self.cfg.num_hidden_layers;
        let layer_types: Vec<String> = self.cfg.layer_types.clone();
        let rms_eps = self.cfg.rms_norm_eps;
        let vocab_size = self.cfg.vocab_size;
        let seq_len = tokens.len();

        // 1. Embeddings for all tokens
        let embed_name = "model.language_model.embed_tokens.weight";
        let mut hidden_states: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for &tid in tokens {
            let embed = self.weights.read_tensor_slice_f32(embed_name, tid as usize * h, h)?;
            hidden_states.push(embed);
        }

        // 2. Process each layer: load weights ONCE, run all tokens
        for layer_idx in 0..num_layers {
            let layer_type = layer_types
                .get(layer_idx)
                .map(|s| s.as_str())
                .unwrap_or("linear_attention");

            let weights = self.load_layer_weights(layer_idx, layer_type)?;

            for pos in 0..seq_len {
                let attn_out = if layer_type == "linear_attention" {
                    self.deltanet_forward(&weights, &hidden_states[pos], layer_idx)?
                } else {
                    self.attention_forward(&weights, &hidden_states[pos], layer_idx, pos)?
                };

                // Residual
                let mut new_hidden = hidden_states[pos].clone();
                for i in 0..h { new_hidden[i] += attn_out[i]; }

                // Post-attention norm + MLP
                let post_norm = weights
                    .get("post_attention_layernorm.weight")
                    .ok_or("missing post_attention_layernorm")?;
                let normed = rms_norm(&new_hidden, post_norm, rms_eps);
                let mlp_out = self.mlp_forward(&weights, &normed)?;

                for i in 0..h { new_hidden[i] += mlp_out[i]; }
                hidden_states[pos] = new_hidden;
            }

            // Debug: dump last token's hidden at every layer
            {
                let last = seq_len - 1;
                let ma = hidden_states[last].iter().map(|v| v.abs()).sum::<f32>() / h as f32;
                eprintln!("[rust] L{} tok{} mean_abs={:.6} first4={:.4?}",
                    layer_idx, last, ma, &hidden_states[last][..4]);
            }
        }

        // 3. Final norm on last token
        let final_w = self.weights.read_tensor_f32("model.language_model.norm.weight")?;
        let last_hidden = rms_norm(&hidden_states[seq_len - 1], &final_w, rms_eps);

        // 4. LM head: logits = hidden @ lm_head.T
        let lm_head_name = "lm_head.weight";
        let mut logits = vec![0.0f32; vocab_size];
        let chunk_size = 1024;
        for chunk_start in (0..vocab_size).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(vocab_size);
            let n_rows = chunk_end - chunk_start;
            let chunk = self.weights.read_tensor_slice_f32(lm_head_name, chunk_start * h, n_rows * h)?;
            let chunk_t = Tensor::from_vec(chunk, vec![n_rows, h]);
            let hidden_t = Tensor::from_vec(last_hidden.clone(), vec![1, h]);
            let logits_chunk = hidden_t.matmul(&chunk_t.transpose());
            logits[chunk_start..chunk_end].copy_from_slice(&logits_chunk.data);
        }

        Ok(logits)
    }

    /// Forward ONE token (legacy — calls forward_sequence internally).
    pub fn forward_one_token(
        &mut self,
        token_id: i32,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        // forward_sequence resets caches, so we can't use it for multi-token
        // unless we handle caching properly. For now, forward_one_token stays.
        let h = self.cfg.hidden_size;
        let num_layers = self.cfg.num_hidden_layers;
        let layer_types: Vec<String> = self.cfg.layer_types.clone();
        let rms_eps = self.cfg.rms_norm_eps;
        let vocab_size = self.cfg.vocab_size;

        let embed_name = "model.language_model.embed_tokens.weight";
        let mut hidden = self.weights.read_tensor_slice_f32(embed_name, token_id as usize * h, h)?;

        for layer_idx in 0..num_layers {
            let layer_type = layer_types
                .get(layer_idx)
                .map(|s| s.as_str())
                .unwrap_or("linear_attention");

            let weights = self.load_layer_weights(layer_idx, layer_type)?;

            let residual = hidden.clone();
            let attn_out = if layer_type == "linear_attention" {
                self.deltanet_forward(&weights, &hidden, layer_idx)?
            } else {
                self.attention_forward(&weights, &hidden, layer_idx, pos)?
            };

            for i in 0..h { hidden[i] = residual[i] + attn_out[i]; }

            let post_norm = weights
                .get("post_attention_layernorm.weight")
                .ok_or("missing post_attention_layernorm")?;
            let normed = rms_norm(&hidden, post_norm, rms_eps);
            let mlp_out = self.mlp_forward(&weights, &normed)?;

            for i in 0..h { hidden[i] += mlp_out[i]; }
        }

        let final_w = self.weights.read_tensor_f32("model.language_model.norm.weight")?;
        hidden = rms_norm(&hidden, &final_w, rms_eps);

        let lm_head_name = "lm_head.weight";
        let mut logits = vec![0.0f32; vocab_size];
        let chunk_size = 1024;
        for chunk_start in (0..vocab_size).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(vocab_size);
            let n_rows = chunk_end - chunk_start;
            let chunk = self.weights.read_tensor_slice_f32(lm_head_name, chunk_start * h, n_rows * h)?;
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
        &self,
        layer_idx: usize,
        layer_type: &str,
    ) -> Result<HashMap<String, Vec<f32>>, String> {
        let pfx = format!("model.language_model.layers.{layer_idx}.");

        let names: Vec<&str> = match layer_type {
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
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ],
            _ => return Err(format!("unknown layer type: {layer_type}")),
        };

        self.weights.load_layer_weights(layer_idx, layer_type, &names, &pfx)
    }

    /// DeltaNet (linear attention) forward — simplified.
    /// For now, just do the QKV projection and output projection.
    /// Full DeltaNet recurrence to be implemented next.
    fn deltanet_forward(
        &mut self,
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
        if layer_idx < 2 {
            let mn = normed.iter().map(|v| v.abs()).sum::<f32>() / h as f32;
            eprintln!("[dbg] layer {} normed mean_abs={:.6} first4={:.4?}", layer_idx, mn, &normed[..4]);
        }
        // 2. QKV projection: [1, h] @ [conv_dim, h]^T = [1, conv_dim]
        let qkv_w = w.get("linear_attn.in_proj_qkv.weight").ok_or("missing in_proj_qkv")?;
        let qkv_t = Tensor::from_vec(qkv_w.clone(), vec![conv_dim, h]);
        let hidden_t = Tensor::from_vec(normed.clone(), vec![1, h]);
        let qkv_proj = hidden_t.matmul(&qkv_t.transpose());
        // qkv_proj.data: [conv_dim] = [Q(2048) | K(2048) | V(4096)]

        // 3. Conv1d: per-channel FIR filter of kernel_size=4 on the QKV sequence.
        //    conv1d.weight: [conv_dim, 1, 4] in safetensors, stored flat as weight[c*conv_k + k].
        //    PyTorch Conv1d padding=conv_k-1 with causal padding:
        //      out[t][c] = sum_{k=0}^{conv_k-1} weight[c][k] * input[t - (conv_k-1) + k]
        //    For causal streaming:
        //      out[c] = weight[c][0]*input[t-3] + weight[c][1]*input[t-2] + weight[c][2]*input[t-1] + weight[c][3]*input[t]
        //    The conv buffer stores [conv_dim, conv_k] as [c * conv_k + k] where k=0 is oldest, k=conv_k-1 is newest.
        let conv_w = w.get("linear_attn.conv1d.weight").ok_or("missing conv1d")?;
        let cbuf = self.deltanet_cache.get_conv_buf_mut(layer_idx, conv_dim, conv_k);
        // Shift buffer: oldest tap 0 is discarded, taps shift left, newest tap (conv_k-1) gets current input
        for c in 0..conv_dim {
            let base = c * conv_k;
            for k in 0..conv_k - 1 {
                cbuf[base + k] = cbuf[base + k + 1];
            }
            cbuf[base + conv_k - 1] = qkv_proj.data[c];
        }
        let mut conv_out = vec![0.0f32; conv_dim];
        for c in 0..conv_dim {
            let mut sum = 0.0f32;
            let base = c * conv_k;
            // In safetensors, weight[c][k] for tap k where k=0 is oldest, k=conv_k-1 is newest.
            // In PyTorch Conv1d with padding=K-1: out[t] = w[0]*x[t-3] + w[1]*x[t-2] + w[2]*x[t-1] + w[3]*x[t]
            // Our buf: [input[t-3], input[t-2], input[t-1], input[t]]
            // So: w[c][k] * buf[k] for k=0..conv_k-1
            for k in 0..conv_k {
                sum += conv_w[base + k] * cbuf[base + k];
            }
            conv_out[c] = sum;
        }
        if layer_idx < 2 {
            let pre_silu_ma = conv_out.iter().map(|v| v.abs()).sum::<f32>() / conv_out.len() as f32;
            eprintln!("[dbg] layer {} conv pre-silu mean_abs={:.6} first4={:.4?}", layer_idx, pre_silu_ma, &conv_out[..4]);
        }
        // SiLU after conv
        for v in conv_out.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
        // 4. Debug: check QKV section magnitudes
        let q_total = n_qk * d_k;   // 2048
        let k_total = n_qk * d_k;   // 2048
        let v_total = n_v * d_v;    // 4096
        if layer_idx < 2 {
            let qkv_ma = conv_out.iter().map(|v| v.abs()).sum::<f32>() / conv_out.len() as f32;
            eprintln!("[dbg] layer {} conv+silu mean_abs={:.6}", layer_idx, qkv_ma);
            let sect0 = &conv_out[..q_total];
            let sect1 = &conv_out[q_total..q_total + k_total];
            let sect2 = &conv_out[q_total + k_total..];
            let m0 = sect0.iter().map(|v| v.abs()).sum::<f32>() / q_total as f32;
            let m1 = sect1.iter().map(|v| v.abs()).sum::<f32>() / k_total as f32;
            let m2 = sect2.iter().map(|v| v.abs()).sum::<f32>() / v_total as f32;
            eprintln!("[dbg] layer {} QKV mean_abs: Q={:.6} K={:.6} V={:.6}", layer_idx, m0, m1, m2);
        }
        // Split into Q, K, V — try Q|K|V order first
        let q_data = &conv_out[..q_total];
        let k_data = &conv_out[q_total..q_total + k_total];
        let v_data = &conv_out[q_total + k_total..];

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
        if layer_idx < 2 {
            let q_ma = q.iter().map(|v| v.abs()).sum::<f32>() / q.len() as f32;
            let k_ma = k.iter().map(|v| v.abs()).sum::<f32>() / k.len() as f32;
            eprintln!("[dbg] layer {} Q after norm+scale mean_abs={:.6} K after norm mean_abs={:.6}", layer_idx, q_ma, k_ma);
        }

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
        // Debug: check if any softplus overflows
        if layer_idx < 2 {
            let max_alpha = alpha.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_alpha = alpha.data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_dt = (0..n_v).map(|h| softplus(alpha.data[h] + dt_bias.get(h).copied().unwrap_or(0.0))).fold(f32::NEG_INFINITY, f32::max);
            eprintln!("[dbg] layer {} alpha range=[{:.2}, {:.2}] max_dt={:.2}", layer_idx, min_alpha, max_alpha, max_dt);
        }

        // 7. Compute beta gates: beta = sigmoid(hidden @ in_proj_b.weight)
        let b_w = w.get("linear_attn.in_proj_b.weight").ok_or("missing in_proj_b")?;
        let b_t = Tensor::from_vec(b_w.clone(), vec![n_v, h]);
        let beta_logits = hidden_t.matmul(&b_t.transpose()); // [1, n_v]
        let beta: Vec<f32> = beta_logits.data.iter().map(|&v| sigmoid(v)).collect();
        if layer_idx < 2 {
            let d_ma = decay.iter().sum::<f32>() / decay.len() as f32;
            let b_ma = beta.iter().sum::<f32>() / beta.len() as f32;
            eprintln!("[dbg] layer {} decay mean={:.6} beta mean={:.6}", layer_idx, d_ma, b_ma);
        }

        // 8. Delta rule state update + output
        let mut output = vec![0.0f32; v_total];

        // Ensure state is initialized for this layer
        if self.deltanet_cache.get(layer_idx).is_none() {
            self.deltanet_cache.init_layer(layer_idx, n_v, d_v, d_k);
        }

        // Qwen3.5 V-head pairing is INTERLEAVED (llama.cpp ggml_repeat_4d,
        // llama-model.cpp:523-525): v_head h_v uses k/q head (h_v % n_qk).
        // Pattern [k0_v0, k1_v1, k0_v2, k1_v3] for n_v=2*n_qk.
        for h_v in 0..n_v {
            let h_qk = h_v % n_qk;
            let q_h = &q[h_qk * d_k..(h_qk + 1) * d_k];
            let k_h = &k[h_qk * d_k..(h_qk + 1) * d_k];
            let v_h = &v_data[h_v * d_v..(h_v + 1) * d_v];
            {
                let decay_h = decay[h_v];
                let beta_h = beta[h_v];

                // Decay state first: S = decay_h * S
                let state = self.deltanet_cache.get_mut(layer_idx).unwrap();
                let state_stride = h_v * d_v * d_k;
                for i in 0..d_v {
                    for j in 0..d_k {
                        let idx = state_stride + i * d_k + j;
                        state[idx] = decay_h * state[idx];
                    }
                }

                // Compute v_pred = S @ k (using decayed state)
                let mut v_pred = vec![0.0f32; d_v];
                for i in 0..d_v {
                    let mut sum = 0.0f32;
                    for j in 0..d_k {
                        sum += state[state_stride + i * d_k + j] * k_h[j];
                    }
                    v_pred[i] = sum;
                }

                // State update: S = S + beta_h * ((v - v_pred) ⊗ k)
                for i in 0..d_v {
                    let delta = v_h[i] - v_pred[i];
                    for j in 0..d_k {
                        let idx = state_stride + i * d_k + j;
                        state[idx] = state[idx] + beta_h * delta * k_h[j];
                    }
                }

                // Output: o_h = S @ q (q already scaled by 1/sqrt(d_k))
                for i in 0..d_v {
                    let mut sum = 0.0f32;
                    for j in 0..d_k {
                        sum += state[state_stride + i * d_k + j] * q_h[j];
                    }
                    output[h_v * d_v + i] = sum;
                }
            }
        }

        if layer_idx < 2 {
            let delta_ma = output.iter().map(|v| v.abs()).sum::<f32>() / output.len() as f32;
            eprintln!("[dbg] layer {} delta output mean_abs={:.8} first4={:.6?}",
                layer_idx, delta_ma, &output[..4]);
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
        if layer_idx < 2 {
            let nm_ma = output.iter().map(|v| v.abs()).sum::<f32>() / output.len() as f32;
            eprintln!("[dbg] layer {} after rmsnorm mean_abs={:.8} first4={:.6?}",
                layer_idx, nm_ma, &output[..4]);
        }

        // 10. Z-gate: z = hidden @ in_proj_z.weight, then output *= silu(z)
        let z_w = w.get("linear_attn.in_proj_z.weight").ok_or("missing in_proj_z")?;
        let z_t = Tensor::from_vec(z_w.clone(), vec![v_total, h]);
        let z = hidden_t.matmul(&z_t.transpose()); // [1, v_total]
        let z_mean = z.data.iter().map(|v| v.abs()).sum::<f32>() / z.data.len() as f32;
        for i in 0..output.len() {
            let z_val = z.data[i];
            let silu_z = z_val * (1.0 / (1.0 + (-z_val).exp()));
            output[i] *= silu_z;
        }
        if layer_idx < 2 {
            let z_out_ma = output.iter().map(|v| v.abs()).sum::<f32>() / output.len() as f32;
            eprintln!("[dbg] layer {} z mean_abs={:.6} after-z-gate mean_abs={:.6}",
                layer_idx, z_mean, z_out_ma);
        }

        // 11. Output projection: [1, v_total] @ [h, v_total]^T = [1, h]
        let o_w = w.get("linear_attn.out_proj.weight").ok_or("missing out_proj")?;
        let o_t = Tensor::from_vec(o_w.clone(), vec![h, v_total]);
        let out_t = Tensor::from_vec(output, vec![1, v_total]);
        let result = out_t.matmul(&o_t.transpose());

        if layer_idx < 2 {
            let mean_abs = result.data.iter().map(|v| v.abs()).sum::<f32>() / h as f32;
            eprintln!("[stream] layer {} (deltanet) OUT mean_abs={:.4} first 4 = {:?}",
                layer_idx, mean_abs, &result.data[..4]);
        }
        Ok(result.data)
    }

    /// Standard attention forward (for full_attention layers).
    fn attention_forward(
        &mut self,
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

        // Q = hidden @ q_proj^T — q_proj is [2h, h] (first h = Q, second h = gate)
        let q_w = w.get("self_attn.q_proj.weight").ok_or("missing q_proj")?;
        let k_w = w.get("self_attn.k_proj.weight").ok_or("missing k_proj")?;
        let v_w = w.get("self_attn.v_proj.weight").ok_or("missing v_proj")?;
        let o_w = w.get("self_attn.o_proj.weight").ok_or("missing o_proj")?;
        let q_norm_w = w.get("self_attn.q_norm.weight");
        let k_norm_w = w.get("self_attn.k_norm.weight");

        let hidden_t = Tensor::from_vec(normed, vec![1, h]);
        // q_proj output: [1, 2h] — split into Q and gate
        let q_all = hidden_t.matmul(&Tensor::from_vec(q_w.clone(), vec![2 * h, h]).transpose());
        let mut q = q_all.data[..h].to_vec();
        let gate = &q_all.data[h..];
        let mut k = hidden_t.matmul(&Tensor::from_vec(k_w.clone(), vec![n_kv * head_dim, h]).transpose()).data;
        let v = hidden_t.matmul(&Tensor::from_vec(v_w.clone(), vec![n_kv * head_dim, h]).transpose()).data;

        // Apply Q norm (per-head RMSNorm with shared weight)
        if let Some(qnw) = q_norm_w {
            for head in 0..n_heads {
                let base = head * head_dim;
                let mut sq = 0.0f32;
                for d in 0..head_dim { sq += q[base + d] * q[base + d]; }
                let r = (sq / head_dim as f32 + rms_eps).sqrt();
                for d in 0..head_dim { q[base + d] = q[base + d] / r * qnw[d]; }
            }
        }
        // Apply K norm (per-head RMSNorm with shared weight)
        if let Some(knw) = k_norm_w {
            for head in 0..n_kv {
                let base = head * head_dim;
                let mut sq = 0.0f32;
                for d in 0..head_dim { sq += k[base + d] * k[base + d]; }
                let r = (sq / head_dim as f32 + rms_eps).sqrt();
                for d in 0..head_dim { k[base + d] = k[base + d] / r * knw[d]; }
            }
        }

        // RoPE (GLM-style split pairs): pair (i, i+rotary_dim/2) for i in 0..rotary_dim/2
        let rotary_dim = (self.cfg.head_dim as f32 * 0.25) as usize;
        let rope_theta = 10000000f32;
        let half_rotary = rotary_dim / 2;
        for head in 0..n_heads {
            let base = head * head_dim;
            for i in 0..half_rotary {
                let i0 = base + i;
                let i1 = base + i + half_rotary;
                let angle = pos as f32 / rope_theta.powf(2.0 * i as f32 / rotary_dim as f32);
                let (sin, cos) = angle.sin_cos();
                let x0 = q[i0];
                let x1 = q[i1];
                q[i0] = x0 * cos - x1 * sin;
                q[i1] = x0 * sin + x1 * cos;
            }
        }
        for head in 0..n_kv {
            let base = head * head_dim;
            for i in 0..half_rotary {
                let i0 = base + i;
                let i1 = base + i + half_rotary;
                let angle = pos as f32 / rope_theta.powf(2.0 * i as f32 / rotary_dim as f32);
                let (sin, cos) = angle.sin_cos();
                let x0 = k[i0];
                let x1 = k[i1];
                k[i0] = x0 * cos - x1 * sin;
                k[i1] = x0 * sin + x1 * cos;
            }
        }

        // Append current token's K, V to cache
        let kv_entry = self.kv_cache.entry(layer_idx).or_insert((Vec::new(), Vec::new()));
        kv_entry.0.extend_from_slice(&k);
        kv_entry.1.extend_from_slice(&v);
        let cached_k = &kv_entry.0;
        let cached_v = &kv_entry.1;
        let seq_len = cached_k.len() / (n_kv * head_dim);

        // Attention over all cached tokens
        let mut attn_out = vec![0.0f32; h];
        for head in 0..n_heads {
            let kv_head = head / (n_heads / n_kv);
            let q_base = head * head_dim;
            // Compute score = q @ k_cache / sqrt(d)
            let mut max_score = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let k_base = t * n_kv * head_dim + kv_head * head_dim;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_base + d] * cached_k[k_base + d];
                }
                score /= (head_dim as f32).sqrt();
                if score > max_score { max_score = score; }
            }
            // Compute softmax numerator for each position and sum V weighted
            let mut sum_exp = 0.0f32;
            let mut weighted_v = vec![0.0f32; head_dim];
            for t in 0..seq_len {
                let k_base = t * n_kv * head_dim + kv_head * head_dim;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_base + d] * cached_k[k_base + d];
                }
                score = (score / (head_dim as f32).sqrt() - max_score).exp();
                sum_exp += score;
                let v_base = t * n_kv * head_dim + kv_head * head_dim;
                for d in 0..head_dim {
                    weighted_v[d] += score * cached_v[v_base + d];
                }
            }
            let inv_sum = 1.0 / sum_exp;
            for d in 0..head_dim {
                attn_out[q_base + d] = weighted_v[d] * inv_sum;
            }
        }

        // Apply output gate: attn_out *= sigmoid(gate)
        for i in 0..h {
            attn_out[i] *= 1.0 / (1.0 + (-gate[i]).exp());
        }

        // Output projection
        let out_t = Tensor::from_vec(attn_out, vec![1, h]);
        let result = out_t.matmul(&Tensor::from_vec(o_w.clone(), vec![h, h]).transpose());
        if layer_idx == 31 {
            eprintln!("[stream] layer {layer_idx} (full_attn) done, o first 4 = {:?}", &result.data[..4]);
        }
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
    /// Generate text from a raw prompt (no chat wrapping).
    /// `stop_tokens` — generation stops when any of these token IDs is produced.
    pub fn generate_with_stop(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        stop_tokens: &[i32],
    ) -> Result<String, String> {
        let mut ids = self.tok.encode(prompt, 1024);
        eprintln!("[generate] prompt tokens: {}", ids.len());
        for _ in 0..max_tokens {
            let last = *ids.last().unwrap() as i32;
            let pos = ids.len() - 1;
            let logits = self.forward_one_token(last, pos)?;
            let next = Self::argmax(&logits) as i32;
            if stop_tokens.contains(&next) {
                break;
            }
            ids.push(next);
            let text = self.tok.decode(&[next]);
            print!("{text}");
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        Ok(self.tok.decode(&ids))
    }

    /// Chat convenience: wraps input in the Ornith chat template and
    /// stops at `<|im_end|>` (looked up dynamically from the tokenizer).
    pub fn chat(&mut self, user_input: &str, max_tokens: usize) -> Result<String, String> {
        let system = "You are Ornith, an open-source agentic coding assistant. Think step by step in a reasoning block, then act. Use the provided tools when they help. Be concise, correct, and direct: write working code and explain only what is non-obvious.";
        let prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
            system, user_input
        );
        let stop_im_end = self.tok.id_of("<|im_end|>");
        let stop_eot = self.tok.id_of("<|endoftext|>");
        let stops: Vec<i32> = [stop_im_end, stop_eot].into_iter().filter(|&id| id >= 0).collect();
        self.generate_with_stop(&prompt, max_tokens, &stops)
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, String> {
        self.generate_with_stop(prompt, max_tokens, &[])
    }
}

/// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
/// NOTE: converter bakes gamma as (1 + gamma) into GGUF, so weight multiplies directly.
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
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}
