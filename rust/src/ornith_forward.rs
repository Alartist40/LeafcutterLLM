//! Native Rust safetensors forward pass for Ornith-1.0-9B.
//!
//! Loads safetensors, runs token-by-token forward through all 32 layers
//! (24 linear_attention + 8 full_attention), and produces logits.
//!
//! Reuses the existing leafcutter inference code (deltanet_forward,
//! attention_forward, etc.) via the safetensor_tensors + engine_keymap
//! bridge.

use crate::bpe_tokenizer::BpeTokenizer;
use crate::engine_keymap::load_layer_weights;
use crate::model::tensor::Tensor;
use crate::ornith_config::OrnithConfig;
use crate::safetensor_tensors::SafetensorTensors;
use std::path::Path;

use crate::cache::deltanet_state::DeltaNetStateCache;
use crate::inference::attention::{attention_forward, AttentionParams};
use crate::inference::deltanet::{deltanet_forward, DeltaNetParams};
use crate::inference::mla;
use crate::inference::sampler::sample_top_p;

/// Top-level safetensors model: holds config + tensor source + state.
pub struct OrnithModel {
    pub cfg: OrnithConfig,
    pub tensors: SafetensorTensors,
    pub deltanet_params: DeltaNetParams,
    pub attn_params: AttentionParams,
}

impl OrnithModel {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let cfg = OrnithConfig::load(dir.join("config.json").to_str().unwrap())?;
        let tensors = SafetensorTensors::open(dir)?;

        // DeltaNet parameters (from config + safetensors shape observations)
        let deltanet_params = DeltaNetParams {
            num_qk_heads: cfg.linear_num_key_heads,
            num_v_heads: cfg.linear_num_value_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            conv_dim: 8192, // 2*QK + V = 2*16*128 + 32*128 = 4096 + 4096 = 8192
            conv_kernel: cfg.linear_conv_kernel_dim,
            state_size: cfg.linear_key_head_dim,
            norm_eps: 1e-5,
        };

        // Standard attention parameters
        let attn_params = AttentionParams {
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            kv_head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
            rope_dim: 0, // full RoPE for Ornith
            use_fused_qkv: false,
            use_gate: false,
            window_size: 0,
        };

        Ok(Self {
            cfg,
            tensors,
            deltanet_params,
            attn_params,
        })
    }

    /// Run forward pass on a single token at a given position.
    /// Returns logits (shape: [vocab_size]).
    pub fn forward_one_token(
        &self,
        token_id: i32,
        pos: usize,
        state: &mut OrnithState,
    ) -> Result<Vec<f32>, String> {
        // 1. Embedding lookup
        let embed = self
            .tensors
            .get("model.language_model.embed_tokens.weight")
            .ok_or("missing embed_tokens")?;
        let hidden_size = self.cfg.hidden_size;
        let mut hidden: Vec<f32> = embed.data
            [token_id as usize * hidden_size..(token_id as usize + 1) * hidden_size]
            .to_vec();
        eprintln!("[t] embed lookup done, hidden[0..4]={:?}", &hidden[..4]);

        // 2. Run all 32 layers
        let total_t0 = std::time::Instant::now();
        eprintln!("[t] starting layer loop at {:?}", total_t0);
        for layer_idx in 0..self.cfg.num_hidden_layers {
            let layer_type = self
                .cfg
                .layer_types
                .get(layer_idx)
                .map(|s| s.as_str())
                .unwrap_or("linear_attention");
            let load_t = total_t0.elapsed();
            let weights = load_layer_weights(&self.tensors, layer_idx, layer_type);
            eprintln!("[t] load layer {} took {:?}", layer_idx, total_t0.elapsed() - load_t);

            let residual = hidden.clone();
            let run_t = total_t0.elapsed();
            hidden = self.run_layer(layer_idx, layer_type, &weights, hidden, pos, state)?;
            eprintln!("[t] layer {} ({}) run={:?} total={:?}", layer_idx, layer_type, total_t0.elapsed() - run_t, total_t0.elapsed());
            for i in 0..hidden.len() {
                hidden[i] += residual[i];
            }
        }
        eprintln!("[t] all 32 layers done in {:?}", total_t0.elapsed());

        // 3. Final norm
        let final_norm = self
            .tensors
            .get("model.language_model.norm.weight")
            .ok_or("missing final norm")?;
        let hidden_t = Tensor::from_vec(hidden, vec![1, hidden_size]);
        let normed = hidden_t.rms_norm(&final_norm, self.cfg.rms_norm_eps);
        let mut hidden: Vec<f32> = normed.data;

        // 4. LM head: logits = hidden @ lm_head.T (use Tensor::matmul for speed)
        let lm_head = self
            .tensors
            .get("lm_head.weight")
            .ok_or("missing lm_head")?;
        let vocab_size = self.cfg.vocab_size;
        let hidden_t = Tensor::from_vec(hidden, vec![1, hidden_size]);
        // Tensor::matmul expects A @ B where A is [m,k] and B is [k,n].
        // lm_head is [vocab, hidden], so we want hidden @ lm_head.T.
        // Transpose lm_head (one-time cost on load would be better, but do it here for now).
        let lm_head_t = lm_head.transpose();
        let logits_t = hidden_t.matmul(&lm_head_t);
        let mut logits = logits_t.data;
        // Truncate to vocab_size in case lm_head was padded.
        logits.truncate(vocab_size);
        Ok(logits)
    }

    fn run_layer(
        &self,
        layer_idx: usize,
        layer_type: &str,
        weights: &std::collections::HashMap<String, Tensor>,
        hidden: Vec<f32>,
        pos: usize,
        state: &mut OrnithState,
    ) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden_size;
        let hidden_tensor = Tensor::from_vec(hidden, vec![1, h]);
        let saved_for_residual = hidden_tensor.data.clone();

        // Attention block
        let attn_out = if layer_type == "linear_attention" {
            // DeltaNet forward
            deltanet_forward(
                &hidden_tensor,
                weights,
                &self.deltanet_params,
                &mut state.deltanet,
                layer_idx,
            )
        } else {
            // Standard attention
            attention_forward(
                &hidden_tensor,
                weights,
                &self.attn_params,
                &mut state.kv,
                layer_idx,
                pos,
            )
        };

        // Add residual handled by caller. Return just the attention output.
        let mut out = attn_out.data;

        // Post-attention norm
        if let Some(post_norm) = weights.get("ffn_norm.weight") {
            let out_t = Tensor::from_vec(out, vec![1, h]);
            let normed = out_t.rms_norm(post_norm, self.cfg.rms_norm_eps);
            out = normed.data;
        }

        // MLP block (SwiGLU)
        if let (Some(gate), Some(up), Some(down)) = (
            weights.get("mlp.gate_proj.weight"),
            weights.get("mlp.up_proj.weight"),
            weights.get("mlp.down_proj.weight"),
        ) {
            let inter = self.cfg.intermediate_size;
            // Use Tensor::matmul (dispatches to BLAS-like backend) instead of
            // naive triple-loop.  This is the difference between 600s/tok and 12s/tok.
            let out_tensor = Tensor::from_vec(out, vec![1, h]);
            let gate_out_t = out_tensor.matmul(gate);
            let up_out_t = out_tensor.matmul(up);
            let mut mlp_hidden = vec![0.0f32; inter];
            crate::ornith_kernels::swiglu(&mut mlp_hidden, &gate_out_t.data, &up_out_t.data);
            let mlp_hidden_t = Tensor::from_vec(mlp_hidden, vec![1, inter]);
            let mlp_out_t = mlp_hidden_t.matmul(down);
            out = mlp_out_t.data;
        }

        Ok(out)
    }

    /// Sample top-1 (greedy) from logits.
    pub fn argmax(logits: &[f32]) -> usize {
        let mut best = 0;
        let mut best_val = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        best
    }
}

/// Per-call state caches (KV for attention, DeltaNet states).
pub struct OrnithState {
    pub kv: crate::cache::KVCache,
    pub deltanet: DeltaNetStateCache,
}

impl OrnithState {
    pub fn new(cfg: &OrnithConfig) -> Self {
        Self {
            kv: crate::cache::KVCache::new(cfg.num_hidden_layers),
            deltanet: DeltaNetStateCache::new(),
        }
    }
}
