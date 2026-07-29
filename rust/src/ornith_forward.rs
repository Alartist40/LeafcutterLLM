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

        // 2. Run all 32 layers
        for layer_idx in 0..self.cfg.num_hidden_layers {
            eprintln!("[native-forward] layer {}/{}", layer_idx, self.cfg.num_hidden_layers);
            let layer_type = self
                .cfg
                .layer_types
                .get(layer_idx)
                .map(|s| s.as_str())
                .unwrap_or("linear_attention");
            let weights = load_layer_weights(&self.tensors, layer_idx, layer_type);

            let residual = hidden.clone();
            hidden = self.run_layer(layer_idx, layer_type, &weights, hidden, pos, state)?;
            // Add residual
            for i in 0..hidden.len() {
                hidden[i] += residual[i];
            }
        }

        // 3. Final norm
        let final_norm = self
            .tensors
            .get("model.language_model.norm.weight")
            .ok_or("missing final norm")?;
        let saved = hidden.clone();
        crate::ornith_kernels::rmsnorm(
            &mut hidden,
            &saved,
            &final_norm.data,
            self.cfg.rms_norm_eps,
        );

        // 4. LM head: logits = hidden @ lm_head.T
        // lm_head.weight shape: [vocab, hidden]
        // We need logits[vocab] = sum_i hidden[i] * lm_head[v, i]
        let lm_head = self
            .tensors
            .get("lm_head.weight")
            .ok_or("missing lm_head")?;
        let vocab_size = self.cfg.vocab_size;
        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            for i in 0..hidden_size {
                sum += hidden[i] * lm_head.data[v * hidden_size + i];
            }
            logits[v] = sum;
        }
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
            let saved = out.clone();
            crate::ornith_kernels::rmsnorm(&mut out, &saved, &post_norm.data, self.cfg.rms_norm_eps);
        }

        // MLP block (SwiGLU)
        if let (Some(gate), Some(up), Some(down)) = (
            weights.get("mlp.gate_proj.weight"),
            weights.get("mlp.up_proj.weight"),
            weights.get("mlp.down_proj.weight"),
        ) {
            let inter = self.cfg.intermediate_size;
            let mut gate_out = vec![0.0f32; inter];
            crate::ornith_kernels::matmul(
                &mut gate_out,
                &out,
                &gate.data,
                1,
                h,
                inter,
            );
            let mut up_out = vec![0.0f32; inter];
            crate::ornith_kernels::matmul(
                &mut up_out,
                &out,
                &up.data,
                1,
                h,
                inter,
            );
            let mut mlp_hidden = vec![0.0f32; inter];
            crate::ornith_kernels::swiglu(&mut mlp_hidden, &gate_out, &up_out);
            let mut mlp_out = vec![0.0f32; h];
            crate::ornith_kernels::matmul(
                &mut mlp_out,
                &mlp_hidden,
                &down.data,
                1,
                inter,
                h,
            );
            out = mlp_out;
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
