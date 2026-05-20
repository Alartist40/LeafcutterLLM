//! Direct llama.cpp FFI server engine
//!
//! Replaces the broken HybridEngine for server mode.
//! Uses llama.cpp's built-in tokenizer so token IDs always match the model.

use std::sync::Mutex;

use crate::llama_ffi::{backend_init, LlamaModel, LlamaContext};

pub struct FfiEngine {
    model: LlamaModel,
    ctx_size: u32,
    threads: i32,
}

impl FfiEngine {
    pub fn load(path: &str) -> Result<Self, String> {
        backend_init();
        let model = LlamaModel::load(std::path::Path::new(path), 0)?;
        Ok(Self {
            model,
            ctx_size: 4096,
            threads: 4,
        })
    }

    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<GenerateResult, String> {
        // Create a fresh context for each request to avoid KV cache contamination
        let mut ctx = LlamaContext::new(&self.model, self.ctx_size, self.threads)
            .map_err(|e| format!("Failed to create context: {}", e))?;

        let prompt_tokens = ctx.tokenize(prompt, true, true);
        if prompt_tokens.is_empty() {
            return Err("Empty prompt after tokenization".to_string());
        }

        let eos = self.model.eos_token();
        let generated = ctx.generate(&prompt_tokens, max_tokens, temperature, eos);

        let text = generated.iter()
            .map(|&t| ctx.token_to_piece(t))
            .collect();

        Ok(GenerateResult {
            text,
            tokens: generated.iter().map(|&t| t as usize).collect(),
        })
    }

    pub fn health(&self) -> HealthInfo {
        HealthInfo {
            n_vocab: self.model.n_vocab(),
            n_embd: self.model.n_embd(),
            n_layer: self.model.n_layer(),
        }
    }
}

pub struct GenerateResult {
    pub text: String,
    pub tokens: Vec<usize>,
}

pub struct HealthInfo {
    pub n_vocab: i32,
    pub n_embd: i32,
    pub n_layer: i32,
}
