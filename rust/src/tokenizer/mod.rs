//! Tokenizers for Leafcutter
//!
//! Two implementations:
//!   - `Tokenizer`: HuggingFace `tokenizers` crate (exact, needs `tokenizer.json`)
//!   - `GgufTokenizer`: GGUF vocab fallback (no external deps, greedy longest-match)

pub mod chat_template;
pub mod gguf;

pub use chat_template::{apply_chat_template_from_gguf, TemplateFamily};
pub use gguf::GgufTokenizer;

use tokenizers::Tokenizer as HFTokenizer;

pub struct Tokenizer {
    inner: HFTokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let inner = HFTokenizer::from_file(path)?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.inner.encode(text, false).expect("Tokenizer encode failed");
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    pub fn decode(&self, tokens: &[usize], skip_special: bool) -> String {
        let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        self.inner.decode(&ids, skip_special).expect("Tokenizer decode failed")
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn apply_chat_template(&self, user_message: &str) -> String {
        format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_message
        )
    }
}
