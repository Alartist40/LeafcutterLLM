//! Tokenizers for Leafcutter
//!
//! Two implementations:
//!   - `Tokenizer`: HuggingFace `tokenizers` crate (exact, needs `tokenizer.json`)
//!   - `GgufTokenizer`: GGUF vocab fallback (no external deps, greedy longest-match)

pub mod chat_template;
pub mod gguf;

pub use chat_template::{apply_chat_template_from_gguf, TemplateFamily};
pub use gguf::{GgufTokenizer, GgufBpeTokenizer};

use tokenizers::Tokenizer as HFTokenizer;

pub trait BaseTokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
}

// Re-export the trait under its short name so callers can write
// `use leafcutter::tokenizer::BaseTokenizer;` and use the methods
// directly on `Tokenizer` / `GgufBpeTokenizer` (or their `dyn`s).
pub use BaseTokenizer as _;

pub struct Tokenizer {
    inner: HFTokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let inner = HFTokenizer::from_file(path)?;
        Ok(Self { inner })
    }

    pub fn apply_chat_template(&self, user_message: &str) -> String {
        format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_message
        )
    }
}

impl BaseTokenizer for Tokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.inner.encode(text, false).expect("Tokenizer encode failed");
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        self.inner.decode(&ids, true).expect("Tokenizer decode failed")
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

// Inherent method aliases so callers can write `tok.vocab_size()` without
// importing the `BaseTokenizer` trait.  These forward to the trait impls.
#[allow(dead_code)]
impl Tokenizer {
    pub fn encode_into(&self, text: &str) -> Vec<usize> { <Self as BaseTokenizer>::encode(self, text) }
    pub fn decode_into(&self, tokens: &[usize]) -> String { <Self as BaseTokenizer>::decode(self, tokens) }
    pub fn vocab_size_inherent(&self) -> usize { <Self as BaseTokenizer>::vocab_size(self) }
}

impl BaseTokenizer for GgufBpeTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        self.encode(text)
    }
    fn decode(&self, tokens: &[usize]) -> String {
        self.decode(tokens)
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

#[allow(dead_code)]
impl GgufBpeTokenizer {
    pub fn encode_into(&self, text: &str) -> Vec<usize> { <Self as BaseTokenizer>::encode(self, text) }
    pub fn decode_into(&self, tokens: &[usize]) -> String { <Self as BaseTokenizer>::decode(self, tokens) }
    pub fn vocab_size_inherent(&self) -> usize { <Self as BaseTokenizer>::vocab_size(self) }
}
