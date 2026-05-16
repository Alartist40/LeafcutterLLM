//! BPE Tokenizer wrapper using HuggingFace tokenizers crate
//!
//! Loads tokenizer.json and provides encode/decode for Qwen2.5 BPE.

use tokenizers::Tokenizer as HFTokenizer;

pub struct Tokenizer {
    inner: HFTokenizer,
}

impl Tokenizer {
    /// Load tokenizer from a HuggingFace tokenizer.json file
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let inner = HFTokenizer::from_file(path)?;
        Ok(Self { inner })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.inner.encode(text, false).expect("Tokenizer encode failed");
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[usize], skip_special: bool) -> String {
        let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        self.inner.decode(&ids, skip_special).expect("Tokenizer decode failed")
    }

    /// Vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Apply Qwen2.5 chat template to a user message.
    ///
    /// Format:
    /// ```text
    /// <|im_start|>system
    /// You are a helpful assistant.<|im_end|>
    /// <|im_start|>user
    /// {message}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    pub fn apply_chat_template(&self, user_message: &str) -> String {
        format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_message
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_roundtrip() {
        let path = "tests/tokenizer.json";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: tokenizer.json not found");
            return;
        }

        let tok = Tokenizer::from_file(path).expect("Failed to load tokenizer");
        println!("Vocab size: {}", tok.vocab_size());

        let text = "Hello, world!";
        let tokens = tok.encode(text);
        println!("Tokens: {:?}", tokens);
        assert!(!tokens.is_empty());

        let decoded = tok.decode(&tokens, false);
        println!("Decoded: {}", decoded);
        assert!(decoded.contains("Hello"));
    }

    #[test]
    fn test_qwen_chat_format() {
        let path = "tests/tokenizer.json";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: tokenizer.json not found");
            return;
        }

        let tok = Tokenizer::from_file(path).expect("Failed to load tokenizer");
        let prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";
        let tokens = tok.encode(prompt);
        println!("Chat prompt tokens: {:?}", tokens);
        assert!(tokens.len() > 10);
    }
}
