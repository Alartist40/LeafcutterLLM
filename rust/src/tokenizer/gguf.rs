//! Simple tokenizer using GGUF vocab (no external dependencies)
//!
//! Reads `tokenizer.ggml.tokens` from the GGUF file and does greedy
//! longest-match tokenization. This is a pragmatic fallback when a
//! full HuggingFace `tokenizer.json` is not available.

use std::collections::HashMap;

pub struct GgufTokenizer {
    /// token_id -> token_string
    vocab: Vec<String>,
    /// token_string -> token_id
    pub token_to_id: HashMap<String, usize>,
    /// BOS token ID
    bos_id: Option<usize>,
    /// EOS token ID
    eos_id: Option<usize>,
}

impl GgufTokenizer {
    /// Build tokenizer from a list of vocab strings (extracted from GGUF).
    pub fn from_vocab(vocab_tokens: Vec<String>) -> Self {
        let mut token_to_id = HashMap::with_capacity(vocab_tokens.len());
        for (i, tok) in vocab_tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), i);
        }

        // Detect common special-token names
        let bos_id = [
            "<s>",
            "<|begin_of_text|>",
            "<|startoftext|>",
            "<|im_start|>",
            "[BOS]",
        ]
        .iter()
        .find_map(|&s| token_to_id.get(s).copied());

        let eos_id = [
            "</s>",
            "<|end_of_text|>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|eot_id|>",
            "[EOS]",
        ]
        .iter()
        .find_map(|&s| token_to_id.get(s).copied());

        Self {
            vocab: vocab_tokens,
            token_to_id,
            bos_id,
            eos_id,
        }
    }

    /// Greedy longest-match tokenization.
    ///
    /// Tries the longest possible prefix at each position. Falls back to
    /// byte-level encoding when no vocab token matches.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<usize> {
        let mut tokens = Vec::new();

        if add_bos {
            if let Some(bos) = self.bos_id {
                tokens.push(bos);
            }
        }

        let mut remaining = text;

        while !remaining.is_empty() {
            let mut matched = false;

            // Try longest match first (up to 64 chars or remaining len)
            let max_len = remaining.len().min(64);
            for len in (1..=max_len).rev() {
                // Safe slicing: iterate to char boundary
                if let Some(prefix) = remaining.get(..len) {
                    if let Some(&id) = self.token_to_id.get(prefix) {
                        tokens.push(id);
                        remaining = &remaining[len..];
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                // Byte fallback
                let first_byte = remaining.as_bytes()[0];
                let byte_token = format!("<0x{:02X}>", first_byte);
                if let Some(&id) = self.token_to_id.get(&byte_token) {
                    tokens.push(id);
                } else {
                    tokens.push(first_byte as usize);
                }
                remaining = &remaining[1..];
            }
        }

        tokens
    }

    /// Decode token IDs back to a string.
    ///
    /// Handles byte tokens (`<0xXX>`) and skips other special tokens.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut result = String::new();
        for &id in tokens {
            if id >= self.vocab.len() {
                continue;
            }
            let piece = &self.vocab[id];

            // Byte token: <0xXX>
            if piece.starts_with("<0x") && piece.len() == 6 && piece.ends_with(">") {
                if let Ok(byte) = u8::from_str_radix(&piece[3..5], 16) {
                    result.push(byte as char);
                }
                continue;
            }

            // Known newline byte token used by Llama
            if piece == "<0x0A>" {
                result.push('\n');
                continue;
            }

            // Skip other special tokens (anything wrapped in < >)
            if piece.starts_with('<') && piece.ends_with('>') {
                continue;
            }

            result.push_str(piece);
        }
        result
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn bos_id(&self) -> Option<usize> { self.bos_id }
    pub fn eos_id(&self) -> Option<usize> { self.eos_id }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenize() {
        let vocab = vec![
            "<s>".to_string(),
            "</s>".to_string(),
            "The".to_string(),
            " cat".to_string(),
            " sat".to_string(),
            " on".to_string(),
            " the".to_string(),
            " mat".to_string(),
            ".".to_string(),
            " T".to_string(),
            "h".to_string(),
            "e".to_string(),
        ];
        let tok = GgufTokenizer::from_vocab(vocab);
        let ids = tok.encode("The cat sat.", true);
        assert_eq!(ids[0], 0); // <s>
        assert_eq!(ids[1], 2); // The
        assert_eq!(ids[2], 3); // " cat"
        assert_eq!(ids[3], 4); // " sat"
        assert_eq!(ids[4], 8); // "."
    }

    #[test]
    fn test_decode() {
        let vocab = vec![
            "<s>".to_string(),
            "</s>".to_string(),
            "The".to_string(),
            " cat".to_string(),
        ];
        let tok = GgufTokenizer::from_vocab(vocab);
        let text = tok.decode(&[0, 2, 3]);
        assert_eq!(text, "The cat");
    }

    #[test]
    fn test_byte_token() {
        let vocab = vec![
            "a".to_string(),
            "<0x20>".to_string(), // space
            "b".to_string(),
        ];
        let tok = GgufTokenizer::from_vocab(vocab);
        let text = tok.decode(&[0, 1, 2]);
        assert_eq!(text, "a b");
    }
}
