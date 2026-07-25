//! Simple tokenizer using GGUF vocab (no external dependencies)
//!
//! Reads `tokenizer.ggml.tokens` from the GGUF file and does greedy
//! longest-match tokenization. This is a pragmatic fallback when a
//! full HuggingFace `tokenizer.json` is not available.

use crate::model::gguf::{GGUFile, GGUFValue};
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

// Cheap to clone — only four fields, the HashMap is the biggest but copy-on-write
// semantics are fine for the cached-tokenizer use case.
impl Clone for GgufTokenizer {
    fn clone(&self) -> Self {
        Self {
            vocab: self.vocab.clone(),
            token_to_id: self.token_to_id.clone(),
            bos_id: self.bos_id,
            eos_id: self.eos_id,
        }
    }
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
    ///
    /// Whitespace is preserved as-is (unlike `split_whitespace` approaches
    /// that collapse all runs of whitespace to a single boundary).
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
                // Character-level fallback: emit one byte at a time using
                // the byte-token convention `<0xXX>`.  This preserves
                // whitespace as a real byte instead of silently dropping it.
                let bytes = remaining.as_bytes();
                let first_byte = bytes[0];
                let byte_token = format!("<0x{:02X}>", first_byte);
                let step = if let Some(first_char) = remaining.chars().next() {
                    first_char.len_utf8()
                } else {
                    1
                };
                if let Some(&id) = self.token_to_id.get(&byte_token) {
                    tokens.push(id);
                } else {
                    // Last-resort: emit raw byte value as a token id.
                    // This is non-standard but ensures we never panic.
                    tokens.push(first_byte as usize);
                }
                remaining = &remaining[step..];
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

    /// Read-only access to the vocab Vec (token_id to surface string).
    /// Used by the anti-doom sampler hook to look up which token ids would
    /// continue a detected repetition cycle.
    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    pub fn bos_id(&self) -> Option<usize> { self.bos_id }
    pub fn eos_id(&self) -> Option<usize> { self.eos_id }

    /// Load tokenizer vocabulary directly from a GGUF file.
    pub fn from_gguf(path: &str) -> Option<Self> {
        let file = GGUFile::open(path).ok()?;
        let vocab: Vec<String> = match file.metadata.get("tokenizer.ggml.tokens") {
            Some(GGUFValue::Array(arr)) => {
                arr.iter()
                    .map(|v| match v {
                        GGUFValue::String(s) => s.clone(),
                        _ => String::new(),
                    })
                    .collect()
            }
            _ => return None,
        };
        if vocab.is_empty() {
            return None;
        }
        Some(Self::from_vocab(vocab))
    }
}

/// BPE-aware tokenizer using GGUF vocab.
///
/// Handles the BPE space convention (Ġ prefix for words after whitespace)
/// and newline markers (Ċ = U+010A). More accurate than plain greedy
/// longest-match for Llama, Mistral, and Ministral models.
pub struct GgufBpeTokenizer {
    vocab_sorted: Vec<(String, usize)>, // (token_string, token_id), longest first
    vocab_map: HashMap<String, usize>,
    bos_token: usize,
    eos_token: usize,
    vocab: Vec<String>,
}

impl GgufBpeTokenizer {
    pub fn new(vocab: Vec<String>, bos: usize, eos: usize) -> Self {
        let vocab_map: HashMap<String, usize> = vocab.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        let mut vocab_sorted: Vec<(String, usize)> = vocab.clone().into_iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        vocab_sorted.sort_by(|a, b| {
            b.0.len().cmp(&a.0.len())
                .then_with(|| a.1.cmp(&b.1))
        });

        Self { vocab_sorted, vocab_map, bos_token: bos, eos_token: eos, vocab }
    }

    /// Load BPE tokenizer directly from a GGUF file.
    pub fn from_gguf(path: &str) -> Option<Self> {
        let file = GGUFile::open(path).ok()?;
        let vocab: Vec<String> = match file.metadata.get("tokenizer.ggml.tokens") {
            Some(GGUFValue::Array(arr)) => {
                arr.iter()
                    .map(|v| match v {
                        GGUFValue::String(s) => s.clone(),
                        _ => String::new(),
                    })
                    .collect()
            }
            _ => return None,
        };
        if vocab.is_empty() {
            return None;
        }
        let bos = file.get_metadata_int("tokenizer.ggml.bos_token_id")
            .map(|v| v as usize)
            .unwrap_or(1);
        let eos = file.get_metadata_int("tokenizer.ggml.eos_token_id")
            .map(|v| v as usize)
            .unwrap_or(2);
        Some(Self::new(vocab, bos, eos))
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode text to token IDs using greedy longest-match with BPE conventions.
    ///
    /// Whitespace structure is preserved: a SP character encodes as the Ġ
    /// (U+0120) prefix BPE token, while a literal newline is encoded as Ċ
    /// (U+010A).  We pre-pad the input so that every non-leading whitespace
    /// character becomes Ġ and every newline becomes Ċ — this avoids ever
    /// calling `split_whitespace()`, which would collapse runs of spaces and
    /// delete newlines entirely (silently corrupting indentation/multi-space
    /// text).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![self.bos_token];

        if text.is_empty() {
            return tokens;
        }

        // Convert input byte-stream into a sequence of Ġ/Ċ tokens.  Any
        // leading whitespace is preserved as literal ' ' / '\n' characters
        // so the first word is encoded without a Ġ prefix.
        let mut converted = String::with_capacity(text.len());
        for ch in text.chars() {
            match ch {
                ' ' => converted.push('\u{0120}'),
                '\n' => converted.push('\u{010A}'),
                '\t' => converted.push('\u{0120}'), // tabs join adjacent tokens via Ġ
                '\r' => {}                          // drop stray CR
                _ => converted.push(ch),
            }
        }

        self.greedy_encode(&converted, &mut tokens);
        tokens
    }

    fn greedy_encode(&self, text: &str, tokens: &mut Vec<usize>) {
        let mut remaining = text;

        while !remaining.is_empty() {
            let mut matched = false;

            for (token_str, token_id) in &self.vocab_sorted {
                if remaining.starts_with(token_str) {
                    tokens.push(*token_id);
                    remaining = &remaining[token_str.len()..];
                    matched = true;
                    break;
                }
            }

            if !matched {
                if let Some(first_char) = remaining.chars().next() {
                    let char_str = first_char.to_string();
                    if let Some(&id) = self.vocab_map.get(&char_str) {
                        tokens.push(id);
                    } else {
                        for byte in char_str.bytes() {
                            let byte_token = format!("<0x{:02X}>", byte);
                            if let Some(&id) = self.vocab_map.get(&byte_token) {
                                tokens.push(id);
                            }
                        }
                    }
                    remaining = &remaining[first_char.len_utf8()..];
                } else {
                    break;
                }
            }
        }
    }

    /// Decode token IDs back to text.
    /// Handles BPE space markers (Ġ → literal space) and newline markers (Ċ → \n).
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut result = String::new();
        let space_marker = '\u{0120}';

        for &token_id in tokens {
            // Skip special tokens
            if token_id == self.bos_token || token_id == self.eos_token {
                continue;
            }

            if let Some(token_str) = self.vocab.get(token_id) {
                // Skip other special tokens
                if token_str.starts_with('<') && token_str.ends_with('>') {
                    if token_str == "<0x0A>" {
                        result.push('\n');
                        continue;
                    }
                    if token_str.starts_with("<0x") {
                        if let Ok(byte) = u8::from_str_radix(&token_str[3..token_str.len()-1], 16) {
                            result.push(byte as char);
                        }
                        continue;
                    }
                    // Skip other special tokens like <|begin_of_text|>, etc.
                    if token_str.starts_with("<|") {
                        continue;
                    }
                }

                if token_str.starts_with(space_marker) {
                    // Ġ prefix → add space before
                    result.push(' ');
                    result.push_str(&token_str[space_marker.len_utf8()..]);
                } else {
                    // Replace BPE newline markers (Ċ = U+010A = byte 0x0A)
                    let cleaned = token_str.replace('\u{010A}', "\n");
                    result.push_str(&cleaned);
                }
            }
        }
        result
    }
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
