//! BPE tokenizer using GGUF vocab + merge rules.
//!
//! Reads `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` from the GGUF
//! file and applies proper GPT-2 style byte-level BPE tokenization. This
//! replaces the earlier greedy longest-match approach which produced wrong
//! token IDs for BPE models (Qwen, Llama, Mistral, etc.).

use crate::model::gguf::{GGUFile, GGUFValue};
use std::collections::HashMap;

/// Split a pre-token into (non-punct prefix, trailing punctuation run).
/// In the GPT-2 byte-mapped space, punctuation is anything that is not an
/// ASCII letter/digit and not a whitespace-mapped char (Ġ space, Ċ newline,
/// ċ tab).  E.g. `"assistant."` → `("assistant", ".")`, `"logue)."` →
/// `("logue", ").")`.
fn split_trailing_punct(s: &str) -> (String, String) {
    // Find the start of the trailing run of punctuation chars.
    let mut punct_start = s.len();
    for (i, c) in s.char_indices().rev() {
        if c.is_ascii_alphanumeric() || c == '\u{0120}' || c == '\u{010A}' || c == '\u{0109}' {
            break;
        }
        punct_start = i;
    }
    if punct_start == s.len() {
        (s.to_string(), String::new())
    } else {
        (s[..punct_start].to_string(), s[punct_start..].to_string())
    }
}

pub struct GgufTokenizer {
    /// token_id -> token_string
    vocab: Vec<String>,
    /// token_string -> token_id
    pub token_to_id: HashMap<String, usize>,
    /// BPE merge rules: "left right" -> merge rank (lower = higher priority)
    merge_ranks: HashMap<String, usize>,
    /// BOS token ID
    bos_id: Option<usize>,
    /// EOS token ID
    eos_id: Option<usize>,
}

impl Clone for GgufTokenizer {
    fn clone(&self) -> Self {
        Self {
            vocab: self.vocab.clone(),
            token_to_id: self.token_to_id.clone(),
            merge_ranks: self.merge_ranks.clone(),
            bos_id: self.bos_id,
            eos_id: self.eos_id,
        }
    }
}

impl GgufTokenizer {
    /// Build tokenizer from a list of vocab strings (extracted from GGUF).
    /// Without merge rules — falls back to character-level encoding.
    pub fn from_vocab(vocab_tokens: Vec<String>) -> Self {
        Self::from_vocab_and_merges(vocab_tokens, Vec::new())
    }

    /// Build tokenizer from vocab strings and BPE merge rules.
    /// Each merge rule is a "left right" string (space-separated).
    pub fn from_vocab_and_merges(vocab_tokens: Vec<String>, merges: Vec<String>) -> Self {
        let mut token_to_id = HashMap::with_capacity(vocab_tokens.len());
        for (i, tok) in vocab_tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), i);
        }

        // Build merge rank map: "left right" -> rank (index in merge list)
        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            merge_ranks.insert(merge.clone(), rank);
        }

        // Detect common special-token names.  We prefer the explicit
        // GGUF metadata (`tokenizer.ggml.bos_token_id` /
        // `tokenizer.ggml.eos_token_id`) when present, otherwise we
        // auto-detect by looking for known marker tokens in the vocab.
        //
        // Important: for models that use ChatML (`<|im_start|>` /
        // `<|im_end|>`) as turn markers — Qwen, Ornith, Llama-3 — these
        // tokens are NOT BOS/EOS for generation.  The real BOS is
        // usually absent (the prompt starts with `<|im_start|>` directly)
        // or is `<|begin_of_text|>` for Llama-3.  We therefore do NOT
        // auto-detect `<|im_start|>` as BOS — leaving it `None` means
        // `encode(..., add_bos=true)` will not add an unwanted prefix
        // token before the user's prompt.
        let bos_id = ["<s>", "<|begin_of_text|>", "<|startoftext|>", "[BOS]"]
            .iter()
            .find_map(|&s| token_to_id.get(s).copied());

        let eos_id = [
            "</s>",
            "<|end_of_text|>",
            "<|eot_id|>",
            "<|im_end|>",
            "[EOS]",
        ]
        .iter()
        .find_map(|&s| token_to_id.get(s).copied());

        Self {
            vocab: vocab_tokens,
            token_to_id,
            merge_ranks,
            bos_id,
            eos_id,
        }
    }

    /// Encode text to token IDs using GPT-2 byte mapping + BPE merges.
    ///
    /// Steps:
    /// 1. Convert input bytes to GPT-2 byte-mapped Unicode chars
    /// 2. Pre-tokenize: split on whitespace boundaries (Ġ prefix convention)
    /// 3. For each pre-token, apply BPE merges in rank order
    /// 4. Look up each resulting subword in the vocab
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<usize> {
        if std::env::var("LEAFCUTTER_TOKENIZER_DEBUG").is_ok() {
            eprintln!(
                "[TOKENIZER] encode called: text={:?}, add_bos={}, merge_ranks={}, vocab={}",
                text, add_bos, self.merge_ranks.len(), self.token_to_id.len()
            );
        }
        let mut tokens = Vec::new();

        if add_bos {
            if let Some(bos) = self.bos_id {
                tokens.push(bos);
            }
        }

        if text.is_empty() {
            return tokens;
        }

        // Step 1: Convert input bytes to GPT-2 byte-mapped Unicode chars.
        // GPT-2 byte-to-unicode: printable ASCII (33-126) stays as-is,
        // everything else (0-32, 127-255) → chr(byte + 256).
        let mut converted = String::with_capacity(text.len() * 2);
        for &byte in text.as_bytes() {
            let c = if byte >= 33 && byte <= 126 {
                byte as char
            } else {
                char::from_u32(byte as u32 + 256).unwrap_or(byte as char)
            };
            converted.push(c);
        }

        // Step 2: If we have merge rules, use proper BPE.
        // Otherwise fall back to greedy longest-match.
        if self.merge_ranks.is_empty() {
            tokens.extend(self.greedy_encode(&converted));
            return tokens;
        }

        // Step 3: Pre-tokenize. GPT-2 BPE splits on whitespace boundaries.
        // Words are: sequences of non-whitespace chars. The whitespace itself
        // becomes a Ġ-prefixed word (GPT-2 convention: space before word).
        // We split the converted string into pre-tokens by finding Ġ boundaries.
        //
        // SPECIAL TOKENS: Qwen/Llama/Ornith embed ChatML markers like
        // `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, ``, etc.
        // These are stored as single-token entries in the vocab and must
        // NEVER be BPE-encoded — the literal `<`, `|`, `_` characters would
        // otherwise explode into 6-8 separate tokens and corrupt the model's
        // understanding.  We split on `<|...|>` boundaries as well.
        let pre_tokens = self.pretokenize(&converted);
        if std::env::var("LEAFCUTTER_TOKENIZER_DEBUG").is_ok() {
            eprintln!("[TOKENIZER] pre_tokens: {:?}", pre_tokens);
        }

        // Step 4: Apply BPE to each pre-token and collect token IDs.
        //
        // If a pre-token IS a special token (matches a vocab entry like
        // `<|im_start|>` or ``), emit it directly — BPE would otherwise
        // shatter it into 6-12 subword IDs that don't mean anything to the
        // model.  This matches how HuggingFace's GPT-2 tokenizer treats
        // added special tokens.
        for word in &pre_tokens {
            // Special-token fast path: if the whole pre-token is in the
            // vocab, emit it as a single ID.  This handles `<|im_start|>`,
            // ``, `<|endoftext|>`, etc.  We check using the byte-mapped
            // form (with Ġ/Ċ) so leading-whitespace words still match.
            let normalized = word.clone();
            if let Some(&id) = self.token_to_id.get(&normalized) {
                tokens.push(id);
                continue;
            }
            // Otherwise apply BPE.
            let word_tokens = self.bpe_apply(word);
            for tok_str in word_tokens {
                if let Some(&id) = self.token_to_id.get(&tok_str) {
                    tokens.push(id);
                } else {
                    // Fallback: try individual chars
                    for c in tok_str.chars() {
                        let char_str = c.to_string();
                        if let Some(&id) = self.token_to_id.get(&char_str) {
                            tokens.push(id);
                        } else {
                            let byte = if c as u32 >= 256 {
                                (c as u32 - 256) as u8
                            } else {
                                c as u8
                            };
                            let byte_token = format!("<0x{:02X}>", byte);
                            if let Some(&id) = self.token_to_id.get(&byte_token) {
                                tokens.push(id);
                            } else {
                                tokens.push(byte as usize);
                            }
                        }
                    }
                }
            }
        }

        tokens
    }

    /// Pre-tokenize: split the GPT-2 byte-mapped string into word-level pieces.
    ///
    /// GPT-2 convention: spaces are part of the following word (Ġ prefix).
    /// So "Hello world" → ["Hello", "Ġworld"] in byte-mapped form.
    /// We split at every Ġ (U+0120, represents space) boundary, keeping the Ġ
    /// with the following word. Newlines (Ċ, U+010A) also start a new word.
    ///
    /// Additionally, we split on special-token boundaries of the form
    /// `<|...|>` (Qwen ChatML markers, Llama-3 header markers, etc.)
    /// so each special token is a separate pre-token and won't be
    /// BPE-encoded into its constituent characters.
    fn pretokenize(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut i = 0;
        while i < n {
            let c = chars[i];
            // Whitespace boundary.  The GPT-2/Mistral pretokenizer regex
            // splits trailing punctuation off a word and keeps a punctuation
            // run + trailing newlines together (e.g. `assistant.\n` →
            // `assistant` + `.\n`, so BPE emits `.\n` as the single token
            // `.\n`).  So when a newline (Ċ, U+010A) follows a word that ends
            // in punctuation, the punct run is detached and merged with the
            // newline.
            if c == '\u{0120}' || c == '\u{010A}' || c == '\u{0109}' {
                // Multiple newlines group into one pre-token (tekken/GPT-2
                // regex: `\s+` / `[\s\p{Z}]+[\r\n]`), so a `.\n\n` stays
                // together as `.\n\n` and can BPE into the single vocab token.
                if c == '\u{010A}' && !current.is_empty() && current.ends_with('\u{010A}') {
                    current.push(c);
                    i += 1;
                    continue;
                }
                if c == '\u{010A}' && !current.is_empty() {
                    let (prefix, punct) = split_trailing_punct(&current);
                    if !punct.is_empty() {
                        // `word` + `.\n` → push `word`, keep `.\n` as current
                        // so subsequent chars append to the newline group.
                        if !prefix.is_empty() {
                            result.push(prefix);
                        }
                        current = punct;
                        current.push(c);
                        i += 1;
                        continue;
                    }
                }
                if !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
                current.push(c);
                i += 1;
                continue;
            }
            // Special-token boundary: `<|...|>` (Qwen/Llama ChatML) or
            // `[...]` (Ministral) markers.  The whole marker must be one
            // pre-token so it ends up in the vocab as a single ID.  We only
            // emit a special pre-token when the exact bracketed string exists
            // in the vocab — e.g. `[INST]`/`[SYSTEM_PROMPT]` are single
            // tokens, but `[THINK]` is NOT (HF splits it into `[`, `TH`,
            // `INK`, `]`), so it must stay in the word stream for BPE to
            // shatter it identically.
            let (is_marker, close_len) = if c == '[' {
                (true, 1usize)
            } else if c == '<' && i + 2 < n && chars[i + 1] == '|' {
                (true, 2usize)
            } else {
                (false, 0)
            };
            if is_marker {
                let mut j = i + 1;
                if c == '[' {
                    while j < n && chars[j] != ']' {
                        j += 1;
                    }
                } else {
                    while j < n - 1 && !(chars[j] == '|' && chars[j + 1] == '>') {
                        j += 1;
                    }
                }
                let token_len = if c == '[' {
                    if j < n { j + 1 - i } else { 0 }
                } else {
                    if j < n - 1 { j + 2 - i } else { 0 }
                };
                if token_len > 0 {
                    let token: String = chars[i..i + token_len].iter().collect();
                    if self.token_to_id.contains_key(&token) {
                        // Flush any accumulated word first.
                        if !current.is_empty() {
                            result.push(std::mem::take(&mut current));
                        }
                        result.push(token);
                        i += token_len;
                        continue;
                    }
                }
                // Not a vocab special token — fall through to treat the
                // opening bracket as a normal character.
            }
            // A newline group is standalone (regex `\s+`); don't let the next
            // word absorb the leading newline(s).  Spaces (Ġ) still attach to
            // the following word per GPT-2 convention.
            if current.contains('\u{010A}') {
                result.push(std::mem::take(&mut current));
            }
            current.push(c);
            i += 1;
        }
        if !current.is_empty() {
            result.push(current);
        }
        result
    }

    /// Apply BPE merges to a single pre-token (word).
    ///
    /// Algorithm:
    /// 1. Start with the word split into individual characters
    /// 2. Repeatedly find the adjacent pair with the lowest merge rank
    /// 3. Merge that pair and repeat until no more merges apply
    /// 4. Return the resulting subwords
    fn bpe_apply(&self, word: &str) -> Vec<String> {
        // Start with individual characters as tokens
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if symbols.len() < 2 {
            return symbols;
        }

        loop {
            // Find the pair with the lowest merge rank
            let mut min_rank: Option<usize> = None;
            let mut merge_idx: usize = 0;

            for i in 0..symbols.len() - 1 {
                let pair = format!("{} {}", symbols[i], symbols[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    match min_rank {
                        None => {
                            min_rank = Some(rank);
                            merge_idx = i;
                        }
                        Some(r) if rank < r => {
                            min_rank = Some(rank);
                            merge_idx = i;
                        }
                        _ => {}
                    }
                }
            }

            // No more merges to apply
            let min_rank = match min_rank {
                Some(r) => r,
                None => break,
            };

            // Merge the pair at merge_idx
            let merged = format!("{}{}", symbols[merge_idx], symbols[merge_idx + 1]);
            symbols[merge_idx] = merged;
            symbols.remove(merge_idx + 1);

            // Safety: if we somehow get into an infinite loop, break
            // (shouldn't happen with proper merge rules, but guards against
            // malformed merge data that merges to the same string)
            if min_rank > self.merge_ranks.len() {
                break;
            }
        }

        symbols
    }

    /// Greedy longest-match encoding (fallback when no merge rules available).
    fn greedy_encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let mut matched = false;
            let max_len = remaining.len().min(64);
            for len in (1..=max_len).rev() {
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
                if let Some(first_char) = remaining.chars().next() {
                    let char_str = first_char.to_string();
                    if let Some(&id) = self.token_to_id.get(&char_str) {
                        tokens.push(id);
                    } else {
                        let byte = if first_char as u32 >= 256 {
                            (first_char as u32 - 256) as u8
                        } else {
                            first_char as u8
                        };
                        let byte_token = format!("<0x{:02X}>", byte);
                        if let Some(&id) = self.token_to_id.get(&byte_token) {
                            tokens.push(id);
                        } else {
                            tokens.push(byte as usize);
                        }
                    }
                    remaining = &remaining[first_char.len_utf8()..];
                } else {
                    break;
                }
            }
        }
        tokens
    }

    /// Decode token IDs back to a string.
    ///
    /// Handles byte tokens (`<0xXX>`) and skips other special tokens.
    /// Reverses the GPT-2 byte-to-unicode mapping: chars in 0x100-0x1FF are
    /// converted back to their original bytes (Ġ U+0120 → 0x20 space, etc.).
    /// All output is accumulated as raw bytes then decoded as UTF-8 (lossy).
    pub fn decode(&self, tokens: &[usize]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(tokens)).to_string()
    }

    /// Decode token IDs to the raw UTF-8 byte stream, without the lossy
    /// UTF-8 pass.  Streaming callers can accumulate these bytes and only
    /// convert to a `String` when a complete UTF-8 sequence is available —
    /// decoding per-token mangles multi-byte chars (e.g. emoji) that split
    /// across byte-level tokens.
    pub fn decode_bytes(&self, tokens: &[usize]) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();

        for &id in tokens {
            if id >= self.vocab.len() {
                continue;
            }
            let piece = &self.vocab[id];

            // Byte token: <0xXX>
            if piece.starts_with("<0x") && piece.len() == 6 && piece.ends_with('>') {
                if let Ok(byte) = u8::from_str_radix(&piece[3..5], 16) {
                    bytes.push(byte);
                }
                continue;
            }

            // Skip other special tokens (anything wrapped in < >)
            if piece.starts_with('<') && piece.ends_with('>') {
                continue;
            }

            // Reverse GPT-2 byte-to-unicode mapping for each char in the piece.
            //
            // The real GPT-2 byte-to-unicode map (as used by llama.cpp and HF)
            // maps only the "non-printable" bytes to U+0100-U+0143, in a fixed
            // order.  It is NOT "every byte -> chr(byte+256)":
            //   - bytes 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF map to THEMSELVES
            //   - the remaining 68 bytes (0x00-0x20, 0x7F-0xA0, 0xAD) map to
            //     U+0100..U+0143 in ascending byte order
            // A naive `cp - 256` for all of 0x100-0x1FF corrupts genuine
            // Latin-1/Latin-Extended chars stored as bytes in the vocab.
            for c in piece.chars() {
                let cp = c as u32;
                if (0x21..=0x7E).contains(&cp)
                    || (0xA1..=0xAC).contains(&cp)
                    || (0xAE..=0xFF).contains(&cp)
                {
                    // Printable byte stored as its own codepoint
                    bytes.push(cp as u8);
                } else if (0x100..=0x143).contains(&cp) {
                    // Non-printable byte stored as chr(byte + 256) in order:
                    // 0x100..0x120 -> 0x00..0x20, 0x121..0x142 -> 0x7F..0xA0,
                    // 0x143 -> 0xAD.
                    let n = (cp - 0x100) as usize;
                    let b = if n < 33 {
                        n as u8
                    } else if n < 67 {
                        (0x7F + (n - 33)) as u8
                    } else {
                        0xAD
                    };
                    bytes.push(b);
                } else {
                    // Regular Unicode char — encode as UTF-8
                    let mut buf = [0u8; 4];
                    let s = c.encode_utf8(&mut buf);
                    bytes.extend_from_slice(s.as_bytes());
                }
            }
        }

        bytes
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

    pub fn bos_id(&self) -> Option<usize> {
        self.bos_id
    }
    pub fn eos_id(&self) -> Option<usize> {
        self.eos_id
    }

    /// Load tokenizer vocabulary and merge rules directly from a GGUF file.
    pub fn from_gguf(path: &str) -> Option<Self> {
        let file = GGUFile::open(path).ok()?;
        let vocab: Vec<String> = match file.metadata.get("tokenizer.ggml.tokens") {
            Some(GGUFValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    GGUFValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => return None,
        };
        if vocab.is_empty() {
            return None;
        }

        // Read BPE merge rules if available
        let merges: Vec<String> = match file.metadata.get("tokenizer.ggml.merges") {
            Some(GGUFValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    GGUFValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => Vec::new(),
        };

        let mut tok = Self::from_vocab_and_merges(vocab, merges);

        // Override auto-detected BOS/EOS with explicit metadata when
        // present.  This is important for Ornith where the GGUF does NOT
        // include `bos_token_id` (the prompt starts with `<|im_start|>`
        // directly) and where `<|im_end|>` is the EOS.
        if let Some(bos) = file
            .get_metadata_int("tokenizer.ggml.bos_token_id")
            .map(|v| v as usize)
        {
            if bos < tok.vocab.len() {
                tok.bos_id = Some(bos);
            }
        }
        if let Some(eos) = file
            .get_metadata_int("tokenizer.ggml.eos_token_id")
            .map(|v| v as usize)
        {
            if eos < tok.vocab.len() {
                tok.eos_id = Some(eos);
            }
        }

        Some(tok)
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

        // Split out special-token markers (`[...]` for Ministral, `<|...|>`
        // for Qwen/Llama) that are exact vocab entries so greedy matching
        // emits them as single IDs.  Without this, `[/SYSTEM_PROMPT]` and
        // `[/INST]` get shattered (e.g. `.`+`[` merges into `.[`=8925 first)
        // and the model sees garbage in place of its control tokens.
        let chars: Vec<char> = converted.chars().collect();
        let n = chars.len();
        let mut i = 0;
        let mut plain_start = 0;
        while i < n {
            let c = chars[i];
            let is_marker = c == '[' || (c == '<' && i + 2 < n && chars[i + 1] == '|');
            if !is_marker {
                i += 1;
                continue;
            }
            let mut j = i + 1;
            if c == '[' {
                while j < n && chars[j] != ']' {
                    j += 1;
                }
                if j >= n {
                    i += 1;
                    continue;
                }
            } else {
                while j < n - 1 && !(chars[j] == '|' && chars[j + 1] == '>') {
                    j += 1;
                }
                if j >= n - 1 {
                    i += 1;
                    continue;
                }
                j += 1;
            }
            let token: String = chars[i..=j].iter().collect();
            if let Some(&id) = self.vocab_map.get(&token) {
                let seg: String = chars[plain_start..i].iter().collect();
                self.greedy_encode(&seg, &mut tokens);
                tokens.push(id);
                i = j + 1;
                plain_start = i;
            } else {
                i += 1;
            }
        }
        let tail: String = chars[plain_start..].iter().collect();
        self.greedy_encode(&tail, &mut tokens);
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
    /// Reverses GPT-2 byte-to-unicode mapping, accumulates as raw bytes, decodes as UTF-8.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut bytes: Vec<u8> = Vec::new();

        for &token_id in tokens {
            // Skip special tokens
            if token_id == self.bos_token || token_id == self.eos_token {
                continue;
            }

            if let Some(token_str) = self.vocab.get(token_id) {
                // Skip other special tokens
                if token_str.starts_with('<') && token_str.ends_with('>') {
                    if token_str == "<0x0A>" {
                        bytes.push(b'\n');
                        continue;
                    }
                    if token_str.starts_with("<0x") {
                        if let Ok(byte) = u8::from_str_radix(&token_str[3..token_str.len()-1], 16) {
                            bytes.push(byte);
                        }
                        continue;
                    }
                    // Skip other special tokens like <|begin_of_text|>, etc.
                    if token_str.starts_with("<|") {
                        continue;
                    }
                }

                // Reverse GPT-2 byte-to-unicode mapping for each char
                // (see GgufTokenizer::decode for the exact map: only the 68
                // non-printable bytes map to U+0100-U+0143, in order).
                for c in token_str.chars() {
                    let cp = c as u32;
                    if (0x21..=0x7E).contains(&cp)
                        || (0xA1..=0xAC).contains(&cp)
                        || (0xAE..=0xFF).contains(&cp)
                    {
                        bytes.push(cp as u8);
                    } else if (0x100..=0x143).contains(&cp) {
                        let n = (cp - 0x100) as usize;
                        let b = if n < 33 {
                            n as u8
                        } else if n < 67 {
                            (0x7F + (n - 33)) as u8
                        } else {
                            0xAD
                        };
                        bytes.push(b);
                    } else {
                        let mut buf = [0u8; 4];
                        let s = c.encode_utf8(&mut buf);
                        bytes.extend_from_slice(s.as_bytes());
                    }
                }
            }
        }

        String::from_utf8_lossy(&bytes).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenize() {
        // Vocab entries use GPT-2 byte mapping: space → Ġ (U+0120)
        let vocab = vec![
            "<s>".to_string(),
            "</s>".to_string(),
            "The".to_string(),
            "Ġcat".to_string(),   // " cat" → Ġcat
            "Ġsat".to_string(),   // " sat" → Ġsat
            "Ġon".to_string(),    // " on" → Ġon
            "Ġthe".to_string(),   // " the" → Ġthe
            "Ġmat".to_string(),   // " mat" → Ġmat
            ".".to_string(),
            "ĠT".to_string(),
            "h".to_string(),
            "e".to_string(),
        ];
        let tok = GgufTokenizer::from_vocab(vocab);
        let ids = tok.encode("The cat sat.", true);
        assert_eq!(ids[0], 0); // <s>
        assert_eq!(ids[1], 2); // The
        assert_eq!(ids[2], 3); // "Ġcat" (was " cat")
        assert_eq!(ids[3], 4); // "Ġsat" (was " sat")
        assert_eq!(ids[4], 8); // "."
    }

    #[test]
    fn test_decode() {
        // Vocab entries use GPT-2 byte mapping: space → Ġ (U+0120)
        let vocab = vec![
            "<s>".to_string(),
            "</s>".to_string(),
            "The".to_string(),
            "Ġcat".to_string(),   // " cat" → Ġcat
        ];
        let tok = GgufTokenizer::from_vocab(vocab);
        let text = tok.decode(&[0, 2, 3]);
        assert_eq!(text, "The cat");  // Ġ → space in decode
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
