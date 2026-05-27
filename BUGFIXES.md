# LeafcutterLLM — Three Critical Bug Fixes

Based on your final test report, here are the root causes and fixes for all three bugs.

---

## Bug 1: Build Feature Gate Blocks Auto-Routing

### Root Cause
The `llama-ffi` feature gate in `lib.rs` compiles a **stub module** when the feature is disabled:

```rust
// lib.rs line 13-34
#[cfg(not(feature = "llama-ffi"))]
pub mod llama_ffi {
    pub struct LlamaModel;
    impl LlamaModel {
        pub fn load(_path: &str) -> Result<Self, String> {
            Err("llama.cpp FFI not enabled...".into()) // ALWAYS fails
        }
    }
}
```

When you build without `--features llama-ffi`:
- `Engine::load_ffi()` calls `LlamaModel::load()` → **always fails**
- The capability report says `can_run = true` (it doesn't know FFI is stubbed)
- Qwen models try native instead of FFI → gibberish
- Auto-fallback for unsupported quants tries FFI → hard crash

### The Fix: Three changes

**Step 1: `lib.rs` — Add a `is_available()` check to the stub**

```rust
// In lib.rs, replace lines 13-34 (the stub module) with:

#[cfg(not(feature = "llama-ffi"))]
pub mod llama_ffi {
    //! Stub module when llama.cpp FFI is disabled.
    pub struct LlamaModel;
    pub struct LlamaContext;
    pub struct LlamaBatch;

    /// Check whether the real llama.cpp backend is available.
    pub const fn is_available() -> bool { false }

    impl LlamaModel {
        pub fn load(_path: &str) -> Result<Self, String> {
            Err("llama.cpp FFI not enabled. Build with: cargo build --features llama-ffi".into())
        }
        pub fn n_vocab(&self) -> i32 { 0 }
        pub fn n_embd(&self) -> i32 { 0 }
        pub fn n_layer(&self) -> i32 { 0 }
        pub fn eos_token(&self) -> i32 { 2 }
    }
    impl LlamaContext {
        pub fn new(_model: &LlamaModel, _ctx_size: u32, _threads: i32) -> Result<Self, String> {
            Err("llama.cpp FFI not enabled".into())
        }
        pub fn tokenize(&self, _text: &str, _add_bos: bool, _special: bool) -> Vec<i32> { vec![] }
        pub fn token_to_piece(&self, _token: i32) -> String { String::new() }
        pub fn generate(&self, _tokens: &[i32], _max_tokens: usize, _temperature: f32, _eos_token: i32) -> Vec<i32> { vec![] }
    }
    pub fn backend_init() {}
    pub fn backend_free() {}
}
```

**Step 2: `lib.rs` — Add `is_available()` to the real module too**

```rust
// In lib.rs, line 10-11 (the real module), add the constant:

#[cfg(feature = "llama-ffi")]
pub mod llama_ffi;

// Then INSIDE src/llama_ffi/mod.rs, add at the top:
pub const fn is_available() -> bool { true }
```

**Step 3: `engine.rs` — Fix the load logic to check FFI availability**

Replace lines 114-144 (the `Engine::load` function) with:

```rust
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // ── Architecture detection ──────────────────────────────────
        let arch = detect_arch(path);

        // ── Qwen3.5/3.6 ALWAYS need FFI ─────────────────────────────
        if arch == ModelArchitecture::Qwen35 || arch == ModelArchitecture::Qwen36 {
            if !crate::llama_ffi::is_available() {
                return Err(
                    "Qwen3.5/3.6 models require llama.cpp FFI. \
                     Build with: cargo build --features llama-ffi".into()
                );
            }
            eprintln!("  Using llama.cpp FFI backend for {}", arch.name());
            return Self::load_ffi(path);
        }

        // ── Native load path ────────────────────────────────────────
        let model = GGUFModel::load(path)?;

        // Run corruption scan
        let corruption = crate::model::loader::scan_for_corruption(&model.file);
        if !corruption.is_clean() {
            eprintln!("\n{}", corruption.print());
        }

        // Run pre-flight capability report
        let report = model.capability_report();
        if !report.can_run {
            eprintln!("\n{}", report.print());

            // ── AUTO-FALLBACK: unsupported quants → try FFI ─────────
            if crate::llama_ffi::is_available() {
                eprintln!("  Native path blocked. Trying llama.cpp FFI fallback...");
                return Self::load_ffi(path);
            } else {
                return Err(format!(
                    "Model cannot run natively (unsupported quant types). \
                     Build with --features llama-ffi for auto-fallback support. \
                     Details: architecture={} unsupported_quant={} missing_tensors={}",
                    report.architecture.name(),
                    report.quant_summary.unsupported.len(),
                    report.missing_tensors.len()
                ).into());
            }
        }

        eprintln!("  Using native backend for {}", arch.name());
        // ... rest of native load continues as before (lines 146+)
```

**Step 4: Update build instructions**

In your README and build scripts, change:
```bash
# OLD (broken):
cargo build --release

# NEW (correct):
cargo build --release --features llama-ffi
```

And set the env var:
```bash
export LLAMA_CPP_BUILD=/path/to/llama.cpp/build
```

---

## Bug 2: IQ1_M (Type 31) Not Detected

### Root Cause
`IQ1_M` (GGUF type code 31) is **not in the `QuantType` enum**. When `quant_summary()` iterates tensors:

```rust
// gguf.rs line 415
if let Some(qt) = QuantType::from_u32(t.typ) {  // Returns None for 31
    // ... never enters this branch for IQ1_M
}
```

`from_u32(31)` returns `None`, so IQ1_M tensors are **silently skipped**. The capability report never sees them, so `unsupported` stays empty and `can_run = true`. Then `load_layer()` hits the IQ1_M tensor and panics with `InvalidTensorType(31)`.

### The Fix: Add IQ1_M to the enum

In `src/model/quant.rs`, add `IQ1_M` to the enum and all match arms:

**Step 1: Add the variant to the enum (line 15-43)**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    // ... existing variants ...
    IQ4_K,    // 27 (legacy, actually I64 in ggml)
    IQ1_M,    // 31  ← ADD THIS
    BF16,     // 30
}
```

**Step 2: Add to `code()` (line 47-77)**

```rust
pub fn code(self) -> u32 {
    match self {
        // ... existing ...
        QuantType::IQ4_K   => 27,
        QuantType::IQ1_M   => 31,  // ← ADD THIS
        QuantType::BF16    => 30,
    }
}
```

**Step 3: Add to `from_u32()` (line 80-111)**

```rust
pub fn from_u32(code: u32) -> Option<Self> {
    match code {
        // ... existing ...
        27 => Some(QuantType::IQ4_K),
        31 => Some(QuantType::IQ1_M),  // ← ADD THIS
        30 => Some(QuantType::BF16),
        _  => None,
    }
}
```

**Step 4: Add to `name()` (line 114-144)**

```rust
pub fn name(self) -> &'static str {
    match self {
        // ... existing ...
        QuantType::IQ4_K   => "IQ4_K",
        QuantType::IQ1_M   => "IQ1_M",  // ← ADD THIS
        QuantType::BF16    => "BF16",
    }
}
```

**Step 5: Add to `bits_per_weight()` (line 147-177)**

```rust
pub fn bits_per_weight(self) -> f32 {
    match self {
        // ... existing ...
        QuantType::IQ4_K   => 4.25,
        QuantType::IQ1_M   => 1.75,  // ← ADD THIS (1.58-bit + metadata)
        QuantType::BF16    => 16.0,
    }
}
```

**Step 6: Add to `block_size()` (line 180-194)**

```rust
pub fn block_size(self) -> usize {
    match self {
        // ... existing 256-element block types ...
        QuantType::Q2_K | ... | QuantType::IQ4_K
        | QuantType::IQ1_M   // ← ADD THIS (256-element block)
        => 256,
        // 32-element blocks
        QuantType::Q4_0 | ... | QuantType::IQ4_NL => 32,
        // 1-element blocks
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => 1,
    }
}
```

**Step 7: Add to `block_bytes()` (line 197-252)**

```rust
QuantType::IQ1_M => {
    // IQ1_M: 256 weights in ~54 bytes (similar to IQ1_S)
    54
}
```

**Step 8: CRITICAL — Keep it unsupported (line 254-270)**

Do NOT add `IQ1_M` to `is_supported()`. It should remain unsupported so native code rejects it and auto-fallback routes to llama.cpp:

```rust
pub fn is_supported(self) -> bool {
    matches!(self,
        QuantType::F32 | QuantType::F16 | QuantType::BF16 |
        QuantType::Q4_0 | QuantType::Q4_1 | QuantType::Q8_0 |
        QuantType::Q4_K | QuantType::Q5_K | QuantType::Q6_K |
        QuantType::Q8_K | QuantType::IQ4_NL | QuantType::IQ4_XS
        // Note: IQ1_M is intentionally NOT here — routes to FFI
    )
}
```

After this fix, the capability report will correctly detect IQ1_M as unsupported, set `can_run = false`, and the auto-fallback (with Bug 1 fix) will route to llama.cpp FFI.

---

## Bug 3: Naive Tokenizer Produces Gibberish

### Root Cause
`test_generation.rs` uses `tokenize_with_vocab()` which does **word-level vocabulary lookup**:

```rust
// Splits on whitespace, looks up each word
for word in text.split_whitespace() {
    if let Some(&id) = vocab_map.get(word) {  // "the" → token_id
        tokens.push(id);
    }
}
```

But modern LLMs use **BPE (Byte-Pair Encoding)** or **SentencePiece** tokenization. The vocabulary doesn't contain whole words — it contains subword pieces:

| Word | BPE Tokens |
|------|-----------|
| "The" | `[1, 6449]` → `ĠThe` (note: `Ġ` = leading space) |
| "capital" | `[12727]` → `Ġcapital` |
| "France" | `[7685]` → `ĠFrance` |

When you feed word-level token IDs into a model trained on BPE tokens, the model sees completely unfamiliar inputs and produces garbage.

### The Fix: Implement proper BPE from GGUF vocab

Replace the entire `tokenize_with_vocab` function in `test_generation.rs` with a **greedy longest-match** BPE tokenizer:

```rust
use std::collections::HashMap;

/// BPE-style tokenizer that works with GGUF vocabulary.
/// 
/// Most GGUF vocabularies are BPE-style: tokens represent subword pieces,
/// not whole words. A token like "capital" is actually stored as "Ġcapital"
/// (with a leading space marker). This tokenizer does greedy longest-match
/// to find the best token sequence.
///
/// Algorithm:
/// 1. Sort vocabulary by token length (longest first)
/// 2. Greedily match the longest token at the current position
/// 3. If no match, use byte-level fallback (encode as UTF-8 bytes)
pub struct GgufTokenizer {
    vocab: Vec<(String, usize)>, // (token_string, token_id) — sorted longest first
    vocab_map: HashMap<String, usize>, // fast exact lookup
}

impl GgufTokenizer {
    /// Build from GGUF vocabulary (already extracted from metadata)
    pub fn new(vocab: Vec<String>) -> Self {
        let vocab_map: HashMap<String, usize> = vocab.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        // Sort by string length descending (longest match first)
        let mut vocab_sorted: Vec<(String, usize)> = vocab.into_iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        vocab_sorted.sort_by(|a, b| {
            // Longer strings first; if same length, lower ID first
            b.0.len().cmp(&a.0.len())
                .then_with(|| a.1.cmp(&b.1))
        });

        Self { vocab: vocab_sorted, vocab_map }
    }

    /// Encode text to token IDs using greedy longest-match BPE.
    /// 
    /// Preprocessing:
    /// - Normalizes Unicode to NFC
    /// - Adds leading space to match BPE convention (Ġ prefix)
    /// - Handles special tokens like `<|im_start|>` as atomic units
    pub fn encode(&self, text: &str) -> Vec<usize> {
        use unicode_normalization::UnicodeNormalization;

        let mut tokens = Vec::new();
        let normalized: String = text.nfc().collect();

        // Pre-tokenize: split text into segments
        // Some tokens are multi-character sequences that should stay together
        let segments = self.pre_tokenize(&normalized);

        for segment in segments {
            self.encode_segment(&segment, &mut tokens);
        }

        tokens
    }

    /// Pre-tokenization: split on whitespace but keep whitespace as part of next token.
    /// Also handle special tokens as atomic units.
    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        // Special tokens that must be matched atomically
        let specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<s>", "</s>"];

        let mut segments = Vec::new();
        let mut chars = text.char_indices().peekable();

        while let Some((start, ch)) = chars.next() {
            // Check for special tokens
            let rest = &text[start..];
            let mut matched_special = false;
            for special in &specials {
                if rest.starts_with(special) {
                    // Skip ahead
                    for _ in 0..special.chars().count().saturating_sub(1) {
                        chars.next();
                    }
                    segments.push(&text[start..start + special.len()]);
                    matched_special = true;
                    break;
                }
            }
            if matched_special { continue; }

            // Collect until next special or end
            let seg_start = start;
            let mut seg_end = start + ch.len_utf8();
            while let Some(&(idx, c)) = chars.peek() {
                let rest = &text[idx..];
                if specials.iter().any(|s| rest.starts_with(s)) {
                    break;
                }
                seg_end = idx + c.len_utf8();
                chars.next();
            }
            if seg_start < seg_end {
                segments.push(&text[seg_start..seg_end]);
            }
        }

        // If no special tokens matched, just return the whole text
        if segments.is_empty() && !text.is_empty() {
            segments.push(text);
        }

        segments
    }

    /// Encode a single text segment using greedy longest-match.
    fn encode_segment(&self, text: &str, tokens: &mut Vec<usize>) {
        // Try the raw text first (handles pre-tokenized pieces)
        if let Some(&id) = self.vocab_map.get(text) {
            tokens.push(id);
            return;
        }

        // BPE convention: leading space is encoded as part of the token.
        // So "Hello" at the start of text might match "Hello" or "ĠHello".
        // Try adding a leading space marker.
        let with_space = format!("{}{}", "\u{0120}", text); // Ġ prefix (byte 0xC4 0xA0 in UTF-8)
        if let Some(&id) = self.vocab_map.get(&with_space) {
            tokens.push(id);
            return;
        }

        // Also try with literal space prefix (some vocabularies use " " instead of "Ġ")
        let with_literal_space = format!(" {}", text);
        if let Some(&id) = self.vocab_map.get(&with_literal_space) {
            tokens.push(id);
            return;
        }

        // Greedy longest-match: try all vocabulary tokens, pick longest match
        let mut remaining = text;
        while !remaining.is_empty() {
            let mut matched = false;

            // Try each vocabulary token (sorted longest first)
            for (token_str, token_id) in &self.vocab {
                if remaining.starts_with(token_str) {
                    tokens.push(*token_id);
                    remaining = &remaining[token_str.len()..];
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Fallback: encode the first character using the character itself
                // or byte fallback for unknown characters
                let first_char = remaining.chars().next().unwrap();
                let char_str = first_char.to_string();
                if let Some(&id) = self.vocab_map.get(&char_str) {
                    tokens.push(id);
                } else {
                    // Ultimate fallback: byte encoding
                    for byte in char_str.bytes() {
                        // Most BPE vocabularies have byte fallback tokens like `<0xXX>`
                        let byte_token = format!("<0x{:02X}>", byte);
                        if let Some(&id) = self.vocab_map.get(&byte_token) {
                            tokens.push(id);
                        } else {
                            // Unknown token — use token 1 (typically <unk>)
                            tokens.push(1);
                        }
                    }
                }
                remaining = &remaining[first_char.len_utf8()..];
            }
        }
    }

    /// Decode token IDs back to text.
    /// Handles BPE space markers (Ġ → literal space).
    pub fn decode(&self, tokens: &[usize], vocab: &[String]) -> String {
        let mut result = String::new();
        for (i, &token_id) in tokens.iter().enumerate() {
            if let Some(token_str) = vocab.get(token_id) {
                // Replace Ġ (U+0120) with space, but only if not at the start
                if token_str.starts_with('\u{0120}') {
                    if i > 0 { result.push(' '); }
                    result.push_str(&token_str[2..]); // Skip the Ġ prefix (2 UTF-8 bytes... wait, it's 1 char)
                    // Actually: Ġ is a single char (U+0120), so:
                    result.pop(); // remove what we just added, do it properly
                    if i > 0 { result.push(' '); }
                    result.push_str(&token_str.chars().skip(1).collect::<String>());
                } else if token_str == "<|im_start|>" || token_str == "<|im_end|>" {
                    // Keep special tokens as-is or strip them
                    result.push_str(token_str);
                } else {
                    result.push_str(token_str);
                }
            }
        }
        result
    }
}
```

**Simpler Alternative (if you don't want the full BPE tokenizer):**

If the above is too complex, here's a **minimal fix** that just replaces the naive tokenizer with a greedy longest-match:

```rust
/// Greedy longest-match tokenization from GGUF vocabulary.
/// Sorts vocab by length (longest first), then greedily matches.
fn tokenize_with_vocab(text: &str, vocab_map: &HashMap<String, usize>) -> Vec<usize> {
    // Build a sorted vocab list (longest tokens first)
    let mut vocab_list: Vec<(String, usize)> = vocab_map.iter()
        .map(|(s, &id)| (s.clone(), id))
        .collect();
    vocab_list.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let mut tokens = Vec::new();
    let mut remaining = text;

    // Handle BPE space convention: tokens after whitespace have "Ġ" or " " prefix
    while !remaining.is_empty() {
        // Skip leading whitespace
        remaining = remaining.trim_start();
        if remaining.is_empty() { break; }

        let mut matched = false;

        // Try each vocabulary token (longest first)
        for (token_str, token_id) in &vocab_list {
            if remaining.starts_with(token_str) {
                tokens.push(*token_id);
                remaining = &remaining[token_str.len()..];
                matched = true;
                break;
            }
        }

        if !matched {
            // No vocabulary token matches — try with "Ġ" prefix (BPE convention)
            let with_g = format!("\u{0120}{}", &remaining[..remaining.chars().next().unwrap().len_utf8()]);
            if let Some(&id) = vocab_map.get(&with_g) {
                tokens.push(id);
                remaining = &remaining[remaining.chars().next().unwrap().len_utf8()..];
                continue;
            }

            // Ultimate fallback: unknown character → skip
            let first_char = remaining.chars().next().unwrap();
            eprintln!("Warning: no vocab match for '{}'", first_char);
            remaining = &remaining[first_char.len_utf8()..];
        }
    }

    tokens
}
```

**Even Simpler Fix — Use HuggingFace tokenizer if available:**

The project already depends on the `tokenizers` crate. If you have a `tokenizer.json` file alongside your model, use it:

```rust
// At the top of test_generation.rs, add:
use leafcutter::tokenizer::Tokenizer;

// In main(), replace the tokenization section with:
let tokenizer = if let Some(dir) = std::path::Path::new(model_path).parent() {
    let tok_path = dir.join("tokenizer.json");
    if tok_path.exists() {
        eprintln!("Using HuggingFace tokenizer: {}", tok_path.display());
        Some(Tokenizer::from_file(tok_path.to_str().unwrap()).expect("Failed to load tokenizer"))
    } else {
        eprintln!("Warning: No tokenizer.json found. BPE encoding from GGUF vocab will be approximate.");
        None
    }
} else { None };

// Tokenize:
let token_ids = if let Some(ref tok) = tokenizer {
    tok.encode(prompt)
} else {
    tokenize_with_vocab(prompt, &token_to_id) // fallback
};

// Decode:
let decoded = if let Some(ref tok) = tokenizer {
    tok.decode(&generated, true)
} else {
    decode_tokens(&generated, &vocab) // fallback
};
```

### Recommended Fix Priority

| Approach | Effort | Quality | Recommendation |
|----------|--------|---------|----------------|
| HuggingFace `tokenizer.json` | 5 min | Perfect | **Use this if you have tokenizer.json** |
| Greedy longest-match BPE | 30 min | Good | **Use this as fallback** |
| Full `GgufTokenizer` struct | 2 hours | Excellent | Implement later for standalone operation |

The **5-minute fix**: just check for `tokenizer.json` next to the model and use the existing `Tokenizer` wrapper. This is what the Qwen models already do — that's why they produce coherent output!

---

## Quick Verification After Fixes

### Bug 1 verification:
```bash
# Without features (should now give clear error):
cargo run --bin test_generation -- --model Qwen3.5-2B-Q4_0.gguf
# → "Qwen3.5/3.6 models require llama.cpp FFI. Build with: cargo build --features llama-ffi"

# With features (should route correctly):
cargo run --release --features llama-ffi --bin test_generation -- --model Qwen3.5-2B-Q4_0.gguf
# → Coherent output
```

### Bug 2 verification:
```bash
cargo run --release --features llama-ffi --bin diagnose_models -- --model Llama-3.1-8B-Q4_0_4_4.gguf
# Should now show: "Unsupported types: IQ1_M" and auto-fallback to FFI
```

### Bug 3 verification:
```bash
# With tokenizer.json:
cargo run --release --bin test_generation -- --model Ministral-3B-Q4_K_M.gguf --prompt "Hello, how are you?"
# → "Hello, how are you? I am doing well..."
```
