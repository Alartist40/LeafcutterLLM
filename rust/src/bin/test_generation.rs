//! Generation quality test — with GGUF-native tokenizer support

use clap::Parser;
use leafcutter::inference::engine::Engine;
use leafcutter::model::gguf::{GGUFile, GGUFValue};
use leafcutter::tokenizer::chat_template::apply_chat_template_from_gguf;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "test_generation")]
struct Args {
    #[arg(short, long, default_value = "/home/xander/Documents/portfolio/AI Models/Qwen3.5-9B-IQ4_NL.gguf")]
    model: String,

    #[arg(short, long, default_value = "Hello")]
    prompt: String,

    #[arg(short, long, default_value_t = 10)]
    tokens: usize,

    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    #[arg(long, default_value_t = false)]
    raw: bool,
}

/// Extract vocabulary from GGUF metadata.
fn extract_vocab(path: &str) -> (Vec<String>, usize, usize) {
    let file = GGUFile::open(path).expect("Failed to open GGUF for vocab extraction");

    let vocab = match file.metadata.get("tokenizer.ggml.tokens") {
        Some(GGUFValue::Array(arr)) => {
            arr.iter()
                .map(|v| match v {
                    GGUFValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect()
        }
        _ => Vec::new(),
    };

    let bos = file.get_metadata_int("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize)
        .unwrap_or(usize::MAX); // usize::MAX means no BOS token
    let eos = file.get_metadata_int("tokenizer.ggml.eos_token_id")
        .map(|v| v as usize)
        .unwrap_or(2);

    (vocab, bos, eos)
}

/// Greedy longest-match BPE tokenizer from GGUF vocabulary.
///
/// Sorts vocab by length (longest first), then greedily matches tokens.
/// Handles BPE space convention: words after whitespace use "Ġ" prefix.
struct GgufBpeTokenizer {
    vocab_sorted: Vec<(String, usize)>, // (token_string, token_id), longest first
    vocab_map: HashMap<String, usize>,
    bos_token: usize, // usize::MAX means no BOS token
    eos_token: usize,
}

impl GgufBpeTokenizer {
    fn new(vocab: Vec<String>, bos: usize, eos: usize) -> Self {
        let vocab_map: HashMap<String, usize> = vocab.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        let mut vocab_sorted: Vec<(String, usize)> = vocab.into_iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        vocab_sorted.sort_by(|a, b| {
            b.0.len().cmp(&a.0.len())
                .then_with(|| a.1.cmp(&b.1))
        });

        Self { vocab_sorted, vocab_map, bos_token: bos, eos_token: eos }
    }

    /// Encode text to token IDs using greedy longest-match.
    ///
    /// BPE convention: prepend Ġ (U+0120) to words that follow whitespace,
    /// then greedily match subword pieces within each word.
    fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        // Only add BOS if it's defined (some models like Qwen3.5 have no BOS)
        if self.bos_token != usize::MAX {
            tokens.push(self.bos_token);
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return tokens;
        }

        // First word: try without Ġ prefix first
        let first = words[0];
        let first_with_g = format!("\u{0120}{}", first);
        if self.vocab_map.contains_key(&first_with_g) {
            self.greedy_encode(&first_with_g, &mut tokens);
        } else {
            self.greedy_encode(first, &mut tokens);
        }

        // Subsequent words: prepend Ġ, then greedy match subpieces
        for word in &words[1..] {
            let with_g = format!("\u{0120}{}", word);
            self.greedy_encode(&with_g, &mut tokens);
        }

        tokens
    }

    /// Greedily encode a string by matching the longest vocab token at each position.
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
    /// Handles BPE space markers (Ġ → literal space).
    fn decode(&self, tokens: &[usize], vocab: &[String]) -> String {
        let mut result = String::new();
        let space_marker = '\u{0120}';

        for &token_id in tokens {
            // Skip special tokens
            if (self.bos_token != usize::MAX && token_id == self.bos_token) || token_id == self.eos_token {
                continue;
            }

            if let Some(token_str) = vocab.get(token_id) {
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
                    // Skip other special tokens like <|begin_of_text|>, <|end_of_text|>, etc.
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

fn main() {
    // Cap rayon thread pool early. Accepts:
    //   - RAYON_NUM_THREADS env var (rayon's standard hook)
    //   - LEAFCUTTER_THREADS env var (programmatic override)
    //   - default = available_parallelism() - 1
    // Idempotent — second call inside rayon returns Err which we ignore.
    if let Ok(s) = std::env::var("LEAFCUTTER_THREADS") {
        if let Ok(n) = s.parse::<usize>() {
            let _ = leafcutter::init::configure_thread_pool(Some(n));
        }
    } else {
        let _ = leafcutter::init::configure_thread_pool(None);
    }

    let args = Args::parse();

    println!("🌿 Leafcutter Generation Test");
    println!("   Model: {}", args.model);
    println!("   Prompt: '{}'", args.prompt);
    println!("   Max tokens: {}", args.tokens);
    println!("   Raw mode: {}", args.raw);

    // Extract vocab from GGUF before loading engine
    let (vocab, bos_id, _eos_id) = extract_vocab(&args.model);
    let tokenizer = GgufBpeTokenizer::new(vocab.clone(), bos_id, _eos_id);
    println!("📚 Vocab extracted: {} tokens", vocab.len());

    let mut engine = Engine::load(&args.model).expect("Failed to load engine");
    println!("✅ Engine loaded: {} layers, hidden_size={}", engine.config.num_hidden_layers, engine.config.hidden_size);

    let use_ffi_tok = engine.is_ffi();

    // Load GGUF metadata for chat template detection
    let gguf_file = GGUFile::open(&args.model).expect("Failed to open GGUF");
    let has_chat_template = gguf_file.metadata.contains_key("tokenizer.chat_template");

    // Format prompt using chat template auto-detection
    let formatted_prompt = if args.raw {
        args.prompt.clone()
    } else if has_chat_template {
        let templated = apply_chat_template_from_gguf(&gguf_file.metadata, "", &args.prompt);
        println!("🎭 Chat template applied (detected: {})", 
            if templated.starts_with("[SYSTEM_PROMPT]") { "Ministral" }
            else if templated.contains("<|start_header_id|>") { "Llama-3" }
            else if templated.contains("[INST]") { "Mistral" }
            else if templated.contains("<|im_start|>") { "ChatML" }
            else if templated.contains("<start_of_turn>") { "Gemma" }
            else { "Unknown" }
        );
        templated
    } else {
        // Fallback for models without chat template
        format!("{}", args.prompt)
    };

    // Tokenize prompt
    let prompt_tokens = if use_ffi_tok {
        engine.tokenize(&formatted_prompt, false)
    } else {
        tokenizer.encode(&formatted_prompt)
    };
    println!("📝 Prompt tokens: {}", prompt_tokens.len());

    // Prefill: check top token from prompt
    println!("\n⏳ Prefill forward pass...");
    let logits = engine.forward(&prompt_tokens);
    let top_prefill = logits.iter().enumerate()
        .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap())
        .map(|(i,v)| (i, *v))
        .unwrap();
    let top_piece = if use_ffi_tok {
        engine.decode(&[top_prefill.0])
    } else {
        vocab.get(top_prefill.0).cloned().unwrap_or_default()
    };
    println!("   Top prefill token: {} (logit={:.2}) -> '{}'", top_prefill.0, top_prefill.1, top_piece);

    // Check top-5 prefill tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_,a),(_,b)| b.partial_cmp(a).unwrap());
    println!("   Top-5 prefill tokens:");
    for (i, (tid, logit)) in indexed.iter().take(5).enumerate() {
        let piece = if use_ffi_tok {
            engine.decode(&[*tid])
        } else {
            vocab.get(*tid).cloned().unwrap_or_default()
        };
        println!("     [{}] id={:>6} logit={:>7.2} -> '{}'", i, tid, logit, piece);
    }

    // Now run generate
    println!("\n⏳ Generating {} tokens...", args.tokens);
    let start = Instant::now();
    let generated = engine.generate(&prompt_tokens, args.tokens, args.temperature, args.top_p);
    let elapsed = start.elapsed();

    let tok_per_sec = generated.len() as f64 / elapsed.as_secs_f64();
    println!("\n✅ Generated {} tokens in {:?} ({:.2} tok/sec)", generated.len(), elapsed, tok_per_sec);

    // Decode each generated token individually
    println!("\n🔍 Token-by-token breakdown:");
    for (i, &tid) in generated.iter().enumerate() {
        let text = if use_ffi_tok {
            engine.decode(&[tid])
        } else {
            vocab.get(tid).cloned().unwrap_or_default()
        };
        println!("   [{}] id={:>6} -> '{}'", i, tid, text);
    }

    let all_tokens: Vec<usize> = prompt_tokens.iter().chain(generated.iter()).copied().collect();
    let decoded = if use_ffi_tok {
        engine.decode(&all_tokens)
    } else {
        tokenizer.decode(&all_tokens, &vocab)
    };
    println!("\n📝 Full decoded output:\n{}", decoded);

    // Basic coherence check: flag repetitive tokens
    let mut repeat_count = 1;
    let mut max_repeat = 1;
    for w in generated.windows(2) {
        if w[0] == w[1] {
            repeat_count += 1;
            max_repeat = max_repeat.max(repeat_count);
        } else {
            repeat_count = 1;
        }
    }
    if max_repeat > 4 {
        println!("\n⚠️  Warning: detected {} consecutive identical tokens (possible degeneration)", max_repeat);
    }
}
