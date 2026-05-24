//! Generation quality test — with GGUF-native tokenizer support

use clap::Parser;
use leafcutter::inference::engine::Engine;
use leafcutter::model::gguf::{GGUFile, GGUFValue};
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
        .unwrap_or(1);
    let eos = file.get_metadata_int("tokenizer.ggml.eos_token_id")
        .map(|v| v as usize)
        .unwrap_or(2);

    (vocab, bos, eos)
}

/// Simple word-level tokenizer (good enough for English test prompts).
fn simple_encode(text: &str, vocab_map: &HashMap<String, usize>, vocab: &[String], bos: usize) -> Vec<usize> {
    let mut tokens = vec![bos];
    for word in text.split_whitespace() {
        let key = word.to_lowercase();
        if let Some(&id) = vocab_map.get(&key) {
            tokens.push(id);
        } else if let Some(&id) = vocab_map.get(word) {
            tokens.push(id);
        } else {
            // fallback: space + word
            let spaced = format!(" {word}");
            if let Some(&id) = vocab_map.get(&spaced) {
                tokens.push(id);
            }
        }
    }
    tokens
}

fn simple_decode(tokens: &[usize], vocab: &[String]) -> String {
    tokens.iter()
        .map(|&t| vocab.get(t).cloned().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("")
}

fn main() {
    let args = Args::parse();

    println!("🌿 Leafcutter Generation Test");
    println!("   Model: {}", args.model);
    println!("   Prompt: '{}'", args.prompt);
    println!("   Max tokens: {}", args.tokens);
    println!("   Raw mode: {}", args.raw);

    // Extract vocab from GGUF before loading engine
    let (vocab, bos_id, _eos_id) = extract_vocab(&args.model);
    let vocab_map: HashMap<String, usize> = vocab.iter().enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();
    println!("📚 Vocab extracted: {} tokens", vocab.len());

    let mut engine = Engine::load(&args.model).expect("Failed to load engine");
    println!("✅ Engine loaded: {} layers, hidden_size={}", engine.config.num_hidden_layers, engine.config.hidden_size);

    let use_ffi_tok = engine.is_ffi();

    // Tokenize prompt
    let prompt_tokens = if use_ffi_tok {
        let prompt = if args.raw {
            args.prompt.clone()
        } else {
            format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", args.prompt)
        };
        engine.tokenize(&prompt, false)
    } else {
        // Use GGUF vocab for native models
        let text = if args.raw {
            args.prompt.clone()
        } else {
            format!("Hello! How can I help you with '{}'?", args.prompt)
        };
        simple_encode(&text, &vocab_map, &vocab, bos_id)
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
        simple_decode(&all_tokens, &vocab)
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
