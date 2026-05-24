//! Generation quality test

use clap::Parser;
use leafcutter::inference::engine::Engine;
use leafcutter::tokenizer::Tokenizer;
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

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long, default_value_t = false)]
    raw: bool,
}

fn main() {
    let args = Args::parse();
    let tok_path = args.tokenizer.as_deref().unwrap_or("tests/tokenizer.json");

    println!("🌿 Leafcutter Generation Test");
    println!("   Model: {}", args.model);
    println!("   Prompt: '{}'", args.prompt);
    println!("   Max tokens: {}", args.tokens);
    println!("   Raw mode: {}", args.raw);

    let mut engine = Engine::load(&args.model).expect("Failed to load engine");
    println!("✅ Engine loaded: {} layers, hidden_size={}", engine.config.num_hidden_layers, engine.config.hidden_size);

    // For FFI backend (Qwen3.5/3.6), use the model's built-in tokenizer.
    // For native backend, use the external tokenizer.json.
    let use_ffi_tok = engine.is_ffi();
    let prompt_tokens = if use_ffi_tok {
        let prompt = if args.raw {
            args.prompt.clone()
        } else {
            format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", args.prompt)
        };
        engine.tokenize(&prompt, false)
    } else {
        let tok = if std::path::Path::new(tok_path).exists() {
            Tokenizer::from_file(tok_path).expect("Failed to load tokenizer")
        } else {
            eprintln!("❌ Tokenizer not found at {}", tok_path);
            std::process::exit(1);
        };
        if args.raw {
            tok.encode(&args.prompt)
        } else {
            let prompt = tok.apply_chat_template(&args.prompt);
            tok.encode(&prompt)
        }
    };
    println!("📝 Prompt tokens: {}", prompt_tokens.len());

    // Prefill: check top token from prompt
    println!("\n⏳ Prefill forward pass...");
    let logits = engine.forward(&prompt_tokens);
    let top_prefill = logits.iter().enumerate()
        .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap())
        .map(|(i,v)| (i, *v))
        .unwrap();
    let top_piece = if use_ffi_tok { engine.decode(&[top_prefill.0]) } else { String::new() };
    println!("   Top prefill token: {} (logit={:.2}) -> '{}'", top_prefill.0, top_prefill.1, top_piece);

    // Check top-5 prefill tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_,a),(_,b)| b.partial_cmp(a).unwrap());
    println!("   Top-5 prefill tokens:");
    for (i, (tid, logit)) in indexed.iter().take(5).enumerate() {
        let piece = if use_ffi_tok { engine.decode(&[*tid]) } else { String::new() };
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
        let text = if use_ffi_tok { engine.decode(&[tid]) } else { String::new() };
        println!("   [{}] id={:>6} -> '{}'", i, tid, text);
    }

    let all_tokens: Vec<usize> = prompt_tokens.iter().chain(generated.iter()).copied().collect();
    let decoded = if use_ffi_tok { engine.decode(&all_tokens) } else { String::new() };
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
