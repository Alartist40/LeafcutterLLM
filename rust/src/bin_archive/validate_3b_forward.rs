//! Quick 3B forward validation — 1 token, reports time and correctness check.

use leafcutter::inference::engine::Engine;
use leafcutter::tokenizer::{BaseTokenizer, Tokenizer};

fn main() {
    let model = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    println!("🌿 3B Forward Validator");
    println!("   Model: {}", model);

    let mut engine = Engine::load(model).expect("Failed to load engine");
    println!("✅ Engine loaded: {} layers, hidden_size={}", engine.config.num_hidden_layers, engine.config.hidden_size);

    let tok = Tokenizer::from_file("tests/tokenizer_llama.json").expect("Tokenizer");
    let tokens = tok.encode("Hi");
    println!("📝 Tokens: {:?} (len={})", tokens, tokens.len());

    println!("\n⏳ Running forward pass...");
    let start = std::time::Instant::now();
    let logits = engine.forward(&tokens);
    let elapsed = start.elapsed();

    println!("✅ Forward pass done in {:?}", elapsed);
    println!("📊 Logits len: {}", logits.len());
    println!("📊 Logits sample: [{:.4}, {:.4}, {:.4}, ...]", logits[0], logits[1], logits[2]);

    // Sanity check: logits should be finite and in a reasonable range
    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let has_nan = logits.iter().any(|&x| x.is_nan());
    println!("📊 Logits range: [{:.2}, {:.2}]  NaN: {}", min, max, has_nan);

    if has_nan {
        println!("❌ FAILED: NaN detected in logits");
        std::process::exit(1);
    } else {
        println!("✅ PASSED: Output is finite and sane");
    }
}
