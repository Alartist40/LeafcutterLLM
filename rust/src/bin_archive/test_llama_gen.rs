//! End-to-end generation test for Llama models
//!
//! Uses the HuggingFace tokenizer for exact BPE, runs prefill + N decode steps.

use leafcutter::inference::engine::Engine;
use leafcutter::tokenizer::{BaseTokenizer, Tokenizer};

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf".to_string());
    let tokenizer_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "tests/tokenizer_llama.json".to_string());

    println!("🌿 Leafcutter Llama Generation Test");
    println!("   Model: {}", model_path);
    println!("   Tokenizer: {}", tokenizer_path);

    let tok = Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
    let mut engine = Engine::load(&model_path).expect("Failed to load engine");
    println!("✅ Engine: {} layers, hidden={}", engine.config.num_hidden_layers, engine.config.hidden_size);

    // Simple prompt (no chat template for basic coherence test)
    let prompt = "The capital of France is".to_string();
    let token_ids = tok.encode(&prompt);
    println!("📝 Prompt tokens: {} tokens", token_ids.len());
    for (i, &id) in token_ids.iter().enumerate().take(20) {
        println!("   [{}] id={} -> {:?}", i, id, tok.decode(&[id]));
    }

    // Prefill: single forward pass
    println!("\n⏳ Prefill...");
    let logits = engine.forward(&token_ids);
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("   Top-5 after prefill:");
    for (i, (tid, logit)) in indexed.iter().take(5).enumerate() {
        println!("     [{}] id={:>6} logit={:>7.2} -> {:?}", i, tid, logit, tok.decode(&[*tid]));
    }

    // Greedy decode: just a few tokens to verify coherence
    let max_tokens = 5;
    println!("\n⏳ Greedy decoding {} tokens...", max_tokens);
    let generated = engine.generate(&token_ids, max_tokens, 0.0, 1.0);

    println!("\n🔍 Generated tokens:");
    for (i, &tid) in generated.iter().enumerate() {
        println!("   [{}] id={:>6} -> {:?}", i, tid, tok.decode(&[tid]));
    }

    let all_text = tok.decode(&token_ids) + &tok.decode(&generated);
    println!("\n📝 Full output:\n{}", all_text);
}
