use leafcutter::inference::engine::Engine;
use leafcutter::tokenizer::{BaseTokenizer, GgufTokenizer};

fn main() {
    let model_path = "../models/Qwen3.5-0.8B-Q4_0.gguf";
    let prompt = "2+2=";

    let mut engine = Engine::load(model_path).expect("Failed to load model");
    let tokenizer = leafcutter::tokenizer::Tokenizer::from_file("../models/tokenizer_qwen35.json")
        .expect("Failed to load tokenizer");
    let tokens = tokenizer.encode(prompt);
    println!("Tokens: {:?}", tokens);

    let logits = engine.forward(&tokens);

    let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 20 tokens:");
    for (rank, (tok, val)) in top.iter().take(20).enumerate() {
        println!("  {:2}. id={:<8} logit={:12.6}", rank + 1, tok, val);
    }

    let think_logit = logits.get(248068).copied().unwrap_or(f32::NAN);
    println!("\n<think> (248068) logit: {:.6}", think_logit);

    // Check embedding magnitudes
    let embed = engine.model.file.get_tensor_row_f32("token_embd.weight", 17).unwrap();
    let embed_mean: f32 = embed.iter().map(|&v| v.abs()).sum::<f32>() / embed.len() as f32;
    println!("Token 17 embed abs_mean: {:.6}", embed_mean);
}
