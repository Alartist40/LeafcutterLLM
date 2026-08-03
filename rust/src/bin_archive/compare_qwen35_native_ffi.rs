use leafcutter::inference::engine::Engine;
use leafcutter::llama_ffi::{backend_free, backend_init, LlamaContext, LlamaModel};
use leafcutter::tokenizer::BaseTokenizer;
use std::path::Path;

fn main() {
    let model_path = "../models/Qwen3.5-0.8B-Q4_0.gguf";
    let prompt = "2+2=";

    println!("=== Qwen3.5 Native vs FFI Comparison ===\n");

    // Native
    let mut engine = Engine::load(model_path).expect("Failed to load model natively");
    let tokenizer = leafcutter::tokenizer::Tokenizer::from_file("../models/tokenizer_qwen35.json")
        .expect("Failed to load tokenizer");
    let native_tokens = tokenizer.encode(prompt);
    println!("Native tokens: {:?}", native_tokens);
    let native_logits = engine.forward(&native_tokens);

    let mut native_top: Vec<(usize, f32)> = native_logits.iter().copied().enumerate().collect();
    native_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nNative top 10:");
    for (rank, (tok, val)) in native_top.iter().take(10).enumerate() {
        println!("  {:2}. id={:<8} logit={:12.6}", rank + 1, tok, val);
    }

    // FFI
    backend_init();
    let model = LlamaModel::load(Path::new(model_path), 0)
        .expect("Failed to load model via FFI");
    let mut ctx = LlamaContext::new(&model, 512, 4)
        .expect("Failed to create context");
    let ffi_tokens = ctx.tokenize(prompt, false, true);
    println!("\nFFI tokens: {:?}", ffi_tokens);
    let ffi_logits = ctx.forward(&ffi_tokens)
        .expect("Forward pass failed");

    let mut ffi_top: Vec<(usize, f32)> = ffi_logits.iter().copied().enumerate().collect();
    ffi_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nFFI top 10:");
    for (rank, (tok, val)) in ffi_top.iter().take(10).enumerate() {
        let piece = ctx.token_to_piece(*tok as i32);
        println!("  {:2}. id={:<8} logit={:12.6} piece='{}'", rank + 1, tok, val, piece.trim());
    }

    // Compare specific tokens
    println!("\n=== Token comparison ===");
    for &(name, tid) in &[
        ("<think>", 248068usize),
        ("4", 17usize),
        ("=", 28usize),
        ("+", 10usize),
        ("2", 17usize),
    ] {
        let native_val = native_logits.get(tid).copied().unwrap_or(f32::NAN);
        let ffi_val = ffi_logits.get(tid).copied().unwrap_or(f32::NAN);
        println!("{} (id={}): native={:.4}  ffi={:.4}  diff={:.4}", 
            name, tid, native_val, ffi_val, native_val - ffi_val);
    }

    // MSE
    let mse: f32 = native_logits.iter().zip(ffi_logits.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / native_logits.len() as f32;
    println!("\nMSE between native and FFI logits: {:.6}", mse);

    backend_free();
}
