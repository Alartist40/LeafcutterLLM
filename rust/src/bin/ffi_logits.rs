#[cfg(feature = "llama-ffi")]
use leafcutter::llama_ffi::{LlamaModel, LlamaContext};

fn main() {
    #[cfg(feature = "llama-ffi")]
    {
        let model_path = std::env::args().nth(1).unwrap_or_else(|| "../models/Qwen3.5-2B-BF16.gguf".to_string());
        let prompt = std::env::args().nth(2).unwrap_or_else(|| "2+2=".to_string());
        
        let model = LlamaModel::load(std::path::Path::new(&model_path), 0)
            .expect("Failed to load model");
        let mut ctx = LlamaContext::new(&model, 4096, 4)
            .expect("Failed to create context");
        
        let tokens = ctx.tokenize(&prompt, false, true);
        println!("Tokens: {:?}", tokens);
        
        let logits = ctx.forward(&tokens).expect("Forward failed");
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("FFI Top 10:");
        for (i, (tid, val)) in indexed.iter().take(10).enumerate() {
            println!("  [{}] id={:<8} logit={:12.6}", i+1, tid, val);
        }
    }
    #[cfg(not(feature = "llama-ffi"))]
    {
        println!("FFI not available");
    }
}
