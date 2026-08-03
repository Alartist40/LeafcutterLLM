use leafcutter::inference::engine::Engine;
use leafcutter::model::gguf::{GGUFile, GGUFValue};
use std::collections::HashMap;

fn main() {
    let model_path = std::env::args().nth(1)
        .unwrap_or_else(|| "../models/Qwen3.5-0.8B-Q4_0.gguf".to_string());
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "2+2=".to_string());
    
    let mut engine = Engine::load(&model_path).expect("load engine");
    
    // Simple BPE tokenization without BOS
    let file = GGUFile::open(&model_path).expect("open");
    let vocab = match file.metadata.get("tokenizer.ggml.tokens") {
        Some(GGUFValue::Array(arr)) => arr.iter().map(|v| match v {
            GGUFValue::String(s) => s.clone(),
            _ => String::new(),
        }).collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    let mut vocab_sorted: Vec<(String, usize)> = vocab.iter().cloned().enumerate().map(|(i, s)| (s, i)).collect();
    vocab_sorted.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));
    let vocab_map: HashMap<String, usize> = vocab.iter().cloned().enumerate().map(|(i, s)| (s, i)).collect();
    
    let mut tokens = Vec::new();
    let words: Vec<&str> = prompt.split_whitespace().collect();
    for (wi, word) in words.iter().enumerate() {
        let with_g = format!("\u{0120}{}", word);
        let text: &str = if wi == 0 && vocab_map.contains_key(&with_g) { &with_g } else { word };
        let mut remaining = text;
        while !remaining.is_empty() {
            let mut matched = false;
            for (token_str, token_id) in &vocab_sorted {
                if remaining.starts_with(token_str) {
                    tokens.push(*token_id);
                    remaining = &remaining[token_str.len()..];
                    matched = true;
                    break;
                }
            }
            if !matched {
                if let Some(c) = remaining.chars().next() {
                    let s = c.to_string();
                    if let Some(&id) = vocab_map.get(&s) {
                        tokens.push(id);
                    }
                    remaining = &remaining[c.len_utf8()..];
                } else {
                    break;
                }
            }
        }
    }
    
    println!("Tokens: {:?}", tokens);
    
    // Call forward to trigger the dump
    let _ = engine.forward(&tokens);
    
    println!("Prefill complete. Check native_l0_input_post_norm.bin");
}
