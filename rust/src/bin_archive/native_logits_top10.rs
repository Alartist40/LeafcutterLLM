use leafcutter::inference::engine::Engine;

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| "../models/Qwen3.5-2B-BF16.gguf".to_string());
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "2+2=".to_string());
    
    let mut engine = Engine::load(&model_path).expect("Failed to load model");
    
    // Simple tokenization
    let file = leafcutter::model::gguf::GGUFile::open(&model_path).expect("open");
    let vocab = match file.metadata.get("tokenizer.ggml.tokens") {
        Some(leafcutter::model::gguf::GGUFValue::Array(arr)) => arr.iter().map(|v| match v {
            leafcutter::model::gguf::GGUFValue::String(s) => s.clone(),
            _ => String::new(),
        }).collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    let vocab_map: std::collections::HashMap<String, usize> = vocab.iter().cloned().enumerate().map(|(i, s)| (s, i)).collect();
    
    let mut tokens = Vec::new();
    for word in prompt.split_whitespace() {
        let with_g = format!("\u{0120}{}", word);
        let text = if tokens.is_empty() && vocab_map.contains_key(&with_g) { &with_g } else { word };
        let mut remaining = text;
        while !remaining.is_empty() {
            let mut matched = false;
            for len in (1..=remaining.len()).rev() {
                let prefix = &remaining[..len];
                if let Some(&id) = vocab_map.get(prefix) {
                    tokens.push(id);
                    remaining = &remaining[len..];
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
                } else { break; }
            }
        }
    }
    
    println!("Tokens: {:?}", tokens);
    let logits = engine.forward(&tokens);
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("Native Top 10:");
    for (i, (tid, val)) in indexed.iter().take(10).enumerate() {
        println!("  [{}] id={:<8} logit={:12.6}", i+1, tid, val);
    }
}
