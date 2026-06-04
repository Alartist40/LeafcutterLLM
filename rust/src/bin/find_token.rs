use leafcutter::model::gguf::{GGUFile, GGUFValue};

fn main() {
    let path = std::env::args().nth(1).expect("Usage: find_token <gguf_path> <search_str>");
    let search = std::env::args().nth(2).unwrap_or_else(|| "<think>".to_string());
    let file = GGUFile::open(&path).expect("Failed to open GGUF");
    
    let vocab: Vec<String> = match file.metadata.get("tokenizer.ggml.tokens") {
        Some(GGUFValue::Array(arr)) => {
            arr.iter().map(|v| match v {
                GGUFValue::String(s) => s.clone(),
                _ => String::new(),
            }).collect()
        }
        _ => {
            println!("No vocab found");
            return;
        }
    };
    
    println!("Vocab size: {}", vocab.len());
    for (i, token) in vocab.iter().enumerate() {
        if token.contains(&search) {
            println!("Token {}: {:?}", i, token);
        }
    }
}
