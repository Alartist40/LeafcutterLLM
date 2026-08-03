use leafcutter::model::gguf::GGUFile;
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&path).expect("open");
    let ids: Vec<usize> = std::env::args().nth(2).unwrap()
        .split(',').map(|s| s.parse().unwrap()).collect();
    if let Some(leafcutter::model::gguf::GGUFValue::Array(arr)) = file.metadata.get("tokenizer.ggml.tokens") {
        for id in ids {
            if id < arr.len() {
                if let leafcutter::model::gguf::GGUFValue::String(s) = &arr[id] {
                    println!("  [{}] {:?} (bytes={})", id, s, s.as_bytes().len());
                }
            }
        }
    }
    // Also check special tokens (added_tokens)
    println!("---added_tokens---");
    if let Some(leafcutter::model::gguf::GGUFValue::Array(arr)) = file.metadata.get("tokenizer.ggml.added_tokens") {
        for (i, t) in arr.iter().enumerate() {
            if let leafcutter::model::gguf::GGUFValue::String(s) = t {
                if s.contains("think") || s.contains("💭") || s.contains("<") {
                    println!("  added[{}] {:?} (bytes={})", i, s, s.as_bytes().len());
                }
            }
        }
    }
}
