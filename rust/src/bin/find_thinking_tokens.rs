use leafcutter::model::gguf::GGUFile;
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&path).expect("open");
    let mut count = 0;
    for (key, val) in &file.metadata {
        if key.starts_with("tokenizer.ggml.tokens") {
            count += 1;
            // Print first 10 tokens that contain "think" or emoji
        }
    }
    eprintln!("Total tokenizer.ggml.tokens entries: {}", count);
    // Find token IDs for thinking-related tokens by scanning metadata arrays
    if let Some(leafcutter::model::gguf::GGUFValue::Array(arr)) = file.metadata.get("tokenizer.ggml.tokens") {
        for (i, t) in arr.iter().enumerate() {
            if let leafcutter::model::gguf::GGUFValue::String(s) = t {
                if s.contains("think") || s.contains("💭") || s.contains("<think") {
                    println!("  [{}] {}", i, s);
                }
            }
        }
    }
}
