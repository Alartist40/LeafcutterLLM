use leafcutter::model::gguf::GGUFile;
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&path).expect("open");
    if let Some(leafcutter::model::gguf::GGUFValue::Array(arr)) = file.metadata.get("tokenizer.ggml.tokens") {
        for (i, t) in arr.iter().enumerate() {
            if i >= 248060 && i < 248080 {
                if let leafcutter::model::gguf::GGUFValue::String(s) = t {
                    println!("  [{}] {:?}", i, s);
                }
            }
        }
    }
    // Print all token types around there too
    if let Some(leafcutter::model::gguf::GGUFValue::Array(arr)) = file.metadata.get("tokenizer.ggml.token_type") {
        for (i, t) in arr.iter().enumerate() {
            if i >= 248060 && i < 248080 {
                println!("  type[{}] = {:?}", i, t);
            }
        }
    }
}
