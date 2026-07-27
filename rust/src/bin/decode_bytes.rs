use leafcutter::model::gguf::GGUFile;
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&path).expect("open");
    if let Some(leafcutter::model::gguf::GGUFValue::Array(arr)) = file.metadata.get("tokenizer.ggml.tokens") {
        for id in [248066, 248067, 248068, 248069] {
            if id < arr.len() {
                if let leafcutter::model::gguf::GGUFValue::String(s) = &arr[id] {
                    print!("  [{}] len={} bytes: ", id, s.len());
                    for b in s.as_bytes().iter() {
                        print!("\\x{:02x}", b);
                    }
                    println!();
                }
            }
        }
    }
}
