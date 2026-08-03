use leafcutter::model::gguf::{GGUFile, GGUFValue};

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_bos <gguf>");
    let file = GGUFile::open(&path).expect("open");
    
    let bos = file.get_metadata_int("tokenizer.ggml.bos_token_id");
    let eos = file.get_metadata_int("tokenizer.ggml.eos_token_id");
    let pad = file.get_metadata_int("tokenizer.ggml.padding_token_id");
    
    println!("BOS: {:?}", bos);
    println!("EOS: {:?}", eos);
    println!("PAD: {:?}", pad);
    
    // Check vocab
    if let Some(GGUFValue::Array(arr)) = file.metadata.get("tokenizer.ggml.tokens") {
        println!("Vocab size: {}", arr.len());
    }
    
    // Print some token ids
    if let Some(GGUFValue::Array(arr)) = file.metadata.get("tokenizer.ggml.tokens") {
        for (i, v) in arr.iter().take(5).enumerate() {
            if let GGUFValue::String(s) = v {
                println!("Token {}: '{}'", i, s);
            }
        }
        for i in [15, 16, 17, 28] {
            if let Some(GGUFValue::String(s)) = arr.get(i) {
                println!("Token {}: '{}'", i, s);
            }
        }
    }
}
