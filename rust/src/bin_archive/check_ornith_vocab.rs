use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("usage: check_ornith_vocab <gguf>");
    let file = GGUFile::open(&path).expect("open GGUF");
    
    // The Ornith GGUF dump earlier showed these IDs in metadata:
    // tokenizer.ggml.padding_token_id = 248044
    // tokenizer.ggml.eos_token_id = 248046
    
    // From the dump_logits output, the tokenizer assigns:
    // 248045 as BOS, 9418 for "Hello", 198 = "\n", etc.
    
    // Find which tokens at those IDs are special tokens
    if let Some(toks) = file.metadata.get("tokenizer.ggml.tokens") {
        if let leafcutter::model::gguf::GGUFValue::Array(arr) = toks {
            for id in [248045, 248046, 248044, 248047] {
                if let Some(leafcutter::model::gguf::GGUFValue::String(s)) = arr.get(id) {
                    println!("token[{}] = {:?}", id, s);
                } else {
                    println!("token[{}] = <not string>", id);
                }
            }
        }
    }
}
