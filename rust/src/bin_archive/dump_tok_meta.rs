//! Dump tokenizer metadata from GGUF to understand the tokenization scheme.
//! Usage: cargo run --release --bin dump_tok_meta -- <model.gguf>

use leafcutter::model::gguf::{GGUFile, GGUFValue};

fn main() {
    let path = std::env::args().nth(1).expect("Usage: dump_tok_meta <model.gguf>");
    let file = GGUFile::open(&path).expect("Failed to open GGUF");

    // Print all tokenizer-related metadata keys
    for (key, val) in &file.metadata {
        if key.contains("tokenizer") || key.contains("model") && !key.contains("architecture") {
                match val {
                    GGUFValue::String(s) => {
                        if key.contains("chat_template") {
                            println!("{} = {} chars", key, s.len());
                            println!("   FULL: {}", s);
                        } else {
                            println!("{} = {:?}", key, &s[..s.len().min(100)]);
                        }
                    }
                GGUFValue::U32(v) => println!("{} = {}", key, v),
                GGUFValue::I32(v) => println!("{} = {}", key, v),
                GGUFValue::F32(v) => println!("{} = {}", key, v),
                GGUFValue::Bool(v) => println!("{} = {}", key, v),
                GGUFValue::Array(arr) => {
                    println!("{} = Array({} elements)", key, arr.len());
                    // Print first 5 elements
                    for (i, v) in arr.iter().take(5).enumerate() {
                        match v {
                            GGUFValue::String(s) => println!("  [{}] = {:?}", i, &s[..s.len().min(60)]),
                            _ => println!("  [{}] = (non-string)", i),
                        }
                    }
                }
                _ => println!("{} = (other type)", key),
            }
        }
    }
}
