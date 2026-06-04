use leafcutter::model::gguf::{GGUFile, GGUFValue};

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = GGUFile::open(&path).unwrap();
    
    let tokens = match file.metadata.get("tokenizer.ggml.tokens") {
        Some(GGUFValue::Array(arr)) => {
            arr.iter().map(|v| match v {
                GGUFValue::String(s) => s.clone(),
                _ => String::new(),
            }).collect::<Vec<_>>()
        }
        _ => Vec::new(),
    };
    
    for arg in std::env::args().skip(2) {
        let id: usize = arg.parse().unwrap();
        if let Some(t) = tokens.get(id) {
            println!("Token {} = '{}'", id, t);
        } else {
            println!("Token {} = <out of range>", id);
        }
    }
}
