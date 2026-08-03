//! Compare GGUF tokenizer vs Ollama's tokens for the Ornith prompt.
//! Reads Ollama context IDs from stdin and shows what our tokenizer
//! would produce for the same text.

use leafcutter::model::gguf::{GGUFile, GGUFValue};
use leafcutter::tokenizer::gguf::GgufTokenizer;
use std::io::Read;

fn main() {
    let gguf_path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&gguf_path).expect("open gguf");

    // Read prompt from stdin (read full)
    let mut prompt = String::new();
    std::io::stdin().read_to_string(&mut prompt).unwrap();
    let prompt = prompt.trim();
    eprintln!("[input prompt, {} chars]", prompt.len());
    eprintln!("[input prompt text]\n{}\n", prompt);

    // Build GGUF tokenizer
    let tok = GgufTokenizer::from_gguf(&gguf_path).expect("gguf tok");

    // Encode
    let ids = tok.encode(prompt, true);
    eprintln!("[our tokenizer produces {} tokens]", ids.len());
    eprintln!("[first 20 IDs] {:?}", &ids[..20.min(ids.len())]);
    eprintln!("[last 20 IDs] {:?}", &ids[ids.len().saturating_sub(20)..]);
}
