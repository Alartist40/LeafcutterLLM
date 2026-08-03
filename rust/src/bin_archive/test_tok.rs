use leafcutter::tokenizer::{Tokenizer, GgufBpeTokenizer, BaseTokenizer};

fn main() {
    let gguf_path = "/home/xander/Downloads/models/ornith-1.0-9b-Q8_0.gguf";
    let gguf_tok = GgufBpeTokenizer::from_gguf(gguf_path);
    match &gguf_tok {
        Some(t) => {
            println!("GGUF tokenizer OK: vocab={}", t.vocab_size());
            for s in ["Hello", "Hi", "The quick brown fox", "def main():"] {
                let ids = t.encode(s);
                println!("encode({:?}) -> {:?}", s, ids.iter().take(12).collect::<Vec<_>>());
            }
            let hf = Tokenizer::from_file("/home/xander/Downloads/models/tokenizer.json").unwrap();
            println!("HF tokenizer vocab={}", hf.vocab_size());
            for s in ["Hello", "Hi", "The quick brown fox", "def main():"] {
                let ids = hf.encode(s);
                println!("HF encode({:?}) -> {:?}", s, ids.iter().take(12).collect::<Vec<_>>());
            }
        }
        None => println!("GGUF tokenizer NONE"),
    }
}
