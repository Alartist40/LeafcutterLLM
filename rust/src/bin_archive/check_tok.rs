use leafcutter::tokenizer::{BaseTokenizer, Tokenizer};

fn main() {
    let tok = Tokenizer::from_file("tests/tokenizer_llama.json").expect("Failed to load tokenizer");
    let tokens = tok.encode("Hello");
    println!("'Hello' -> {:?}", tokens);
    println!("Decoded: {:?}", tok.decode(&tokens));
}
