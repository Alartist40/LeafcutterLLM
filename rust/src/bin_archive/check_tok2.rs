use leafcutter::tokenizer::{BaseTokenizer, Tokenizer};

fn main() {
    let tok = Tokenizer::from_file("tests/tokenizer_llama.json").expect("Failed to load tokenizer");
    let tokens = tok.encode(" from");
    println!("' from' -> {:?}", tokens);
    println!("Decoded: {:?}", tok.decode(&tokens));
    
    let tokens2 = tok.encode("!");
    println!("'!' -> {:?}", tokens2);
    
    let tokens3 = tok.encode(",");
    println!("',' -> {:?}", tokens3);
    
    // Check what token 110645 is
    let text = tok.decode(&[110645]);
    println!("110645 -> {:?}", text);
}
