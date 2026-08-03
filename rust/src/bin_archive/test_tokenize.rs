use leafcutter::tokenizer::{BaseTokenizer, Tokenizer};

fn main() {
    let tok = Tokenizer::from_file("tests/tokenizer_llama.json").expect("load tokenizer");
    let text = "The capital of France is";
    let ids = tok.encode(text);
    println!("Text: {:?}", text);
    println!("Token IDs: {:?}", ids);
    for (i, &id) in ids.iter().enumerate() {
        let decoded = tok.decode(&[id]);
        println!("  [{}] id={} -> {:?}", i, id, decoded);
    }
}
