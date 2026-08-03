//! Test: load the Ornith tokenizer and encode/decode a sample.
use leafcutter::bpe_tokenizer::BpeTokenizer;

fn main() {
    let path = "/home/xander/Downloads/models/ornith safetensor/tokenizer.json";
    println!("Loading tokenizer from {path}");

    let tok = match BpeTokenizer::load(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    println!("\nSpecial tokens:");
    println!("  <|im_start|> id = {}", tok.id_of("<|im_start|>"));
    println!("  <|im_end|> id = {}", tok.id_of("<|im_end|>"));

    let test = "The capital of France is";
    let ids = tok.encode(test, 1024);
    println!("\nEncoded \"{test}\": {} tokens", ids.len());
    println!("  ids: {:?}", &ids[..ids.len().min(20)]);

    let decoded = tok.decode(&ids);
    println!("\nDecoded back: \"{decoded}\"");
    println!("Match: {}", decoded == test);
}
