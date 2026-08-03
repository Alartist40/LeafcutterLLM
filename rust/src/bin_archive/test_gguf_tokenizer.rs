use leafcutter::tokenizer::BaseTokenizer;
use leafcutter::inference::engine::Engine;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let engine = Engine::load(path).expect("load model");

    let tok = engine.tokenizer_from_model().expect("model has vocab");
    println!("Vocab size: {}", tok.vocab_size());
    println!("BOS: {:?}, EOS: {:?}", tok.bos_id(), tok.eos_id());

    let text = "The capital of France is";
    let ids = tok.encode(text, true);
    println!("Text: {:?}", text);
    println!("Token IDs: {:?}", ids);
    for (i, &id) in ids.iter().enumerate() {
        println!("  [{}] id={} -> {:?}", i, id, tok.decode(&[id]));
    }

    // Compare with HF tokenizer if available
    if let Ok(hf) = leafcutter::tokenizer::Tokenizer::from_file("tests/tokenizer_llama.json") {
        let hf_ids = hf.encode(text);
        println!("\nHF tokenizer IDs: {:?}", hf_ids);
        println!("Match: {}", ids == hf_ids);
    }
}
