use leafcutter::inference::engine::Engine;
use leafcutter::tokenizer::{BaseTokenizer, Tokenizer};

fn main() {
    let tok = Tokenizer::from_file("tests/tokenizer_llama.json").unwrap();
    let mut engine = Engine::load("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
    let ids = tok.encode("The capital of France is");
    println!("Prompt: {:?}", tok.decode(&ids));
    for temp in [0.0_f32, 0.3, 0.7, 1.0] {
        // Fresh engine for each temp to avoid cache contamination
        let mut eng = Engine::load("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
        let gen = eng.generate(&ids, 6, temp, 0.95);
        let text = tok.decode(&gen);
        println!("temp={:.1}: {}", temp, text.replace('\n', "\\n"));
    }
}
