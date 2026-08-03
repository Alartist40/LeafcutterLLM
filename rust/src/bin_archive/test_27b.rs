use leafcutter::inference::engine::Engine;
use leafcutter::tokenizer::{BaseTokenizer, Tokenizer};

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Qwen3.6-27B-IQ4_NL.gguf";
    println!("Loading 27B model...");
    let mut engine = Engine::load(path).expect("Failed to load engine");
    println!("✅ Loaded: {} layers, hidden={}, vocab={}",
        engine.config.num_hidden_layers,
        engine.config.hidden_size,
        engine.config.vocab_size);

    // Use HF tokenizer if available, else raw tokens
    let token_ids = if let Ok(tok) = Tokenizer::from_file("tests/tokenizer_qwen35.json") {
        let ids = tok.encode("The capital of France is");
        println!("Tokenizer IDs: {:?}", ids);
        ids
    } else {
        println!("No tokenizer found, using raw token [1]");
        vec![1usize]
    };

    println!("\n⏳ Running forward pass...");
    let start = std::time::Instant::now();
    let logits = engine.forward(&token_ids);
    let elapsed = start.elapsed();
    println!("Forward done in {:?}", elapsed);

    let top = logits.iter().enumerate()
        .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap())
        .map(|(i,v)| (i, *v))
        .unwrap();
    println!("Top token: id={} logit={:.4}", top.0, top.1);

    // Greedy decode 3 tokens
    println!("\n⏳ Greedy decoding 3 tokens...");
    let gen = engine.generate(&token_ids, 3, 0.0, 1.0);
    println!("Generated: {:?}", gen);
}
