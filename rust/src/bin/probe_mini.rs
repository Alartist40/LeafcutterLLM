use leafcutter::inference::engine::Engine;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/home/xander/Downloads/models/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf".to_string());
    let mut engine = Engine::load(&path).unwrap();
    let prompt = "The capital of France is";
    let tokens = engine.tokenize(prompt, false);
    eprintln!("Tokens: {:?}", tokens);
    let logits = engine.forward(&tokens);
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("\nTop 20 logits:");
    for i in 0..20 {
        let s = engine.decode(&[indexed[i].0]);
        eprintln!("  rank={:2} id={:<8} logit={:8.3} text={:?}", i+1, indexed[i].0, indexed[i].1, s);
    }
}
