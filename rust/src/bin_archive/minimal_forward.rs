use leafcutter::inference::engine::Engine;

fn main() {
    let path = "/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf";
    let mut engine = Engine::load(path).unwrap();
    // Just test forward with a single token
    let tokens: Vec<usize> = vec![9707]; // A common token like "The" or similar
    let logits = engine.forward(&tokens);
    // Find top 5
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("Top 10 logits after 1-token forward:");
    for i in 0..10 {
        eprintln!("  id={} logit={:.6}", indexed[i].0, indexed[i].1);
    }
    eprintln!("Last 10 logits:");
    for i in (indexed.len()-10)..indexed.len() {
        eprintln!("  id={} logit={:.6}", indexed[i].0, indexed[i].1);
    }
}
