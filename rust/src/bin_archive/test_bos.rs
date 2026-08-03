use leafcutter::inference::engine::Engine;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let mut engine = Engine::load(path).expect("Failed to load");
    
    // Test with BOS
    let prompt = vec![128000usize, 9906usize];
    let logits = engine.forward(&prompt);
    
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("Leafcutter with BOS [128000, 9906] top 10:");
    for i in 0..10 {
        println!("  token={}: logit={:.6}", indexed[i].0, indexed[i].1);
    }
}
