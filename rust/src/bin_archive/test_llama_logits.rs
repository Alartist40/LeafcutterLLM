use leafcutter::inference::engine::Engine;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let mut engine = Engine::load(path).unwrap();
    let tokens = vec![9906usize]; // "Hello"
    
    let logits = engine.forward(&tokens);
    
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_,a),(_,b)| b.partial_cmp(a).unwrap());
    
    println!("Top-10 logits for token 'Hello' (9906):");
    for (i, (tid, logit)) in indexed.iter().take(10).enumerate() {
        println!("  [{}] id={:>6} logit={:>8.4}", i, tid, logit);
    }
    
    // Also print hidden state stats
    println!("\nLogit stats: min={:.4} max={:.4} mean={:.4} abs_mean={:.4}",
        logits.iter().cloned().fold(f32::INFINITY, f32::min),
        logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        logits.iter().sum::<f32>() / logits.len() as f32,
        logits.iter().map(|x| x.abs()).sum::<f32>() / logits.len() as f32,
    );
}
