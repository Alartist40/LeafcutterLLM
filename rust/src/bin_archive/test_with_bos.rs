use leafcutter::inference::engine::Engine;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let mut engine = Engine::load(path).unwrap();
    
    // BOS = 128000, Hello = 9906
    for tokens in [vec![9906usize], vec![128000usize, 9906]] {
        engine.kv_cache.clear();
        engine.seq_offset = 0;
        let logits = engine.forward(&tokens);
        
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|(_,a),(_,b)| b.partial_cmp(a).unwrap());
        
        println!("\n=== tokens={:?} ===", tokens);
        for (i, (tid, logit)) in indexed.iter().take(5).enumerate() {
            println!("  [{}] id={:>6} logit={:>8.4}", i, tid, logit);
        }
    }
}
