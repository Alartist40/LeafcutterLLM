use leafcutter::inference::engine::Engine;

fn rms(data: &[f32]) -> f32 {
    let mean_sq = data.iter().map(|v| v * v).sum::<f32>() / data.len() as f32;
    mean_sq.sqrt()
}

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let mut engine = Engine::load(path).unwrap();
    let tokens = vec![9906usize];
    
    for num_layers in [1, 2, 4, 8, 14, 28] {
        let saved = engine.config.num_hidden_layers;
        engine.config.num_hidden_layers = num_layers;
        engine.kv_cache.clear();
        engine.seq_offset = 0;
        
        let logits = engine.forward(&tokens);
        
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|(_,a),(_,b)| b.partial_cmp(a).unwrap());
        
        println!("\n=== {} layers ===", num_layers);
        for (i, (tid, logit)) in indexed.iter().take(5).enumerate() {
            println!("  [{}] id={:>6} logit={:>8.4}", i, tid, logit);
        }
        
        engine.config.num_hidden_layers = saved;
    }
}
