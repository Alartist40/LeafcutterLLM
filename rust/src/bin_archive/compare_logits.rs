use leafcutter::inference::engine::Engine;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let mut engine = Engine::load(path).unwrap();
    
    for tokens in [vec![9906usize], vec![128000, 9906]] {
        engine.kv_cache.clear();
        engine.seq_offset = 0;
        let logits = engine.forward(&tokens);
        
        println!("\n=== tokens={:?} ===", tokens);
        for tid in [791, 9906, 110645, 16751] {
            println!("  token {} logit = {:.4}", tid, logits[tid]);
        }
    }
}
