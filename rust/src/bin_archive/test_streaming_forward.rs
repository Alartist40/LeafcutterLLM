//! Test: streaming native forward pass — prompt through all layers.
//! Uses forward_sequence to process all tokens layer-by-layer (loads
//! weights ONCE per layer instead of once per token per layer — ~5x faster).
use leafcutter::streaming_ornith::StreamingOrnith;
use std::path::Path;

fn main() {
    let dir = Path::new("/home/xander/Downloads/models/ornith safetensor");
    eprintln!("Loading Ornith from {}", dir.display());

    let mut model = StreamingOrnith::open(dir).expect("open model");
    eprintln!(
        "Loaded: hidden={} vocab={} layers={}",
        model.cfg.hidden_size, model.cfg.vocab_size, model.cfg.num_hidden_layers
    );

    let prompt = "The capital of France is";
    let prompt_ids = model.tok.encode(prompt, 1024);
    eprintln!("\nPrompt: \"{prompt}\"");
    eprintln!("Tokens: {prompt_ids:?}");

    let t0 = std::time::Instant::now();
    let logits = model.forward_sequence(&prompt_ids).expect("forward");
    let elapsed = t0.elapsed();
    eprintln!("\nForward took {:.2}s", elapsed.as_secs_f64());
    eprintln!("Logits: {} values", logits.len());

    // Top-5
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("\nTop-5 predictions:");
    for (i, &(id, logit)) in indexed.iter().take(5).enumerate() {
        let token_str = model.tok.decode(&[id as i32]);
        eprintln!("  {}. id={} logit={:.3} text=\"{token_str}\"", i + 1, id, logit);
    }

    // Check specific tokens
    for target_id in [11751, 220, 198, 13] {
        let logit = logits[target_id];
        let token_str = model.tok.decode(&[target_id as i32]);
        eprintln!("  token {target_id} ({token_str:?}) logit={logit:.3}");
    }

    // Did we get " Paris"?
    let paris_rank = indexed.iter().position(|&(id, _)| id == 11751).unwrap_or(999);
    eprintln!("\n\" Paris\" (id=11751) rank={} logit={:.3}", paris_rank, logits[11751]);
    
    // Final hidden state mean_abs
    eprintln!("Final logit top: id={} logit={:.3}", indexed[0].0, indexed[0].1);
}
