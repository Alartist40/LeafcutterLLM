//! logit_diff — prints the top-K logits (id + value) for a fixed prompt so
//! two runs (e.g. with and without LEAFCUTTER_Q8_GEMV) can be diffed to
//! confirm the Q8_K integer-dot path only perturbs logits slightly.
//!
//! Usage:
//!     cargo run --release --bin logit_diff -- <model-path> ["prompt"]
//!     LEAFCUTTER_Q8_GEMV=1 cargo run --release --bin logit_diff -- <model-path>

use leafcutter::inference::engine::Engine;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = match args.get(1) {
        Some(p) => p.clone(),
        None => {
            eprintln!("Usage: logit_diff <model-path> [prompt]");
            std::process::exit(2);
        }
    };
    let prompt = args.get(2).cloned().unwrap_or_else(|| "Hello world".to_string());

    let mut engine = match Engine::load(&model_path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    let tokens = engine.tokenize(&prompt, true);
    eprintln!("prompt {:?} -> {} tokens", prompt, tokens.len());

    let logits = engine.forward(&tokens);
    eprintln!("logits len: {}", logits.len());

    // Top-K by value.
    const K: usize = 10;
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    println!("top-{} logits:", K);
    for i in 0..K {
        println!("  id={:>6}  val={:+.6}", idx[i], logits[idx[i]]);
    }

    // Optional full dump for diffing full logit vectors.
    if args.iter().any(|a| a == "--all") {
        println!("full:");
        for (i, v) in logits.iter().enumerate() {
            println!("{} {:.6}", i, v);
        }
    }
}
