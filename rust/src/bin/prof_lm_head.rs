//! Benchmark: profile a few forward calls with LEAFCUTTER_PROFILE to see
//! the lm_head timing breakdown.
use leafcutter::inference::engine::Engine;
use std::time::Instant;

fn main() {
    let gguf = "/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf";

    eprintln!("Loading model...");
    let t0 = Instant::now();
    let mut engine = Engine::load(gguf).expect("Engine::load");
    eprintln!("Loaded in {:.2}s", t0.elapsed().as_secs_f64());

    let prompt = "Hello";
    let ids = engine.tokenize(prompt, false);
    eprintln!("Tokenized '{}': {:?}", prompt, ids);

    // Warm up
    eprintln!("Warm up...");
    let _ = engine.forward_native(&ids).expect("warmup forward");

    // Profile once
    eprintln!("\n=== Profile (LEAFCUTTER_PROFILE=1) ===");
    std::env::set_var("LEAFCUTTER_PROFILE", "1");
    let t1 = Instant::now();
    let logits = engine.forward_native(&ids).expect("forward_native");
    let elapsed = t1.elapsed();
    eprintln!("Total: {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    eprintln!("Logits: {}", logits.len());

    // Print top-5
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("\nTop-5:");
    for (i, (id, logit)) in indexed.iter().take(5).enumerate() {
        eprintln!("  {}. id={} logit={:.3}", i + 1, id, logit);
    }
}