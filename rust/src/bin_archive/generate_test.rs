//! generate_test — load a model, generate N tokens from a prompt, and report
//! per-token timing + RSS. The definitive coherence / memory validation tool.
//!
//! Usage:
//!     cargo run --release --bin generate_test -- <model-path> ["prompt"] [max-tokens]
//! Env:
//!     LEAFCUTTER_NO_CACHE=1  → stream weights from disk per token (bounded RSS)

use leafcutter::inference::engine::Engine;

fn rss_mb() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .map(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0)
                / 1024
        })
        .unwrap_or(0)
}

fn peak_mb() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .map(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0)
                / 1024
        })
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).expect("Usage: generate_test <model> [prompt] [max_tokens]");
    let prompt = args.get(2).cloned().unwrap_or_else(|| "The capital of France is".to_string());
    let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(15);

    println!("🌿 Leafcutter Generation Test");
    println!("   Model: {}", model);
    println!("   Prompt: {:?}", prompt);
    println!("   Max tokens: {}", max_tokens);
    println!("   RSS at start: {} MB (peak {})", rss_mb(), peak_mb());

    let t0 = std::time::Instant::now();
    let mut engine = Engine::load(model).expect("Failed to load engine");
    println!("   Engine loaded in {:.1}s | {} layers, hidden={}, vocab={}",
        t0.elapsed().as_secs_f64(), engine.config.num_hidden_layers, engine.config.hidden_size, engine.config.vocab_size);
    println!("   RSS after load: {} MB (peak {})", rss_mb(), peak_mb());

    let tokens = engine.tokenize(&prompt, true);
    println!("   Prompt -> {} tokens: {:?}", tokens.len(), tokens);

    let t1 = std::time::Instant::now();
    let gen = engine.generate_native(&tokens, max_tokens, 0.0, 1.0);
    let dt = t1.elapsed();

    println!("\n=== Generated {} tokens in {:.1}s ({:.2} tok/s) ===",
        gen.len(), dt.as_secs_f64(), gen.len() as f64 / dt.as_secs_f64());
    println!("   Token ids: {:?}", gen);

    let text = engine.decode(&gen);
    println!("   Decoded: {:?}", text);
    println!("   Prompt+text: {:?}", format!("{}{}", prompt, text));

    println!("\n=== MEMORY ===");
    println!("   RSS final: {} MB", rss_mb());
    println!("   PEAK RSS (VmHWM): {} MB", peak_mb());
    println!("══════════════════════════════════════");
}
