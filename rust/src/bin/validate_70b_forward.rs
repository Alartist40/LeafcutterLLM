//! Minimal 70B forward pass — 1 token through all 80 layers.
//! Reports RSS before/after to validate inference memory.

use leafcutter::inference::engine::Engine;
use leafcutter::tokenizer::{BaseTokenizer, Tokenizer};

fn read_rss_mb() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            return parts.get(1).and_then(|s| s.parse::<usize>().ok()).map(|kb| kb / 1024);
        }
    }
    None
}

fn read_peak_mb() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmHWM:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            return parts.get(1).and_then(|s| s.parse::<usize>().ok()).map(|kb| kb / 1024);
        }
    }
    None
}

fn main() {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/xander/Documents/portfolio/AI Models/Meta-Llama-3.1-70B-Instruct-Q4_K_S.gguf".to_string());

    println!("🌿 Leafcutter 70B Forward Validator");
    println!("   Model: {}", model);

    let rss_start = read_rss_mb().unwrap_or(0);
    println!("📊 RSS before load: {} MB", rss_start);

    let mut engine = Engine::load(&model).expect("Failed to load engine");
    let rss_loaded = read_rss_mb().unwrap_or(0);
    println!("✅ Engine loaded: {} layers, hidden_size={}", engine.config.num_hidden_layers, engine.config.hidden_size);
    println!("📊 RSS after load:  {} MB", rss_loaded);

    let tok = Tokenizer::from_file("tests/tokenizer_llama.json").expect("Tokenizer");
    let tokens = tok.encode("Hi");
    println!("📝 Tokens: {:?} (len={})", tokens, tokens.len());

    println!("\n⏳ Running forward pass...");
    let start = std::time::Instant::now();
    let _logits = engine.forward(&tokens);
    let elapsed = start.elapsed();
    let rss_forward = read_rss_mb().unwrap_or(0);
    let peak_forward = read_peak_mb().unwrap_or(0);

    println!("✅ Forward pass done in {:?}", elapsed);
    println!("📊 RSS after forward: {} MB", rss_forward);
    println!("📊 Peak RSS (total):  {} MB", peak_forward);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  70B FORWARD PASS MEMORY");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  RSS before load:          {:>8} MB", rss_start);
    println!("  RSS after load:           {:>8} MB", rss_loaded);
    println!("  RSS after forward:        {:>8} MB", rss_forward);
    println!("  PEAK RSS (VmHWM):         {:>8} MB  ★", peak_forward);
    println!("───────────────────────────────────────────────────────────────");
    println!("  Claim: 70B inference peak < 2.5 GB RAM");
    if peak_forward < 2560 {
        println!("  RESULT: ✅ CLAIM VALIDATED — peak {} MB", peak_forward);
    } else {
        println!("  RESULT: ⚠️  Peak {} MB exceeds 2.5 GB target", peak_forward);
    }
    println!("═══════════════════════════════════════════════════════════════");
}
