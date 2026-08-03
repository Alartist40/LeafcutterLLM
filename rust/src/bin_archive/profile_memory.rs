//! Memory profiler — verify layer streaming keeps RSS bounded.

use leafcutter::inference::engine::Engine;

fn read_rss_mb() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").ok().unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(Ok(kb)) = parts.get(1).map(|s| s.parse::<usize>()) {
                return kb / 1024;
            }
        }
    }
    0
}

fn read_peak_mb() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").ok().unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmHWM:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(Ok(kb)) = parts.get(1).map(|s| s.parse::<usize>()) {
                return kb / 1024;
            }
        }
    }
    0
}

fn main() {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/xander/Documents/portfolio/AI Models/Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf".to_string());

    println!("🌿 Leafcutter Memory Profiler");
    println!("   Model: {}", model);
    println!();

    let baseline = read_rss_mb();
    println!("[0] Baseline (before load):     {:>6} MB", baseline);

    let mut engine = Engine::load(&model).expect("Failed to load engine");
    let after_load = read_rss_mb();
    let peak_after_load = read_peak_mb();
    println!("[1] After load:                 {:>6} MB  (peak: {} MB)", after_load, peak_after_load);
    println!("    Layers: {}, hidden_size: {}", engine.config.num_hidden_layers, engine.config.hidden_size);

    // Use a single-token prompt for consistent measurement
    let tokens = vec![1usize]; // BOS token

    for i in 0..5 {
        let _logits = engine.forward(&tokens);
        let rss = read_rss_mb();
        let peak = read_peak_mb();
        println!("[{}] After forward #{}:          {:>6} MB  (peak: {} MB)", i + 2, i + 1, rss, peak);
    }

    let final_rss = read_rss_mb();
    let final_peak = read_peak_mb();

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  MEMORY PROFILE RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Baseline RSS:               {:>8} MB", baseline);
    println!("  RSS after load:             {:>8} MB", after_load);
    println!("  RSS after 5 forwards:       {:>8} MB", final_rss);
    println!("  PEAK RSS (VmHWM):           {:>8} MB  ★", final_peak);
    println!("───────────────────────────────────────────────────────────────");
    if final_peak < 1024 {
        println!("  RESULT: ✅ Under 1GB — layer streaming is working");
    } else if final_peak < 2048 {
        println!("  RESULT: ⚠️  Under 2GB — acceptable but higher than expected");
    } else {
        println!("  RESULT: ❌ Over 2GB — madvise may not be clearing pages");
    }
    println!("═══════════════════════════════════════════════════════════════");
}
