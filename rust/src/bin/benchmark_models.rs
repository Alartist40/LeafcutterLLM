//! Comprehensive model benchmark — tests all available GGUF models

use leafcutter::inference::engine::Engine;
use std::time::Instant;

fn read_rss_mb() -> usize {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(Ok(v)) = parts.get(1).map(|s| s.parse::<usize>()) {
                    return v / 1024;
                }
            }
        }
    }
    0
}

fn test_model(path: &str, prompt: &str, max_tokens: usize) -> Option<ModelResult> {
    let model_name = path.split('/').last().unwrap_or(path).to_string();
    println!("\n{}", "=".repeat(70));
    println!("Testing: {}", model_name);
    println!("{}", "=".repeat(70));

    let rss_before = read_rss_mb();
    let load_start = Instant::now();
    let mut engine = match Engine::load(path) {
        Ok(e) => e,
        Err(e) => {
            println!("❌ FAILED TO LOAD: {}", e);
            return None;
        }
    };
    let load_time = load_start.elapsed();
    let rss_after_load = read_rss_mb();

    let arch = engine.model.architecture.name();
    let backend = if engine.is_ffi() { "FFI (llama.cpp)" } else { "Native" };
    println!("Architecture: {} | Backend: {}", arch, backend);
    println!("Load time: {:.2}s | RSS after load: {} MB", load_time.as_secs_f64(), rss_after_load);

    let prompt_tokens = if engine.is_ffi() {
        engine.tokenize(prompt, false)
    } else {
        let tok = engine.tokenizer_from_model()
            .expect("Native models require GGUF embedded tokenizer");
        tok.encode(prompt, false)
    };

    let gen_start = Instant::now();
    let generated = engine.generate(&prompt_tokens, max_tokens, 0.7, 0.9);
    let gen_time = gen_start.elapsed();
    let rss_after_gen = read_rss_mb();

    let tok_per_sec = generated.len() as f64 / gen_time.as_secs_f64();
    let time_per_tok_ms = if !generated.is_empty() {
        gen_time.as_secs_f64() * 1000.0 / generated.len() as f64
    } else {
        0.0
    };

    let output = if engine.is_ffi() {
        engine.decode(&generated)
    } else {
        engine.tokenizer_from_model().map(|t| t.decode(&generated)).unwrap_or_default()
    };
    let full_output = if engine.is_ffi() {
        engine.decode(&prompt_tokens.iter().chain(generated.iter()).copied().collect::<Vec<_>>())
    } else {
        engine.tokenizer_from_model().map(|t| t.decode(&prompt_tokens.iter().chain(generated.iter()).copied().collect::<Vec<_>>())).unwrap_or_default()
    };

    println!("Generated {} tokens in {:.2}s ({:.2} tok/s, {:.1} ms/tok)",
        generated.len(), gen_time.as_secs_f64(), tok_per_sec, time_per_tok_ms);
    println!("RSS after gen: {} MB | Delta: {} MB", rss_after_gen, rss_after_gen.saturating_sub(rss_after_load));
    println!("Output: {}", output.trim());

    Some(ModelResult {
        name: model_name,
        arch: arch.to_string(),
        backend: backend.to_string(),
        load_time_sec: load_time.as_secs_f64(),
        gen_time_sec: gen_time.as_secs_f64(),
        tokens_generated: generated.len(),
        tok_per_sec,
        time_per_tok_ms,
        rss_load_mb: rss_after_load,
        rss_gen_mb: rss_after_gen,
        output: output.trim().to_string(),
        full_output: full_output.trim().to_string(),
    })
}

#[derive(Debug)]
struct ModelResult {
    name: String,
    arch: String,
    backend: String,
    load_time_sec: f64,
    gen_time_sec: f64,
    tokens_generated: usize,
    tok_per_sec: f64,
    time_per_tok_ms: f64,
    rss_load_mb: usize,
    rss_gen_mb: usize,
    output: String,
    full_output: String,
}

fn main() {
    let models = vec![
        ("/home/xander/Documents/portfolio/AI Models/Qwen3.5-0.8B-Q4_0.gguf", "Hello", 10),
        ("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf", "Hello", 10),
        ("/home/xander/Documents/portfolio/AI Models/Qwen3.5-2B-BF16.gguf", "Hello", 10),
        ("/home/xander/Documents/portfolio/AI Models/Meta-Llama-3.1-8B-Instruct-Q4_0_4_4.gguf", "Hello", 10),
        ("/home/xander/Documents/portfolio/AI Models/Qwen3.5-9B-IQ4_NL.gguf", "Hello", 10),
        ("/home/xander/Documents/portfolio/AI Models/Qwen3.6-27B-IQ4_NL.gguf", "Hello", 10),
        ("/home/xander/Documents/portfolio/AI Models/Meta-Llama-3.1-70B-Instruct-IQ1_M.gguf", "Hello", 5),
    ];

    let mut results = Vec::new();

    for (path, prompt, tokens) in models {
        if !std::path::Path::new(path).exists() {
            println!("\n⚠️  Model not found: {}", path);
            continue;
        }
        if let Some(result) = test_model(path, prompt, tokens) {
            results.push(result);
        }
    }

    // Summary table
    println!("\n\n{}", "=".repeat(120));
    println!("BENCHMARK SUMMARY");
    println!("{}", "=".repeat(120));
    println!("{:<45} {:>10} {:>12} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "Model", "Backend", "Load(s)", "Gen(s)", "Tok/s", "ms/Tok", "RSS(MB)", "Status");
    println!("{}", "-".repeat(120));

    for r in &results {
        let status = if r.output.len() > 10 && r.output.chars().any(|c| c.is_alphabetic()) {
            "✅ Coherent"
        } else {
            "❌ Garbled"
        };
        println!("{:<45} {:>10} {:>12.2} {:>10.2} {:>10.2} {:>10.1} {:>10} {:>12}",
            r.name, r.backend, r.load_time_sec, r.gen_time_sec, r.tok_per_sec,
            r.time_per_tok_ms, r.rss_gen_mb, status);
    }

    println!("\n{}", "=".repeat(120));
    println!("DETAILED OUTPUTS");
    println!("{}", "=".repeat(120));
    for r in &results {
        println!("\n🤖 {} ({}):", r.name, r.backend);
        println!("   {}", r.full_output.replace('\n', " \\n "));
    }
}
