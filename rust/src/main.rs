//! LeafcutterLLM v0.8.0 — Full Rust Rewrite
//!
//! Memory-safe LLM inference engine with layer streaming and K-quant support.
//!
//! Usage:
//!   cargo run --release -- --model /path/to/model.gguf --port 8081
//!
//! TEAM NOTE: This is Option C — the full Rust rewrite.
//! Preserve this file and all tests. See LEAFcutter_TEST_RESULTS.md for benchmarks.

use clap::Parser;
use std::sync::{Arc, Mutex};

use leafcutter::inference::engine::Engine;

#[derive(Parser, Debug)]
#[command(name = "leafcutter")]
#[command(about = "Memory-safe LLM inference engine")]
struct Args {
    #[arg(short, long, default_value = "/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf")]
    model: String,

    #[arg(short, long, default_value_t = 8081)]
    port: u16,

    #[arg(long, default_value_t = false)]
    benchmark: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    println!("🌿 LeafcutterLLM v0.8.0 (Rust Rewrite)");
    println!("   Model: {}", args.model);

    // Load model
    let engine = match Engine::load(&args.model) {
        Ok(e) => {
            println!("✅ Model loaded: {} layers, vocab={}", e.config.num_hidden_layers, e.config.vocab_size);
            Arc::new(Mutex::new(e))
        }
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    if args.benchmark {
        run_benchmark(engine);
        return;
    }

    // Start HTTP server
    leafcutter::api::run_server(engine, args.port).await;
}

fn run_benchmark(engine: Arc<Mutex<Engine>>) {
    use std::time::Instant;

    println!("\n🏁 Running benchmark...");
    let mut eng = engine.lock().unwrap();

    let prompt = "Hello";
    let tokens: Vec<usize> = prompt.bytes().map(|b| b as usize).collect();

    let start = Instant::now();
    let generated = eng.generate(&tokens, 10, 0.7, 0.9);
    let elapsed = start.elapsed();

    let tok_per_sec = generated.len() as f64 / elapsed.as_secs_f64();
    println!("Generated {} tokens in {:?}", generated.len(), elapsed);
    println!("Throughput: {:.2} tok/sec", tok_per_sec);
}
