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

use leafcutter::bridge::HybridEngine;

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

    /// Print Cynapse synapse metadata JSON and exit
    #[arg(long, default_value_t = false)]
    meta: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Cynapse synapse metadata query — must work without loading a model
    if args.meta {
        let meta = serde_json::json!({
            "name": "leafcutter",
            "version": "0.8.0",
            "description": "CPU-optimized LLM inference engine with quantization support",
            "author": "Alartist40",
            "capabilities": [
                "llm_inference",
                "model_loading",
                "quantization",
                "speculative_decoding",
                "cpu_optimized"
            ],
            "command": "",
            "args": [],
            "env": {}
        });
        println!("{}", meta.to_string());
        return;
    }

    println!("🌿 LeafcutterLLM v0.8.0 (Rust Rewrite)");
    println!("   Model: {}", args.model);

    // Load model (tries native Rust, falls back to llama.cpp bridge)
    let engine = match HybridEngine::load(&args.model) {
        Ok(e) => {
            let backend = if e.native.is_some() { "native" } else { "bridge" };
            println!("✅ Model loaded via {} backend", backend);
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

fn run_benchmark(engine: Arc<Mutex<HybridEngine>>) {
    use std::time::Instant;

    println!("\n🏁 Running benchmark...");
    let mut eng = engine.lock().unwrap();

    let prompt = "Hello";

    let start = Instant::now();
    let generated_text = eng.generate(prompt, 10, 0.7, 0.9);
    let elapsed = start.elapsed();

    let tok_per_sec = generated_text.len() as f64 / elapsed.as_secs_f64();
    println!("Generated '{}' in {:?}", generated_text.trim(), elapsed);
    println!("Throughput: {:.2} chars/sec", tok_per_sec);
}
