//! LeafcutterLLM v0.9.0 — Unified CLI
//!
//! Subcommands:
//!   leafcutter server   --model PATH [--port 8081]     # HTTP API server
//!   leafcutter generate --model PATH --prompt "..."      # One-shot text generation
//!   leafcutter chat     --model PATH                     # Interactive chat
//!   leafcutter list-models [--dir ~/models]              # List downloaded GGUF models
//!
//! For Cynapse integration:
//!   leafcutter --meta                                    # Print synapse metadata JSON

use clap::{Parser, Subcommand};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

// ─── FFI imports for generate/chat ───────────────────────────────────────────
use leafcutter::llama_ffi::{backend_init, backend_free, LlamaModel, LlamaContext};

// ─── Server imports ──────────────────────────────────────────────────────────
use leafcutter::ffi_server::FfiEngine;

#[derive(Parser)]
#[command(name = "leafcutter")]
#[command(about = "LeafcutterLLM — Run LLMs locally, fast and light")]
#[command(version = "0.9.0")]
struct Cli {
    /// Print Cynapse synapse metadata JSON and exit
    #[arg(long, global = true)]
    meta: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the HTTP API server (OpenAI-compatible)
    Server {
        /// Path to GGUF model file
        #[arg(short, long, default_value = "/home/xander/Documents/portfolio/AI Models/Qwen3.5-9B-IQ4_NL.gguf")]
        model: String,
        /// HTTP port to listen on
        #[arg(short, long, default_value_t = 8081)]
        port: u16,
        /// Run a quick benchmark instead of starting the server
        #[arg(long, default_value_t = false)]
        benchmark: bool,
    },
    /// Generate text from a prompt (one-shot)
    Generate {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,
        /// Prompt text
        #[arg(short, long)]
        prompt: String,
        /// Max tokens to generate
        #[arg(short, long, default_value = "128")]
        max_tokens: usize,
        /// Sampling temperature (0.0 = greedy)
        #[arg(short, long, default_value = "0.8")]
        temperature: f32,
        /// Number of CPU threads
        #[arg(long, default_value = "4")]
        threads: i32,
        /// Context size
        #[arg(long, default_value = "2048")]
        ctx_size: u32,
        /// GPU layers to offload (0 = CPU only)
        #[arg(long, default_value = "0")]
        gpu_layers: i32,
    },
    /// Interactive chat session
    Chat {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,
        /// System prompt
        #[arg(long, default_value = "You are a helpful assistant.")]
        system: String,
        /// Max tokens per response
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        /// Number of CPU threads
        #[arg(long, default_value = "4")]
        threads: i32,
        /// Context size
        #[arg(long, default_value = "4096")]
        ctx_size: u32,
        /// GPU layers to offload
        #[arg(long, default_value = "0")]
        gpu_layers: i32,
    },
    /// List available models in the models directory
    ListModels {
        /// Models directory
        #[arg(short, long, default_value = "~/models")]
        dir: PathBuf,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Cynapse synapse metadata query — must work without loading a model
    if cli.meta {
        let meta = serde_json::json!({
            "name": "leafcutter",
            "version": "0.9.0",
            "description": "CPU-optimized LLM inference engine with quantization support and llama.cpp FFI",
            "author": "Alartist40",
            "capabilities": [
                "llm_inference",
                "model_loading",
                "quantization",
                "speculative_decoding",
                "cpu_optimized",
                "llama_cpp_ffi",
                "text_generation",
                "interactive_chat"
            ],
            "commands": ["server", "generate", "chat", "list-models"],
            "env": { "LD_LIBRARY_PATH": "path/to/llama.cpp/build/bin" }
        });
        println!("{}", meta.to_string());
        return;
    }

    match cli.command {
        Some(Commands::Server { model, port, benchmark }) => {
            run_server(&model, port, benchmark).await;
        }
        Some(Commands::Generate { model, prompt, max_tokens, temperature, threads, ctx_size, gpu_layers }) => {
            cmd_generate(&model, &prompt, max_tokens, temperature, threads, ctx_size, gpu_layers);
        }
        Some(Commands::Chat { model, system, max_tokens, temperature, threads, ctx_size, gpu_layers }) => {
            cmd_chat(&model, &system, max_tokens, temperature, threads, ctx_size, gpu_layers);
        }
        Some(Commands::ListModels { dir }) => {
            cmd_list_models(&dir);
        }
        None => {
            // Default to server mode for backward compatibility
            run_server(
                "/home/xander/Documents/portfolio/AI Models/Qwen3.5-9B-IQ4_NL.gguf",
                8081,
                false,
            ).await;
        }
    }
}

// ─── Server Mode ─────────────────────────────────────────────────────────────

async fn run_server(model_path: &str, port: u16, benchmark: bool) {
    println!("🌿 LeafcutterLLM v0.9.0 (FFI Server Mode)");
    println!("   Model: {}", model_path);

    let engine = match FfiEngine::load(model_path) {
        Ok(e) => {
            println!("✅ Model loaded via direct llama.cpp FFI");
            Arc::new(e)
        }
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    if benchmark {
        run_ffi_benchmark(&engine);
        return;
    }

    leafcutter::api::run_server(engine, port).await;
}

fn run_ffi_benchmark(engine: &Arc<FfiEngine>) {
    use std::time::Instant;

    println!("\n🏁 Running benchmark...");
    let prompt = "Hello";
    let start = Instant::now();
    let result = engine.generate(prompt, 10, 0.7).unwrap();
    let elapsed = start.elapsed();

    let tok_per_sec = result.tokens.len() as f64 / elapsed.as_secs_f64();
    println!("Generated '{}' in {:?}", result.text.trim(), elapsed);
    println!("Throughput: {:.2} tok/sec", tok_per_sec);
}

// ─── Generate Mode (FFI) ─────────────────────────────────────────────────────

fn cmd_generate(
    model_path: &PathBuf,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    threads: i32,
    ctx_size: u32,
    gpu_layers: i32,
) {
    backend_init();

    eprintln!("Loading model: {}", model_path.display());
    let model = match LlamaModel::load(model_path, gpu_layers) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!("✅ Model loaded. n_vocab={}, n_embd={}, n_layer={}",
             model.n_vocab(), model.n_embd(), model.n_layer());

    let mut ctx = match LlamaContext::new(&model, ctx_size, threads) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("❌ Failed to create context: {}", e);
            std::process::exit(1);
        }
    };

    let tokens = ctx.tokenize(prompt, true, true);
    eprintln!("📝 Prompt tokens: {}", tokens.len());

    let generated = ctx.generate(&tokens, max_tokens, temperature, model.eos_token());

    let text: String = generated.iter()
        .map(|&t| ctx.token_to_piece(t))
        .collect();
    print!("{}", text);
    io::stdout().flush().unwrap();

    backend_free();
}

// ─── Chat Mode (FFI) ─────────────────────────────────────────────────────────

fn cmd_chat(
    model_path: &PathBuf,
    system: &str,
    max_tokens: usize,
    temperature: f32,
    threads: i32,
    ctx_size: u32,
    gpu_layers: i32,
) {
    backend_init();

    eprintln!("╔══════════════════════════════════════════╗");
    eprintln!("║     🌿 LeafcutterLLM Chat — v0.9.0       ║");
    eprintln!("╚══════════════════════════════════════════╝");
    eprintln!("Loading model: {}", model_path.display());

    let model = match LlamaModel::load(model_path, gpu_layers) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!("✅ Model loaded. n_vocab={}, n_embd={}, n_layer={}",
             model.n_vocab(), model.n_embd(), model.n_layer());

    let mut ctx = match LlamaContext::new(&model, ctx_size, threads) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("❌ Failed to create context: {}", e);
            std::process::exit(1);
        }
    };

    let mut conversation = format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
        system
    );

    eprintln!("\nType your message and press Enter. Type 'quit' or 'exit' to stop.\n");

    loop {
        print!("\n🧑 You: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            eprintln!("👋 Goodbye!");
            break;
        }
        if input.is_empty() {
            continue;
        }

        conversation.push_str(&format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            input
        ));

        let tokens = ctx.tokenize(&conversation, false, true);

        // Context management: truncate if approaching limit
        let tokens = if tokens.len() > ctx_size as usize - max_tokens {
            eprintln!("[⚠️  Context nearly full — truncating older messages]");
            let system_prompt = format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", system
            );
            let system_tokens = ctx.tokenize(&system_prompt, false, true);
            let keep = ctx_size as usize - max_tokens - system_tokens.len();
            let mut truncated = system_tokens;
            truncated.extend_from_slice(&tokens[tokens.len() - keep..]);
            truncated
        } else {
            tokens
        };

        print!("\n🤖 Assistant: ");
        io::stdout().flush().unwrap();

        let generated = ctx.generate(&tokens, max_tokens, temperature, model.eos_token());
        let response: String = generated.iter()
            .map(|&t| ctx.token_to_piece(t))
            .collect();
        print!("{}", response);
        println!();

        conversation.push_str(&response);
        conversation.push_str("<|eot_id|>");
    }

    backend_free();
}

// ─── List Models ─────────────────────────────────────────────────────────────

fn cmd_list_models(dir: &PathBuf) {
    let dir_str = dir.to_string_lossy().to_string();
    let dir = shellexpand::tilde(&dir_str);
    let dir = PathBuf::from(dir.as_ref());

    eprintln!("📁 Models directory: {}", dir.display());

    if !dir.exists() {
        eprintln!("❌ Directory does not exist.");
        return;
    }

    let mut found = false;
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                found = true;
                let size = std::fs::metadata(&path)
                    .map(|m| format_size(m.len()))
                    .unwrap_or_else(|_| "?".into());
                println!("  {:<50} {}", path.file_name().unwrap_or_default().to_string_lossy(), size);
            }
        }
    }

    if !found {
        eprintln!("No .gguf models found in {}", dir.display());
        eprintln!("Download models with: cynapse model download <hf-id> <filename>");
    }
}

fn format_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    format!("{:.2} {}", size, UNITS[unit_idx])
}
