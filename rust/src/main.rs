//! LeafcutterLLM v0.9.0 — Unified CLI
//!
//! Subcommands:
//!   leafcutter server   --model PATH [--port 8081]     # HTTP API server (requires llama-ffi)
//!   leafcutter generate --model PATH --prompt "..."      # One-shot text generation
//!   leafcutter chat     --model PATH                     # Interactive chat
//!   leafcutter list-models [--dir ~/models]              # List downloaded GGUF models
//!
//! For Cynapse integration:
//!   leafcutter --meta                                    # Print synapse metadata JSON

use clap::{CommandFactory, Parser, Subcommand};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "llama-ffi")]
use leafcutter::llama_ffi::{backend_init, backend_free, LlamaModel, LlamaContext};

#[cfg(feature = "llama-ffi")]
use leafcutter::api::FfiEngine;

use leafcutter::model::gguf::GGUFile;
use leafcutter::tokenizer::chat_template::apply_chat_template_from_gguf;

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
        #[arg(short, long, default_value = "")]
        model: String,
        /// HTTP port to listen on
        #[arg(short, long, default_value_t = 8081)]
        port: u16,
        /// Host/interface to bind (default: 127.0.0.1 loopback; set to
        /// 0.0.0.0 to expose on all interfaces — only with auth enabled).
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Engine type (native-streaming or llama-ffi)
        #[arg(short, long, default_value = "native-streaming")]
        engine: String,
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
        /// System prompt (optional)
        #[arg(long, default_value = "")]
        system: String,
        /// Skip chat-template formatting and use raw prompt
        #[arg(long, default_value_t = false)]
        raw: bool,
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
        Some(Commands::Server { model, port, host, engine, benchmark }) => {
            #[cfg(feature = "llama-ffi")]
            run_server(&model, port, &host, &engine, benchmark).await;
            #[cfg(not(feature = "llama-ffi"))]
            {
                eprintln!("❌ Server mode requires llama.cpp FFI. Build with: cargo build --features llama-ffi");
                std::process::exit(1);
            }
        }
        Some(Commands::Generate { model, prompt, system, raw, max_tokens, temperature, threads, ctx_size, gpu_layers }) => {
            #[cfg(feature = "llama-ffi")]
            cmd_generate(&model, &prompt, &system, raw, max_tokens, temperature, threads, ctx_size, gpu_layers);
            #[cfg(not(feature = "llama-ffi"))]
            cmd_generate_native(&model, &prompt, &system, raw, max_tokens, temperature);
        }
        Some(Commands::Chat { model, system, max_tokens, temperature, threads, ctx_size, gpu_layers }) => {
            #[cfg(feature = "llama-ffi")]
            cmd_chat(&model, &system, max_tokens, temperature, threads, ctx_size, gpu_layers);
            #[cfg(not(feature = "llama-ffi"))]
            {
                eprintln!("❌ Chat mode requires llama.cpp FFI. Build with: cargo build --features llama-ffi");
                eprintln!("   (Native engine generate is available via: leafcutter generate --model PATH --prompt \"...\")");
                std::process::exit(1);
            }
        }
        Some(Commands::ListModels { dir }) => {
            cmd_list_models(&dir);
        }
        None => {
            // No subcommand: print help and exit. Never bake a hardcoded
            // user path into the binary.
            let _ = Cli::command().print_help();
            println!();
            std::process::exit(0);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// llama-ffi enabled paths
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "llama-ffi")]
async fn run_server(model_path: &str, port: u16, host: &str, engine_type: &str, benchmark: bool) {
    use leafcutter::api::{FfiEngine, NativeStreamingEngine, LeafcutterEngine};

    println!("🌿 LeafcutterLLM v0.9.5 (Server Mode: {})", engine_type);
    println!("   Model: {}", model_path);
    println!("   Host: {}", host);

    let engine: Arc<dyn LeafcutterEngine> = if engine_type == "native-streaming" {
        match NativeStreamingEngine::load(model_path) {
            Ok(e) => {
                println!("✅ Native Streaming Engine loaded (low RAM mode)");
                Arc::new(e)
            }
            Err(e) => {
                eprintln!("❌ Failed to load native engine: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        match FfiEngine::load(model_path) {
            Ok(e) => {
                println!("✅ llama.cpp FFI Engine loaded (Full load mode)");
                Arc::new(e)
            }
            Err(e) => {
                eprintln!("❌ Failed to load FFI engine: {}", e);
                std::process::exit(1);
            }
        }
    };

    if benchmark {
        run_ffi_benchmark(&engine);
        return;
    }

    leafcutter::api::run_server(engine, port, host).await;
}

#[cfg(feature = "llama-ffi")]
fn run_ffi_benchmark(engine: &Arc<dyn leafcutter::api::LeafcutterEngine>) {
    use std::time::Instant;

    println!("\n🏁 Running benchmark...");
    let prompt = "Hello";
    let start = Instant::now();
    let (text, tokens) = engine.generate(prompt, 10, 0.7, 0.9).unwrap();
    let elapsed = start.elapsed();

    let tok_per_sec = tokens.len() as f64 / elapsed.as_secs_f64();
    println!("Generated '{}' in {:?}", text.trim(), elapsed);
    println!("Throughput: {:.2} tok/sec", tok_per_sec);
}

#[cfg(feature = "llama-ffi")]
fn cmd_generate(
    model_path: &PathBuf,
    prompt: &str,
    system: &str,
    raw: bool,
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

    // Auto-detect chat template from GGUF metadata
    let prompt_text = if raw {
        prompt.to_string()
    } else {
        let gguf = GGUFile::open(model_path.to_str().unwrap()).ok();
        let formatted = if let Some(ref file) = gguf {
            apply_chat_template_from_gguf(&file.metadata, system, prompt)
        } else {
            if system.is_empty() {
                prompt.to_string()
            } else {
                format!("{system}\n\n{prompt}")
            }
        };
        eprintln!("🎭 Chat template applied");
        formatted
    };

    let tokens = ctx.tokenize(&prompt_text, true, true);
    eprintln!("📝 Prompt tokens: {}", tokens.len());
    eprintln!("📝 FFI Token IDs: {:?}", tokens);

    let generated = ctx.generate(&tokens, max_tokens, temperature, model.eos_token());

    let text: String = generated.iter()
        .map(|&t| ctx.token_to_piece(t))
        .collect();
    print!("{}", text);
    io::stdout().flush().unwrap();

    backend_free();
}

#[cfg(feature = "llama-ffi")]
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
        print!("You: ");
        if io::stdout().flush().is_err() {
            break;
        }

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim().to_string();

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

        // Context management: truncate if approaching limit.
        // Use saturating_sub to avoid underflow when max_tokens > ctx_size.
        let available = (ctx_size as usize).saturating_sub(max_tokens);
        let tokens = if tokens.len() > available {
            eprintln!("[⚠️  Context nearly full — truncating older messages]");
            let system_prompt = format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", system
            );
            let system_tokens = ctx.tokenize(&system_prompt, false, true);
            let keep = available.saturating_sub(system_tokens.len());
            let mut truncated = system_tokens;
            truncated.extend_from_slice(&tokens[tokens.len().saturating_sub(keep)..]);
            truncated
        } else {
            tokens
        };

        print!("\n🤖 Assistant: ");
        if io::stdout().flush().is_err() {
            break;
        }

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

// ═════════════════════════════════════════════════════════════════════════════
// Native engine fallback (no llama-ffi)
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "llama-ffi"))]
fn cmd_generate_native(
    model_path: &PathBuf,
    prompt: &str,
    system: &str,
    raw: bool,
    max_tokens: usize,
    temperature: f32,
) {
    use leafcutter::tokenizer::{Tokenizer, GgufBpeTokenizer, BaseTokenizer};
    use leafcutter::inference::engine::Engine;

    eprintln!("🌿 LeafcutterLLM Native Engine (no llama.cpp FFI)");
    eprintln!("   Model: {}", model_path.display());

    let mut engine = match Engine::load(model_path.to_str().unwrap()) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    // Try HF tokenizer first, then fall back to GGUF-native BPE tokenizer
    let expected_vocab = engine.config.vocab_size;
    let hf_tok = Tokenizer::from_file("models/tokenizer_qwen35.json")
        .or_else(|_| Tokenizer::from_file("tests/tokenizer_qwen35.json"))
        .or_else(|_| Tokenizer::from_file("tests/tokenizer_llama.json"))
        .or_else(|_| Tokenizer::from_file("tests/tokenizer.json"))
        .ok();
    let gguf_tok = GgufBpeTokenizer::from_gguf(model_path.to_str().unwrap());

    let use_hf = match (&hf_tok, &gguf_tok) {
        (Some(t), _) if t.vocab_size() == expected_vocab => {
            eprintln!("📝 Using HF tokenizer (vocab={})", t.vocab_size());
            true
        }
        (Some(t), Some(_)) => {
            // Temporarily allow HF tokenizer even with vocab mismatch for testing
            eprintln!("⚠️  HF tokenizer vocab mismatch (HF={}, model={}), using HF tokenizer anyway for testing",
                t.vocab_size(), expected_vocab);
            true
        }
        (None, Some(_)) => {
            eprintln!("📝 Using GGUF-native tokenizer (vocab={})", gguf_tok.as_ref().unwrap().vocab_size());
            false
        }
        (Some(t), None) => {
            eprintln!("⚠️  HF tokenizer vocab mismatch (HF={}, model={}), no GGUF vocab fallback",
                t.vocab_size(), expected_vocab);
            true
        }
        (None, None) => {
            eprintln!("❌ No tokenizer found. Place tests/tokenizer_llama.json or ensure GGUF has tokenizer.ggml.tokens");
            std::process::exit(1);
        }
    };

    // Auto-detect chat template from GGUF metadata
    let prompt_text = if raw {
        prompt.to_string()
    } else {
        let gguf = GGUFile::open(model_path.to_str().unwrap()).ok();
        if let Some(ref file) = gguf {
            let formatted = apply_chat_template_from_gguf(&file.metadata, system, prompt);
            let family = if formatted.starts_with("[SYSTEM_PROMPT]") {
                "Ministral"
            } else if formatted.contains("<|start_header_id|>") {
                "Llama-3"
            } else if formatted.contains("[INST]") {
                "Mistral"
            } else if formatted.contains("<|im_start|>") {
                "ChatML"
            } else if formatted.contains("<start_of_turn>") {
                "Gemma"
            } else {
                "Unknown / plain"
            };
            eprintln!("🎭 Chat template applied (detected: {})", family);
            formatted
        } else {
            if system.is_empty() {
                prompt.to_string()
            } else {
                format!("{system}\n\n{prompt}")
            }
        }
    };

    let tokens = if use_hf {
        match &hf_tok {
            Some(t) => t.encode(&prompt_text),
            None => {
                eprintln!("❌ HF tokenizer selected but not available");
                std::process::exit(1);
            }
        }
    } else {
        match &gguf_tok {
            Some(t) => t.encode(&prompt_text),
            None => {
                eprintln!("❌ No GGUF-embedded tokenizer found. Cannot tokenize input.");
                eprintln!("   Ensure the GGUF file contains tokenizer.ggml.tokens metadata.");
                std::process::exit(1);
            }
        }
    };
    eprintln!("📝 Prompt tokens: {}", tokens.len());

    let info = engine.info();
    eprintln!("   Arch: {}  Layers: {}  Hidden: {}", info.architecture, info.total_layers, info.hidden_size);

    let generated = engine.generate(&tokens, max_tokens, temperature, 0.9);
    let text = if use_hf {
        match &hf_tok {
            Some(t) => t.decode(&generated),
            None => String::new(),
        }
    } else {
        match &gguf_tok {
            Some(t) => t.decode(&generated),
            None => String::new(),
        }
    };
    println!("{}", text);
}

// ═════════════════════════════════════════════════════════════════════════════
// Shared helpers
// ═════════════════════════════════════════════════════════════════════════════

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
