//! LeafcutterLLM v0.9.0 — Unified CLI
//!
//! Friendly commands (no FFI needed, native engine):
//!   leafcutter list                      List available GGUF models
//!   leafcutter run <model>               Start streaming chat (Ollama-style)
//!   leafcutter help                      Show all commands
//!
//! Advanced:
//!   leafcutter serve --model PATH [--port 8081]          HTTP API server
//!   leafcutter generate --model PATH --prompt "..."      One-shot generation
//!   leafcutter chat --model PATH                         Interactive chat (FFI)
//!   leafcutter list-models [--dir ~/models]              List models (legacy flag form)
//!
//! For Cynapse integration:
//!   leafcutter --meta                                    Print synapse metadata JSON

use clap::{CommandFactory, Parser, Subcommand};
use std::io::{self, Write};
use std::path::PathBuf;

#[cfg(feature = "llama-ffi")]
use std::sync::Arc;

#[cfg(feature = "llama-ffi")]
use leafcutter::llama_ffi::{backend_init, backend_free, LlamaModel, LlamaContext};

#[cfg(feature = "llama-ffi")]
use leafcutter::api::FfiEngine;

use leafcutter::model::gguf::GGUFile;
use leafcutter::profiles::{render_prompt, resolve_profile};
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
    /// List available GGUF models (auto-detects ./models or ~/Downloads/models)
    List,
    /// Start a streaming chat session with a model (Ollama-style, native engine)
    Run {
        /// Model name (fuzzy match) or direct path to .gguf file
        model: String,
        /// Sampling temperature (0.0 = greedy)
        #[arg(short, long, default_value_t = 0.7)]
        temp: f32,
        /// Top-p sampling
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,
        /// Max tokens per response
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// Engine backend: "native" (default) or "ollama" (HTTP API)
        #[arg(long, default_value = "native")]
        engine: String,
        /// Ollama host (when --engine ollama is selected)
        #[arg(long, default_value = "http://127.0.0.1:11434")]
        ollama_host: String,
    },
    /// Start the HTTP API server (OpenAI-compatible, for Cynapse/Hermes/OpenCode integration)
    Serve {
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
    /// Start the HTTP API server (alias for 'serve', requires llama-ffi)
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
        Some(Commands::List) => {
            cmd_list_auto();
        }
        Some(Commands::Run { model, temp, top_p, max_tokens, engine, ollama_host }) => {
            if engine == "ollama" {
                cmd_run_ollama(&model, temp, top_p, max_tokens, &ollama_host);
            } else {
                cmd_run(&model, temp, top_p, max_tokens);
            }
        }
        Some(Commands::Serve { model, port, host, engine, benchmark }) => {
            cmd_serve(&model, port, &host, &engine, benchmark).await;
        }
        Some(Commands::Server { model, port, host, engine, benchmark }) => {
            cmd_serve(&model, port, &host, &engine, benchmark).await;
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
// Friendly commands — work without FFI (native engine only)
// ═════════════════════════════════════════════════════════════════════════════

/// Resolve the models directory: LEAF_MODELS_DIR env, then ./models, then ~/Downloads/models
fn resolve_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LEAF_MODELS_DIR") {
        let expanded = shellexpand::tilde(&dir);
        return PathBuf::from(expanded.as_ref());
    }
    let candidates = ["./models", "~/Downloads/models"];
    for c in &candidates {
        let expanded = shellexpand::tilde(c);
        let p = PathBuf::from(expanded.as_ref());
        if p.exists() {
            return p;
        }
    }
    // Default to ./models even if it doesn't exist (for error message)
    PathBuf::from("./models")
}

/// Scan models dir for .gguf files, return (path, size) pairs sorted by name
fn scan_models(dir: &PathBuf) -> Vec<(PathBuf, u64)> {
    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                if let Ok(meta) = std::fs::metadata(&path) {
                    models.push((path, meta.len()));
                }
            }
        }
    }
    models.sort_by(|a, b| a.0.file_name().cmp(&b.0.file_name()));
    models
}

/// Find a model by fuzzy name match (case-insensitive substring)
fn find_model(name: &str) -> Option<PathBuf> {
    // Direct path
    let p = PathBuf::from(shellexpand::tilde(name).as_ref());
    if p.exists() && p.extension().and_then(|s| s.to_str()) == Some("gguf") {
        return Some(p);
    }
    // Scan models dir
    let dir = resolve_models_dir();
    let models = scan_models(&dir);

    // Index number (e.g. "0", "1", "2")
    if let Ok(idx) = name.parse::<usize>() {
        if idx < models.len() {
            return Some(models[idx].0.clone());
        }
    }

    let needle = name.to_lowercase();
    // Exact match first
    for (path, _) in &models {
        if path.file_name().unwrap().to_str().unwrap().to_lowercase() == needle {
            return Some(path.clone());
        }
    }
    // Substring match
    for (path, _) in &models {
        let fname = path.file_name().unwrap().to_str().unwrap().to_lowercase();
        if fname.contains(&needle) {
            return Some(path.clone());
        }
    }
    None
}

fn cmd_list_auto() {
    let dir = resolve_models_dir();
    eprintln!("Leaves in: {}", dir.display());
    eprintln!();
    let models = scan_models(&dir);
    if models.is_empty() {
        eprintln!("No .gguf models found.");
        eprintln!("Download a model and place it in: {}", dir.display());
        eprintln!("Or set LEAF_MODELS_DIR=/path/to/models");
        return;
    }
    for (i, (path, size)) in models.iter().enumerate() {
        let name = path.file_name().unwrap().to_string_lossy();
        println!("  [{}] {:<50} {}", i, name, format_size(*size));
    }
    eprintln!();
    eprintln!("Run: leafcutter run <name>");
}

fn cmd_run(model_arg: &str, mut temp: f32, top_p: f32, max_tokens: usize) {
    use leafcutter::inference::engine::Engine;
    use leafcutter::tokenizer::chat_template::apply_chat_template_from_gguf;
    use leafcutter::model::gguf::GGUFile;

    // Resolve model
    let path = match find_model(model_arg) {
        Some(p) => p,
        None => {
            eprintln!("Model '{}' not found.", model_arg);
            eprintln!();
            cmd_list_auto();
            std::process::exit(1);
        }
    };
    let path_str = path.to_string_lossy().to_string();

    let mut engine = match Engine::load(&path_str) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    let info = engine.info();
    eprintln!("Leaf: {}", path.file_name().unwrap().to_string_lossy());
    eprintln!("Arch: {}  Layers: {}  Hidden: {}", info.architecture, info.total_layers, info.hidden_size);
    eprintln!("Temp: {:.1}  Max tokens: {}", temp, max_tokens);

    // Resolve the per-architecture profile (Ollama Modelfile-style) and
    // announce it so the user can see which template + defaults are
    // being used.
    let gguf_for_profile = GGUFile::open(&path_str).ok();
    let profile = resolve_profile(
        &gguf_for_profile.as_ref().map(|f| &f.metadata).cloned().unwrap_or_default(),
        None,
    );
    eprintln!(
        "Profile: {} ({})",
        profile.name, profile.description
    );
    eprintln!();
    eprintln!("Type /bye to exit, /clear to reset context, /help for commands.");
    eprintln!("─────────────────────────────────────────────────");

    let gguf = GGUFile::open(&path_str).ok();
    let mut conversation: Vec<(String, String)> = Vec::new();

    // Start the background CPU/thermal/RSS safety monitor.
    // Pure advisory — never throttles or blocks execution.  Disable with
    // LEAFCUTTER_CPU_MONITOR=0.
    leafcutter::cpu_monitor::start();

    loop {
        print!("\n> ");
        io::stdout().flush().ok();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }

        // In-session commands
        if input.starts_with('/') {
            let parts: Vec<&str> = input.splitn(2, ' ').collect();
            match parts[0] {
                "/bye" | "/quit" | "/exit" => {
                    // Explicit cleanup: clear conversation, drop caches
                    conversation.clear();
                    engine.kv_cache.clear();
                    engine.ssm_cache.clear();
                    engine.deltanet_cache.clear();
                    engine.seq_offset = 0;
                    eprintln!("[cache flushed] Goodbye!");
                    break;
                }
                "/clear" => {
                    conversation.clear();
                    // Clear engine internal caches so no stale KV/SSM state remains
                    engine.kv_cache.clear();
                    engine.ssm_cache.clear();
                    engine.deltanet_cache.clear();
                    engine.seq_offset = 0;
                    eprintln!("[context cleared — cache flushed]");
                    continue;
                }
                "/help" => {
                    eprintln!();
                    eprintln!("Commands:");
                    eprintln!("  /bye        Exit");
                    eprintln!("  /clear      Clear conversation context");
                    eprintln!("  /temp <f>   Set temperature (current: {:.1})", temp);
                    eprintln!("  /help       Show this help");
                    eprintln!();
                    continue;
                }
                "/temp" => {
                    if parts.len() < 2 {
                        eprintln!("Usage: /temp <float>  (current: {:.1})", temp);
                    } else if let Ok(v) = parts[1].trim().parse::<f32>() {
                        temp = v;
                        eprintln!("[temperature set to {:.1}]", temp);
                    } else {
                        eprintln!("Invalid temperature: {}", parts[1]);
                    }
                    continue;
                }
                _ => {
                    eprintln!("Unknown command: {}  (try /help)", parts[0]);
                    continue;
                }
            }
        }

        // Build prompt from conversation history
        conversation.push(("user".into(), input.clone()));

        let system = ""; // empty → profile supplies its default
        let prompt_text = if let Some(ref file) = gguf {
            // Profile-based rendering: each architecture has its own
            // template family, default system prompt, sampling, and
            // stop tokens.  This is what Ollama does with Modelfiles.
            let profile = resolve_profile(&file.metadata, None);
            render_prompt(&profile, system, &input)
        } else {
            input.clone()
        };

        let tokens = engine.tokenize(&prompt_text, true);
        if tokens.is_empty() {
            eprintln!("[tokenization failed — no tokenizer available]");
            continue;
        }

        eprintln!();
        if std::env::var("LEAFCUTTER_DEBUG_PROMPT").is_ok() {
            eprintln!("[DEBUG prompt, {} tokens]\n{}\n[END prompt]",
                tokens.len(), prompt_text);
        }
        let _ = io::stdout().flush();

        let gen_start = std::time::Instant::now();
        let mut generated_text = String::new();
        // Build stop-token list from the active profile so reasoning models
        // (Ornith/Qwen3.5) don't get cut off by mid-stream `<|im_end|>`
        // markers.  We use `` (the reasoning start tag) as the
        // ONLY stop signal — that way generation continues through the
        // full thinking block, the `` close, AND the response body,
        // and only halts when the model emits the final response terminator.
        let stop_token_ids: Vec<usize> = profile
            .stop_tokens
            .iter()
            .map(|s| s.0)
            .collect();
        let mut in_thinking = profile.opens_with_thinking;
        let mut thinking_done = false;
        let debug_chunks = std::env::var("LEAFCUTTER_CHUNK_DEBUG").is_ok();
        let generated_ids = engine.generate_streaming_with_stops(
            &tokens,
            max_tokens,
            temp,
            top_p,
            &stop_token_ids,
            |id, chunk| {
                if debug_chunks {
                    eprintln!("[chunk] id={} surface={:?} in_thinking={} thinking_done={}", id, chunk, in_thinking, thinking_done);
                }
                // Ornith thinking-stream protocol (matching Ollama):
                //   id=248068  →  (opens thinking block — emitted by model)
                //   id=248069  →  (closes thinking block — emitted by model)
                // Both are control tokens; never print them as text.
                match id {
                    248068 => { in_thinking = true; return true; }   // swallow opener
                    248069 => {                                       // swallow closer
                        in_thinking = false;
                        thinking_done = true;
                        let _ = io::stdout().flush();
                        return true;
                    }
                    _ => {}
                }
                if in_thinking && !thinking_done {
                    eprint!("💭{}", chunk);
                    let _ = io::stdout().flush();
                    return true;
                }
                eprint!("{}", chunk);
                let _ = io::stdout().flush();
                generated_text.push_str(chunk);
                true
            },
        );
        let gen_elapsed = gen_start.elapsed();
        let gen_tokens = generated_ids.len();
        let tok_per_sec = if gen_tokens > 0 && gen_elapsed.as_secs_f64() > 0.0 {
            gen_tokens as f64 / gen_elapsed.as_secs_f64()
        } else { 0.0 };
        eprintln!();

        // Get peak RSS from /proc/self/status (Linux)
        let peak_rss = get_peak_rss_mb();

        // Stats line
        eprintln!("─────────────────────────────────────────────────");
        eprintln!("Model: {} | Tokens: {} | Time: {:.2}s | Speed: {:.2} tok/s | RAM: {}",
            path.file_name().unwrap().to_string_lossy(),
            gen_tokens,
            gen_elapsed.as_secs_f64(),
            tok_per_sec,
            format_rss(peak_rss),
        );
        eprintln!();
        conversation.push(("assistant".into(), generated_text));
    }
}

/// Streaming chat via Ollama's HTTP API (`ollama serve`).
///
/// This is the "ground truth" backend: routes inference through Ollama
/// directly, producing IDENTICAL output to `ollama run`.  Used to verify
/// whether divergence from native Leafcutter output is due to forward-
/// pass bugs or chat-template/sampling drift.
fn cmd_run_ollama(
    model_arg: &str,
    temp: f32,
    top_p: f32,
    max_tokens: usize,
    host: &str,
) {
    use leafcutter::ollama_backend::{ChatMessage, OllamaClient};
    use std::io::{self, Write};

    let model_name = model_arg.to_string();
    let client = OllamaClient::new(host.to_string(), model_name.clone());

    eprintln!(
        "🌿 Leafcutter via Ollama (HTTP)\n   Model: {}\n   Host:  {}\n   Temp:  {:.2}  Top-p: {:.2}  Max tokens: {}\n─────────────────────────────────────────────────",
        model_name,
        host,
        temp,
        top_p,
        max_tokens
    );

    let mut conversation: Vec<ChatMessage> = Vec::new();

    loop {
        print!("\n> ");
        io::stdout().flush().ok();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();
        if input == "/bye" {
            break;
        }
        if input == "/clear" {
            conversation.clear();
            println!("[cache flushed]");
            continue;
        }
        if input.is_empty() {
            continue;
        }
        conversation.push(ChatMessage {
            role: "user".into(),
            content: input.into(),
            thinking: None,
        });

        let stop: Vec<String> = vec!["<|im_end|>".into()];

        let gen_start = std::time::Instant::now();
        let mut token_count = 0usize;
        let mut saw_thinking = false;
        let mut thinking_done = false;
        let result = client.chat_streaming(
            &conversation,
            temp,
            top_p,
            20,
            max_tokens as i32,
            &stop,
            |content, thinking| {
                // Stream the thinking block first, then the response.
                // Both go to stdout so they interleave naturally.
                if let Some(t) = thinking {
                    if !saw_thinking && !t.is_empty() {
                        print!("\n💭 ");
                        saw_thinking = true;
                    }
                    if !t.is_empty() {
                        print!("{}", t);
                        let _ = io::stdout().flush();
                    }
                }
                if !content.is_empty() {
                    if saw_thinking && !thinking_done {
                        println!();
                        println!();
                        thinking_done = true;
                    }
                    print!("{}", content);
                    let _ = io::stdout().flush();
                    token_count += content.len();
                }
                true
            },
        );
        if let Err(e) = result {
            eprintln!("\n[ollama error] {}", e);
            break;
        }
        let elapsed = gen_start.elapsed().as_secs_f64();
        eprintln!();
        eprintln!("─────────────────────────────────────────────────");
        eprintln!(
            "Model: {} | Tokens: {} chars | Time: {:.2}s",
            model_name, token_count, elapsed
        );
        eprintln!();
    }
}

async fn cmd_serve(model_path: &str, port: u16, host: &str, engine_type: &str, benchmark: bool) {
    // If no model specified, try to auto-detect the largest one
    let model_path = if model_path.is_empty() {
        let dir = resolve_models_dir();
        let models = scan_models(&dir);
        if models.is_empty() {
            eprintln!("No model specified and no .gguf models found in {}", dir.display());
            eprintln!("Usage: leafcutter serve --model <path>");
            std::process::exit(1);
        }
        let (largest, _) = models.iter().max_by_key(|(_, s)| s).unwrap();
        eprintln!("[auto-selected: {}]", largest.display());
        largest.to_string_lossy().to_string()
    } else {
        model_path.to_string()
    };

    #[cfg(feature = "llama-ffi")]
    {
        run_server_ffi(&model_path, port, host, engine_type, benchmark).await;
    }
    #[cfg(not(feature = "llama-ffi"))]
    {
        // Native-only serve: limited to native-streaming engine
        if engine_type != "native-streaming" {
            eprintln!("Engine '{}' requires llama-ffi. Building with native-streaming instead.", engine_type);
        }
        run_server_native(&model_path, port, host).await;
    }
}

#[cfg(not(feature = "llama-ffi"))]
async fn run_server_native(model_path: &str, port: u16, host: &str) {
    use leafcutter::api::{NativeStreamingEngine, LeafcutterEngine, run_server};
    use std::sync::Arc;

    eprintln!("LeafcutterLLM (Serve Mode: native-streaming, no FFI)");
    eprintln!("   Model: {}", model_path);
    eprintln!("   Host: {}:{}", host, port);
    eprintln!();

    let engine: Arc<dyn LeafcutterEngine> = match NativeStreamingEngine::load(model_path) {
        Ok(e) => {
            eprintln!("Native Streaming Engine loaded (low RAM mode)");
            Arc::new(e)
        }
        Err(e) => {
            eprintln!("Failed to load engine: {}", e);
            std::process::exit(1);
        }
    };

    run_server(engine, port, host).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// llama-ffi enabled paths
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "llama-ffi")]
async fn run_server_ffi(model_path: &str, port: u16, host: &str, engine_type: &str, benchmark: bool) {
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

/// Get peak RSS in bytes from /proc/self/status (Linux only).
/// Returns 0 on non-Linux or if /proc is unavailable.
fn get_peak_rss_bytes() -> u64 {
    // VmHWM is the peak resident set size ("high water mark")
    if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
        for line in contents.lines() {
            if line.starts_with("VmHWM:") {
                // Format: "VmHWM:\t   123 KB"
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<u64>() {
                        return kb * 1024;
                    }
                }
            }
        }
    }
    0
}

/// Get peak RSS, return as MB for display
fn get_peak_rss_mb() -> u64 {
    get_peak_rss_bytes() / (1024 * 1024)
}

/// Format RSS for the stats line — shows "123 MB" or "1.2 GB"
fn format_rss(mb: u64) -> String {
    if mb >= 1024 {
        format!("{:.1} GB", mb as f64 / 1024.0)
    } else {
        format!("{} MB", mb)
    }
}
