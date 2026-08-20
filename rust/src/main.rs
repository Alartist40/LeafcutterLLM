//! LeafcutterLLM v0.9.0 — Unified CLI
//!
//! Friendly commands (no FFI needed, native engine):
//!   leafcutter list                      List available models (auto-detects dirs)
//!   leafcutter source add <dir>          Point at a folder of models (persisted)
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
use colored::Colorize;
use std::io::{self, Write};
use std::path::PathBuf;

// ── Leafcutter palette — Gold & Purple (mirrors Paraclea) ─────────────────────
/// Bright gold — user prompts, values, active state.  RGB(255, 215, 0)
fn gold(s: &str) -> colored::ColoredString   { s.truecolor(255, 215, 0).bold() }
/// Vivid purple — borders, system labels, separators.  RGB(177, 74, 237)
fn purple(s: &str) -> colored::ColoredString { s.truecolor(177, 74, 237).bold() }
/// Dim purple — secondary info, non-bold separators.   RGB(177, 74, 237)
fn dim_purple(s: &str) -> colored::ColoredString { s.truecolor(177, 74, 237) }


#[cfg(feature = "llama-ffi")]
use std::sync::Arc;

#[cfg(feature = "llama-ffi")]
use leafcutter::llama_ffi::{backend_init, backend_free, LlamaModel, LlamaContext};

#[cfg(feature = "llama-ffi")]
use leafcutter::api::FfiEngine;

use leafcutter::model::gguf::GGUFile;
use leafcutter::profiles::{render_chat_prompt, render_prompt, resolve_profile};
use leafcutter::tokenizer::chat_template::apply_chat_template_from_gguf;

#[derive(Parser)]
#[command(name = "leafcutter")]
#[command(about = "LeafcutterLLM — Run LLMs locally, fast and light")]
#[command(version = "0.9.0")]
#[command(disable_help_subcommand = true)]
struct Cli {
    /// Print Cynapse synapse metadata JSON and exit
    #[arg(long, global = true)]
    meta: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// List available GGUF models (auto-detects ./models, ~/Downloads/models, and /source dirs)
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
        /// Engine backend: "auto" (default — detect format & hardware),
        /// "native" (GGUF Rust engine), "safetensor" (Python reference),
        /// or "ollama" (HTTP API)
        #[arg(long, default_value = "auto")]
        engine: String,
        /// Ollama host (when --engine ollama is selected)
        #[arg(long, default_value = "http://127.0.0.1:11434")]
        ollama_host: String,
    },
    /// Start the HTTP API server (OpenAI-compatible, for Cynapse integration)
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
        /// Teacher-forcing oracle gate (kimi-k3-in-c idea): prefill the
        /// prompt, then feed a known reference continuation token-by-token
        /// and report how often the model's top-1 prediction matches the
        /// reference. A cheap sanity check that the engine + quant are
        /// producing sane logits (useful as a gate before big runs).
        #[arg(long, default_value_t = false)]
        tf_check: bool,
        /// Max tokens to generate
        #[arg(short = 'n', long, default_value = "128")]
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
    /// Manage the model source directories (Ollama-style `/source`)
    Source {
        /// Operation: add <dir>, remove <dir>, or list
        #[command(subcommand)]
        op: SourceOp,
    },
    /// Launch the launcher menu or a specific app (Ollama-style)
    Launch {
        /// App to launch: cynapse or a registered app.
        /// Omit to show the launcher menu.
        app: Option<String>,
        /// Model to use (defaults to the app's last used model)
        #[arg(long)]
        model: Option<String>,
        /// Configure the app without launching it
        #[arg(long)]
        config: bool,
        /// Restore an app config to its pre-launch state
        #[arg(long)]
        restore: bool,
        /// Automatically answer yes to prompts
        #[arg(short, long)]
        yes: bool,
        /// Extra arguments passed to the app after `--`
        #[arg(last = true)]
        passthrough: Vec<String>,
    },
    /// Manage the app registry used by `leafcutter launch`
    App {
        /// Operation: add <name>, remove <name>, or list
        #[command(subcommand)]
        op: AppOp,
    },
    /// Show the command list (alias: 'leaf help')
    Help,
    /// Update leafcutter to the latest release from GitHub
    /// (downloads the prebuilt binary; falls back to a source rebuild if no
    /// prebuilt exists yet for your OS/CPU)
    Update {
        /// Rebuild from source even when a prebuilt binary is available
        #[arg(long)]
        from_source: bool,
    },
}

#[derive(clap::Subcommand)]
enum AppOp {
    /// Register an app for `leafcutter launch`
    Add {
        /// App name (e.g. cynapse)
        name: String,
        /// Command / executable to run
        #[arg(long)]
        command: String,
        /// Start a leafcutter server first
        #[arg(long)]
        needs_server: bool,
    },
    /// Remove a previously registered app
    Remove { name: String },
    /// List registered apps (built-in + config)
    List,
}

#[derive(clap::Subcommand)]
enum SourceOp {
    /// Add a directory where models live (persists to ~/.config/leafcutter)
    Add { dir: String },
    /// Remove a previously added source directory
    Remove { dir: String },
    /// List configured source directories
    List,
}

#[tokio::main]
async fn main() {
    // Cap rayon's global pool BEFORE any par_iter() runs (defaults to ALL
    // logical CPUs → pegs every core and spins up laptop fans during a
    // single inference workload). Default = physical cores − 1.
    leafcutter::init::configure_thread_pool(None);
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
            } else if engine == "safetensor" || engine == "safetensors" {
                cmd_run_safetensor(&model, temp, top_p, max_tokens);
            } else if engine == "native" || engine == "leafcutter" {
                // Native = the fast Rust `Engine` (GGUF prefill + KV/layer
                // caches + streaming). This is the Ollama-style REPL.
                cmd_run(&model, temp, top_p, max_tokens);
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
        Some(Commands::Generate { model, prompt, system, raw, tf_check, max_tokens, temperature, threads, ctx_size, gpu_layers }) => {
            #[cfg(feature = "llama-ffi")]
            cmd_generate(&model, &prompt, &system, raw, max_tokens, temperature, threads, ctx_size, gpu_layers, tf_check);
            #[cfg(not(feature = "llama-ffi"))]
            cmd_generate_native(&model, &prompt, &system, raw, max_tokens, temperature, tf_check);
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
        Some(Commands::Source { op }) => {
            cmd_source(op);
        }
        Some(Commands::Launch {
            app,
            model,
            config,
            restore,
            yes,
            passthrough,
        }) => {
            cmd_launch(app.as_deref(), model.as_deref(), config, restore, yes, &passthrough);
        }
        Some(Commands::App { op }) => {
            cmd_app(op);
        }
        Some(Commands::Help) => {
            cmd_help();
        }
        Some(Commands::Update { from_source }) => {
            cmd_update(from_source);
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

/// Resolve all models directories: LEAF_MODELS_DIR env, then config-file
/// `/source` dirs, then the defaults. cwd-independent for `leafcutter`
/// installed on PATH (default lives in ~/.local/share/leafcutter/models).
fn resolve_models_dirs() -> Vec<PathBuf> {
    leafcutter::config::model_dirs()
}

/// Scan every models dir for .gguf files and safetensors model dirs,
/// returning (path, size) pairs sorted by name.
fn scan_models(dirs: &[PathBuf]) -> Vec<(PathBuf, u64)> {
    let mut models = Vec::new();
    for dir in dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let is_gguf = path.extension().and_then(|s| s.to_str()) == Some("gguf");
                let is_st_dir =
                    path.is_dir() && leafcutter::detect::looks_like_safetensors_dir(&path);
                if is_gguf || is_st_dir {
                    let size = if is_st_dir {
                        leafcutter::detect::model_dir_size(&path)
                    } else {
                        std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
                    };
                    models.push((path, size));
                }
            }
        }
    }
    models.sort_by(|a, b| a.0.file_name().cmp(&b.0.file_name()));
    models
}

/// Streaming chat via the safetensor subprocess backend.
///
/// Spawns `scripts/leafcutter_safetensor_run.py` (HuggingFace
/// transformers + safetensors).  This is the reference-correct path
/// for hybrid models like Qwen3.5 / Ornith while the native GGUF
/// engine is being debugged.  CPU-only by default; users with a
/// CUDA build of torch get GPU acceleration automatically (set
/// CUDA_VISIBLE_DEVICES if needed).
fn cmd_run_safetensor(model_arg: &str, mut temp: f32, _top_p: f32, mut max_tokens: usize) {
    use leafcutter::model::gguf::GGUFile;
    use leafcutter::profiles::{render_chat_prompt, resolve_profile};

    // Resolve the model directory (accept either a path or a fuzzy name).
    let path = std::path::PathBuf::from(model_arg);
    let path = if path.exists() {
        path
    } else {
        // Try common dirs before giving up.
        let home = std::env::var("HOME").unwrap_or_default();
        let candidates = [
            format!("{}/Downloads/models", home),
            format!("{}/Documents/portfolio/LeafcutterLLM", home),
        ];
        let mut found = None;
        for dir in &candidates {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for entry in rd.flatten() {
                    let p = entry.path();
                    if p.is_dir()
                        && p.file_name()
                            .map(|n| n.to_string_lossy().to_lowercase()
                                .contains(&model_arg.to_lowercase()))
                            .unwrap_or(false)
                    {
                        found = Some(p);
                        break;
                    }
                }
            }
            if found.is_some() {
                break;
            }
        }
        match found {
            Some(p) => p,
            None => {
                eprintln!("Safetensor model directory '{}' not found.", model_arg);
                std::process::exit(1);
            }
        }
    };

    // Read metadata from config.json (best-effort; safetensors may not
    // have a GGUF-style .gguf file, but most do ship a config.json).
    let config_path = path.join("config.json");
    let profile = if config_path.exists() {
        match GGUFile::open(config_path.to_string_lossy().as_ref()) {
            Ok(_) => resolve_profile(&Default::default(), None),
            Err(_) => resolve_profile(&Default::default(), None),
        }
    } else {
        resolve_profile(&Default::default(), None)
    };

    eprintln!(
        "🌿 Leafcutter via Safetensors (transformers)\n   Model: {}\n   Temp:  {:.2}  Max tokens: {}\n─────────────────────────────────────────────────",
        path.display(),
        temp,
        max_tokens
    );
    eprintln!("Profile: {} ({})", profile.name, profile.description);
    eprintln!();
    eprintln!("Type /bye to exit, /clear to reset context, /help for commands.");

    let model_dir = path.to_string_lossy().to_string();
    let mut conversation: Vec<(String, String)> = Vec::new();
    let mut system_prompt = String::new();
    let mut total_tokens = 0usize;
    let mut total_time = 0.0f64;
    let mut turn_count = 0usize;

    loop {
        print!("\n>>> ");
        std::io::stdout().flush().ok();
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }
        if input.starts_with('/') {
            let parts: Vec<&str> = input.splitn(2, ' ').collect();
            match parts[0] {
                "/bye" | "/quit" | "/exit" => {
                    eprintln!("Goodbye!");
                    break;
                }
                "/clear" => {
                    conversation.clear();
                    system_prompt.clear();
                    eprintln!("[context cleared]");
                    continue;
                }
                "/help" => {
                    eprintln!();
                    eprintln!("Available Commands:");
                    eprintln!("  /temp <f>         Set temperature");
                    eprintln!("  /set system <t>   Override system prompt");
                    eprintln!("  /clear            Clear conversation");
                    eprintln!("  /bye, /quit       Exit");
                    eprintln!("  /show stats       Rolling stats");
                    eprintln!();
                    continue;
                }
                "/temp" => {
                    if parts.len() < 2 {
                        eprintln!("temperature: {:.2}", temp);
                    } else if let Ok(v) = parts[1].trim().parse::<f32>() {
                        temp = v;
                        eprintln!("[temperature = {:.2}]", temp);
                    } else {
                        eprintln!("Invalid: {}", parts[1]);
                    }
                    continue;
                }
                "/set" => {
                    let rest = parts.get(1).copied().unwrap_or("").trim();
                    if rest.starts_with("system") {
                        let new_sys = rest.trim_start_matches("system").trim();
                        if new_sys == "default" || new_sys == "reset" || new_sys.is_empty() {
                            system_prompt.clear();
                            eprintln!("[system reset to profile default]");
                        } else {
                            system_prompt = new_sys.to_string();
                            eprintln!("[system prompt set: {}]", new_sys);
                        }
                    } else {
                        eprintln!("Usage: /set system <text>  or  /set system default");
                    }
                    continue;
                }
                "/show" => {
                    if parts.len() >= 2 && parts[1] == "stats" {
                        let avg_speed = if total_time > 0.0 {
                            total_tokens as f64 / total_time
                        } else { 0.0 };
                        eprintln!("  Turns: {}", turn_count);
                        eprintln!("  Tokens out: {}", total_tokens);
                        eprintln!("  Time: {:.2}s", total_time);
                        eprintln!("  Avg speed: {:.2} tok/s", avg_speed);
                    } else {
                        eprintln!("Usage: /show stats");
                    }
                    continue;
                }
                _ => {
                    eprintln!("Unknown: {}  (try /help)", parts[0]);
                    continue;
                }
            }
        }

        conversation.push(("user".into(), input.clone()));
        let prompt = render_chat_prompt(&profile, &system_prompt, &conversation);
        eprintln!();

        let gen_start = std::time::Instant::now();
        let stop = profile.stop_tokens.iter().map(|s| s.1.to_string()).collect::<Vec<_>>();
        let cb_start = std::time::Instant::now();
        let mut first_token_time: Option<std::time::Duration> = None;
        let result = leafcutter::safetensor_backend::stream(
            &model_dir,
            &prompt,
            max_tokens,
            temp,
            0.95,
            20,
            &stop,
            |text, in_thinking| {
                if first_token_time.is_none() {
                    first_token_time = Some(cb_start.elapsed());
                }
                if in_thinking {
                    eprint!("💭{}", text);
                } else {
                    eprint!("{}", text);
                }
                let _ = std::io::stdout().flush();
                true
            },
        );
        let gen_elapsed = gen_start.elapsed();
        eprintln!();
        match result {
            Ok(n) => {
                total_tokens += n;
                total_time += gen_elapsed.as_secs_f64();
                turn_count += 1;
                eprintln!("─────────────────────────────────────────────────");
                eprintln!("Turn {}: {} tokens in {:.1}s ({:.2} tok/s)",
                    turn_count, n, gen_elapsed.as_secs_f64(),
                    if gen_elapsed.as_secs_f64() > 0.0 { n as f64 / gen_elapsed.as_secs_f64() } else { 0.0 });
                eprintln!();
                conversation.push(("assistant".into(), String::new()));
            }
            Err(e) => {
                eprintln!("[safetensor backend error] {}", e);
                eprintln!("Hint: install Python deps:  pip install transformers torch safetensors");
                continue;
            }
        }
    }
}


/// Find a model by fuzzy name match (case-insensitive substring)
fn find_model(name: &str) -> Option<PathBuf> {
    // Direct path: a .gguf file, or an existing directory (safetensors model folder)
    let p = PathBuf::from(shellexpand::tilde(name).as_ref());
    let is_gguf = p.extension().and_then(|s| s.to_str()) == Some("gguf");
    if p.exists() && (is_gguf || p.is_dir()) {
        return Some(p);
    }
    // Scan models dirs
    let dirs = resolve_models_dirs();
    let models = scan_models(&dirs);

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
    let dirs = resolve_models_dirs();
    println!("{}", purple("╔══════════════════════════════════════════════════════╗"));
    println!("{}", gold("║  🌿 LeafcutterLLM — Available Models                 ║"));
    println!("{}", purple("╚══════════════════════════════════════════════════════╝"));
    println!("{}", dim_purple("  Searching in:"));
    for d in &dirs {
        println!("    {}", dim_purple(&format!("- {}", d.display())));
    }
    println!();
    let models = scan_models(&dirs);
    if models.is_empty() {
        println!("  {}", "No models found.".yellow());
        println!("  {}", dim_purple("Point the tool at your models with:"));
        println!("    {}", gold("leafcutter source add <dir>"));
        println!("  {}", dim_purple("Or set LEAF_MODELS_DIR=/path/to/models"));
        return;
    }
    for (i, (path, size)) in models.iter().enumerate() {
        let name = path.file_name().unwrap().to_string_lossy();
        let kind = if path.is_dir() { dim_purple("[safetensors]").to_string() } else { String::new() };
        println!("  {} {:<50} {} {}",
            gold(&format!("[{}]", i)),
            name.bold(),
            gold(&format_size(*size)),
            kind,
        );
    }
    println!();
    println!("  {} {}", dim_purple("Run:"), gold("leafcutter run <name>"));
}


fn cmd_source(op: SourceOp) {
    use leafcutter::config;
    match op {
        SourceOp::Add { dir } => {
            let expanded = shellexpand::tilde(&dir).as_ref().to_string();
            let p = PathBuf::from(&expanded);
            if !p.exists() || !p.is_dir() {
                eprintln!("Source dir does not exist (or is not a directory): {}", expanded);
                std::process::exit(1);
            }
            match config::add_model_dir(&expanded) {
                added if added => {
                    eprintln!("Added source: {}", expanded);
                    eprintln!("Run `leafcutter list` to see models from there.");
                }
                _ => eprintln!("Already in sources: {}", expanded),
            }
        }
        SourceOp::Remove { dir } => {
            let expanded = shellexpand::tilde(&dir).as_ref().to_string();
            match config::remove_model_dir(&expanded) {
                removed if removed => {
                    eprintln!("Removed source: {}", expanded);
                }
                _ => eprintln!("Not in sources: {}", expanded),
            }
        }
        SourceOp::List => {
            let dirs = resolve_models_dirs();
            if dirs.is_empty() {
                eprintln!("No source directories configured.");
                return;
            }
            for (i, d) in dirs.iter().enumerate() {
                println!("  [{}] {}", i, d.display());
            }
        }
    }
}

fn cmd_launch(app: Option<&str>, model: Option<&str>, config_only: bool, restore: bool, yes: bool, passthrough: &[String]) {
    use leafcutter::launch::{self, LaunchRequest};

    // `leafcutter launch` with no app → launcher menu.
    let app = match app {
        Some(a) => a.to_string(),
        None => {
            cmd_launch_menu();
            return;
        }
    };

    let req = LaunchRequest {
        app: app.clone(),
        model_override: model.map(|s| s.to_string()),
        force_configure: false,
        configure_only: config_only,
        restore,
        yes,
        extra_args: pasthru(passthrough),
    };

    let models = scan_models(&resolve_models_dirs())
        .into_iter()
        .map(|(p, _)| p.to_string_lossy().into_owned())
        .collect::<Vec<_>>();

    if let Err(e) = launch::launch(&req, &models) {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}

/// Convert `-- args...` (post-`--`) into the passthrough list.
fn pasthru(args: &[String]) -> Vec<String> {
    args.to_vec()
}

/// The no-arg launcher menu: list apps + the current model.
fn cmd_launch_menu() {
    use leafcutter::config;
    eprintln!("🌿 leafcutter launcher");
    eprintln!();
    eprintln!("Apps:");
    let mut names: Vec<String> = config::load()
        .apps
        .keys()
        .cloned()
        .chain(config::builtin_apps().into_iter().map(|(n, _)| n))
        .collect();
    names.sort();
    names.dedup();
    for n in names {
        let entry = config::resolve_app(&n).unwrap();
        let server = if entry.needs_server { " (server)" } else { "" };
        let model = config::app_model(&n).unwrap_or_else(|| "-".into());
        eprintln!("  - {:<12} model: {}{}", n, model, server);
    }
    eprintln!();
    eprintln!("Run: leafcutter launch <app> [--model <model>]");
    eprintln!("     leafcutter launch <app> --config   (configure only)");
    eprintln!("     leafcutter launch <app> --restore  (undo launch config)");
}

fn cmd_help() {
    println!("🌿 LeafcutterLLM — Run LLMs locally, fast and light");
    println!();
    println!("Usage: leafcutter <command> [options]     (alias: leaf)");
    println!();
    println!("Commands:");
    println!("  list                 List available models (auto-detects model dirs)");
    println!("  run <model>          Start a streaming chat session (Ollama-style)");
    println!("  generate --model     One-shot text generation from a prompt");
    println!("  chat --model         Interactive chat session (requires llama-ffi build)");
    println!("  serve --model        Start the HTTP API server (OpenAI-compatible)");
    println!("  server               Alias for 'serve'");
    println!("  list-models          List models (legacy flag form, --dir)");
    println!("  source <op>          Manage model source directories (add/remove/list)");
    println!("  launch [app]         Launcher menu or start an app (e.g. cynapse)");
    println!("  app <op>             Manage the app registry used by 'launch'");
    println!("  update               Update leafcutter to the latest GitHub release");
    println!("  help                 Show this command list");
    println!();
    println!("Options:");
    println!("  -h, --help           Print help");
    println!("  -V, --version        Print version");
    println!();
    println!("Examples:");
    println!("  leafcutter list");
    println!("  leafcutter run ornith");
    println!("  leafcutter launch cynapse --model ornith");
    println!("  leafcutter serve --model /path/to/model.gguf --port 8081");
    println!();
    println!("For Cynapse integration:  leafcutter --meta   (synapse metadata JSON)");
}

/// Detect the GitHub release asset name for the current OS/CPU, e.g.
/// `leafcutter-linux-x86_64`. Returns None for platforms we don't publish
/// prebuilt binaries for (e.g. some BSDs) so the caller can fall back to a
/// source rebuild.
fn release_asset_name() -> Option<String> {
    let os = match std::env::consts::OS {
        "linux" => "linux",
        "macos" => "macos",
        "windows" => "windows",
        _ => return None,
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x86_64",
        "aarch64" => "aarch64",
        _ => return None,
    };
    Some(format!("leafcutter-{}-{}{}", os, arch, if os == "windows" { ".exe" } else { "" }))
}

/// `leafcutter update` — self-update from the GitHub release feed.
///
/// Tries to download the prebuilt binary for the current OS/CPU first. If no
/// prebuilt exists (new/unknown platform) or `--from-source` is passed, it
/// falls back to a source rebuild: clone/pull LeafcutterLLM into
/// `~/.leafcutter` (the same location `install.sh` uses) and rebuild with
/// cargo. The current executable is then replaced in place.
fn cmd_update(from_source: bool) {
    let repo = "Alartist40/LeafcutterLLM";
    let cur_exe = std::env::current_exe().unwrap_or_else(|_| {
        eprintln!("❌ Could not determine the current executable path.");
        std::process::exit(1);
    });

    if from_source {
        update_from_source(repo, &cur_exe);
        return;
    }

    let Some(asset) = release_asset_name() else {
        eprintln!(
            "ℹ️  No prebuilt binary for {}/{} — falling back to a source rebuild.",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
        update_from_source(repo, &cur_exe);
        return;
    };

    let latest_url = format!(
        "https://api.github.com/repos/{}/releases/latest",
        repo
    );
    let dl_url = format!(
        "https://github.com/{}/releases/latest/download/{}",
        repo, asset
    );

    // Probe for a prebuilt binary (release asset). A redirect/200 means the
    // asset exists for this platform; 404 means it doesn't yet.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(15))
        .timeout_read(std::time::Duration::from_secs(15))
        .user_agent("leafcutter-update/0.9.0")
        .build();

    println!("🌿 Checking for updates from {}", repo);
    let latest_tag = agent
        .get(&latest_url)
        .call()
        .ok()
        .and_then(|r| {
            let body = r.into_string().ok()?;
            serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v.get("tag_name").and_then(|t| t.as_str()).map(String::from))
        });

    match latest_tag {
        Some(tag) => println!("   latest release: {}", tag),
        None => println!("   (could not read latest release tag)"),
    }

    let probe = agent.head(&dl_url).call();
    let asset_exists = matches!(probe, Ok(_) | Err(ureq::Error::Status(302, _)));

    if !asset_exists {
        eprintln!("ℹ️  No prebuilt binary ({}) — falling back to a source rebuild.", asset);
        update_from_source(repo, &cur_exe);
        return;
    }
    println!("⬇️  Downloading {} …", asset);
    let resp = match agent.get(&dl_url).call() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("❌ Download failed: {e}");
            std::process::exit(1);
        }
    };
    let mut bytes = Vec::new();
    if let Err(e) = resp.into_reader().read_to_end(&mut bytes) {
        eprintln!("❌ Download failed: {e}");
        std::process::exit(1);
    }
    if bytes.is_empty() {
        eprintln!("❌ Downloaded file is empty.");
        std::process::exit(1);
    }

    // Best-effort sanity check that it's a real executable (ELF/Mach-O/PE).
    let looks_like_binary = bytes.len() >= 4
        && matches!(
            (&bytes[0], &bytes[1], &bytes[2], &bytes[3]),
            (0x7f, 0x45, 0x4c, 0x46)          // ELF
                | (0x4d, 0x5a, _, _)          // PE
                | (0xcf, 0xfa, 0xed, 0xfe)    // Mach-O (64-bit)
        );
    if !looks_like_binary {
        eprintln!("❌ Downloaded file does not look like a binary — aborting.");
        std::process::exit(1);
    }

    // Replace the running executable. Write to a temp file first, then rename
    // over the target (atomic on POSIX; works on Windows for a stopped exe).
    let exe_dir = cur_exe.parent().unwrap_or_else(|| std::path::Path::new("."));
    let tmp_path = exe_dir.join(format!(".leafcutter-update-{}", std::process::id()));
    if let Err(e) = std::fs::write(&tmp_path, &bytes) {
        eprintln!("❌ Could not write update to {}: {e}", tmp_path.display());
        std::process::exit(1);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o755));
    }
    if let Err(e) = std::fs::rename(&tmp_path, &cur_exe) {
        eprintln!("❌ Could not replace {}: {e}", cur_exe.display());
        let _ = std::fs::remove_file(&tmp_path);
        std::process::exit(1);
    }

    println!("✅ Updated to the latest release. Restart leafcutter to use it.");
}

/// Rebuild leafcutter from source into `~/.leafcutter` (same location
/// `install.sh` uses), then copy the fresh binary over the current one.
fn update_from_source(repo: &str, cur_exe: &std::path::Path) {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let install_dir = std::path::PathBuf::from(format!("{}/.leafcutter", home));
    let base_url = format!("https://github.com/{}.git", repo);

    println!("🛠  Rebuilding from source in {} …", install_dir.display());

    if install_dir.join(".git").exists() {
        println!("   pulling latest …");
        let ok = std::process::Command::new("git")
            .arg("-C")
            .arg(&install_dir)
            .arg("pull")
            .arg("--rebase")
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            eprintln!("❌ `git pull` failed in {}", install_dir.display());
            std::process::exit(1);
        }
    } else {
        println!("   cloning …");
        if !std::process::Command::new("git")
            .arg("clone")
            .arg("--depth")
            .arg("1")
            .arg(&base_url)
            .arg(&install_dir)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            eprintln!("❌ `git clone` failed. Is git installed and the network up?");
            std::process::exit(1);
        }
    }

    if !std::process::Command::new("cargo")
        .current_dir(install_dir.join("rust"))
        .args(["build", "--release", "--bin", "leafcutter"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        eprintln!("❌ `cargo build --release` failed. Is cargo installed?");
        std::process::exit(1);
    }

    let built = install_dir.join("rust/target/release/leafcutter");
    if let Err(e) = std::fs::copy(&built, cur_exe) {
        eprintln!("❌ Could not install built binary to {}: {e}", cur_exe.display());
        std::process::exit(1);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(cur_exe, std::fs::Permissions::from_mode(0o755));
    }
    println!("✅ Updated from source. Restart leafcutter to use it.");
}

fn cmd_app(op: AppOp) {
    use leafcutter::config;
    match op {
        AppOp::Add {
            name,
            command,
            needs_server,
        } => {
            let entry = config::AppEntry {
                command: command.clone(),
                args: Vec::new(),
                needs_server,
                env: std::collections::HashMap::new(),
                model: None,
                model_fp: None,
            };
            if config::add_app(&name, entry) {
                eprintln!("Registered app '{}' (command: {})", name, command);
            } else {
                eprintln!("App '{}' already registered.", name);
            }
        }
        AppOp::Remove { name } => {
            if config::remove_app(&name) {
                eprintln!("Removed app '{}'.", name);
            } else {
                eprintln!("App '{}' not found.", name);
            }
        }
        AppOp::List => {
            eprintln!("Registered apps:");
            let mut names: Vec<String> = config::load()
                .apps
                .keys()
                .cloned()
                .chain(config::builtin_apps().into_iter().map(|(n, _)| n))
                .collect();
            names.sort();
            names.dedup();
            for n in names {
                let entry = config::resolve_app(&n).unwrap();
                let server = if entry.needs_server { " (server)" } else { "" };
                eprintln!("  - {}{}", n, server);
            }
        }
    }
}

fn cmd_run(model_arg: &str, mut temp: f32, mut top_p: f32, mut max_tokens: usize) {
    use leafcutter::inference::engine::Engine;
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

    // ── Colony dispatch ─────────────────────────────────────────────
    // A single `leafcutter run` opens a window to any model: safetensors
    // directories route to the reference Python backend; GGUF stays on the
    // native Rust engine (which self-tunes cache vs streaming by RAM).
    use leafcutter::detect::{choose_tier, probe_hardware, probe_model, ModelKind, Tier};
    let probe = probe_model(&path);
    match probe.kind {
        ModelKind::Safetensors => {
            eprintln!(
                "  🌿 [safetensors dir] → reference Python backend ({:.1} MB)",
                probe.size_mb()
            );
            cmd_run_safetensor(&path_str, temp, top_p, max_tokens);
            return;
        }
        ModelKind::Unknown => {
            eprintln!(
                "  ⚠️  Model format of '{}' is not recognised (not .gguf, no safetensors shards).",
                path.display()
            );
        }
        ModelKind::Gguf => {}
    }

    let mut engine = match Engine::load(&path_str) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    let info = engine.info();

    // ── Colony dispatch info: hardware + tier for the banner ────────
    let hw = probe_hardware();
    let prefer_gpu =
        std::env::var("LEAFCUTTER_PREFER_GPU").map(|v| v == "1").unwrap_or(false);
    let tier = choose_tier(hw.gpu, hw.ram_available_mb, probe.size_bytes, prefer_gpu);
    let _ = Tier::Gpu; // (Tier 1 reported when a GPU backend ships)

    // Resolve the per-architecture profile (Ollama Modelfile-style).
    let gguf_for_profile = GGUFile::open(&path_str).ok();
    let profile = resolve_profile(
        &gguf_for_profile.as_ref().map(|f| &f.metadata).cloned().unwrap_or_default(),
        None,
    );

    // If temp/top_p/max_tokens are still at CLI defaults, adopt profile defaults.
    // (Ollama behavior: Modelfile params override CLI unless CLI explicitly sets.)
    if temp == 0.7 {
        temp = profile.sampling.temperature;
    }
    if top_p == 0.9 {
        top_p = profile.sampling.top_p;
    }
    let mut system_prompt: String = String::new(); // empty → profile default
    let gguf = GGUFile::open(&path_str).ok();

    // --- Welcome banner (model card) ---
    let model_name = path.file_name().unwrap().to_string_lossy();
    let file_mb = path.metadata().map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
    let npu_suffix = if hw.npu.is_present() {
        format!(" · npu:{}", hw.npu.label())
    } else {
        String::new()
    };

    // Helper: print a key-value row inside the banner box.
    // Label is dim-purple, value is gold, padded to 34 chars with purple right border.
    macro_rules! banner_row {
        ($label:expr, $value:expr) => {{
            let val = truncate_str(&$value.to_string(), 34);
            let padded = format!("{:<34}", val);
            eprintln!("  {}  {}: {}{}",
                purple("║"),
                dim_purple($label),
                gold(&padded),
                purple("║"),
            );
        }};
    }

    eprintln!();
    eprintln!("  {}", purple("╔══════════════════════════════════════════════╗"));
    eprintln!("  {}", gold( "║  🌿 LeafcutterLLM — Native Engine            ║"));
    eprintln!("  {}", purple("╠══════════════════════════════════════════════╣"));
    banner_row!("Model   ", model_name);
    banner_row!("Arch    ", info.architecture);
    banner_row!("Layers  ", format!("{} layers, {} hidden", info.total_layers, info.hidden_size));
    banner_row!("Size    ", format!("{:.1} MB", file_mb));
    banner_row!("Hardware", format!("{} · {} cores · {:.0} GiB free{}", hw.os, hw.cpu_cores, hw.ram_available_mb as f64 / 1024.0, npu_suffix));
    banner_row!("Tier    ", format!("{} — {}", tier.number(), tier.label()));
    banner_row!("Profile ", profile.name);
    banner_row!("Temp    ", format!("{:.2}  (top_p={:.2})", temp, top_p));
    banner_row!("Max tok ", max_tokens);
    eprintln!("  {}", purple("╚══════════════════════════════════════════════╝"));
    eprintln!();
    eprintln!("  {} {}",
        dim_purple("Type"),
        gold("/help for commands · /bye to exit"),
    );
    eprintln!("  {}", dim_purple("─────────────────────────────────────────────────"));

    let mut conversation: Vec<(String, String)> = Vec::new();

    // Rolling stats for /stats command.
    let mut total_tokens: usize = 0;
    let mut total_time: f64 = 0.0;
    let mut turn_count: usize = 0;

    // Start the background CPU/thermal/RSS safety monitor.
    leafcutter::cpu_monitor::start();

    loop {
        // Gold ">>> " prompt — matches Paraclea's "You >" in gold.
        print!("\n{} ", gold(">>>"));
        io::stdout().flush().ok();


        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break, // EOF — stdin closed (piped input finished)
            Ok(_) => {}
            Err(_) => break,
        }
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }

        // --- In-session slash commands (Ollama-style) ---
        if input.starts_with('/') {
            let parts: Vec<&str> = input.splitn(3, ' ').collect();
            let cmd = parts[0];
            match cmd {
                "/bye" | "/quit" | "/exit" => {
                    conversation.clear();
                    engine.kv_cache.clear();
                    engine.ssm_cache.clear();
                    engine.deltanet_cache.clear();
                    engine.seq_offset = 0;
                    eprintln!("\n{} {}\n",
                        purple("Leafcutter >"),
                        gold("Goodbye! See you soon."),
                    );
                    break;
                }
                "/clear" => {
                    conversation.clear();
                    engine.kv_cache.clear();
                    engine.ssm_cache.clear();
                    engine.deltanet_cache.clear();
                    engine.seq_offset = 0;
                    eprintln!("{}", dim_purple("[context cleared — caches flushed]"));
                    continue;
                }
                "/help" | "/?" => {
                    print_help(temp, top_p, max_tokens, &profile);
                    continue;
                }
                "/set" => {
                    // Ollama: /set parameter value
                    // Also: /set system <text> to override system prompt.
                    if parts.len() < 2 {
                        eprintln!("Usage: /set <temp|top_p|topk|max|system|repeat> <value>");
                        eprintln!("  /set temp 0.8        Set temperature");
                        eprintln!("  /set top_p 0.9       Set top-p");
                        eprintln!("  /set max 512         Set max tokens");
                        eprintln!("  /set system You are...  Set system prompt");
                        eprintln!("  /set system default   Reset to profile default");
                    } else {
                        let key = parts[1];
                        match key {
                            "temp" | "temperature" => {
                                if parts.len() < 3 {
                                    eprintln!("temperature: {:.2}", temp);
                                } else if let Ok(v) = parts[2].trim().parse::<f32>() {
                                    temp = v;
                                    eprintln!("[temperature = {:.2}]", temp);
                                } else {
                                    eprintln!("Invalid: {}", parts[2]);
                                }
                            }
                            "top_p" | "topp" => {
                                if parts.len() < 3 {
                                    eprintln!("top_p: {:.2}", top_p);
                                } else if let Ok(v) = parts[2].trim().parse::<f32>() {
                                    top_p = v;
                                    eprintln!("[top_p = {:.2}]", top_p);
                                } else {
                                    eprintln!("Invalid: {}", parts[2]);
                                }
                            }
                            "max" | "max_tokens" | "maxtokens" => {
                                if parts.len() < 3 {
                                    eprintln!("max_tokens: {}", max_tokens);
                                } else if let Ok(v) = parts[2].trim().parse::<usize>() {
                                    max_tokens = v;
                                    eprintln!("[max_tokens = {}]", max_tokens);
                                } else {
                                    eprintln!("Invalid: {}", parts[2]);
                                }
                            }
                            "system" => {
                                // "/set system" alone shows current; "/set system default" resets.
                                if parts.len() < 3 {
                                    if system_prompt.is_empty() {
                                        eprintln!("[system = profile default]");
                                    } else {
                                        eprintln!("[system] {}", system_prompt);
                                    }
                                } else if parts[2].trim() == "default" || parts[2].trim() == "reset" {
                                    system_prompt.clear();
                                    eprintln!("[system reset to profile default]");
                                } else {
                                    system_prompt = parts[2].trim().to_string();
                                    eprintln!("[system prompt set: {}]", truncate_str(&system_prompt, 60));
                                }
                            }
                            _ => {
                                eprintln!("Unknown /set key: {}  (temp, top_p, max, system)", key);
                            }
                        }
                    }
                    continue;
                }
                "/show" => {
                    // Ollama: /show info|profile|system|stats
                    if parts.len() < 2 {
                        eprintln!("Usage: /show <info|profile|system|history|stats>");
                    } else {
                        match parts[1] {
                            "info" => {
                                eprintln!("  Model:    {}", model_name);
                                eprintln!("  Arch:     {}", info.architecture);
                                eprintln!("  Layers:   {}", info.total_layers);
                                eprintln!("  Hidden:   {}", info.hidden_size);
                                eprintln!("  Size:     {:.1} MB", file_mb);
                                let peak = get_peak_rss_mb();
                                eprintln!("  Peak RAM: {}", format_rss(peak));
                            }
                            "profile" => {
                                eprintln!("  Profile:  {} ({})", profile.name, profile.description);
                                eprintln!("  Archs:    {:?}", profile.architectures);
                                eprintln!("  Temp:     {:.2}  Top_p: {:.2}  Top_k: {}  Repeat: {:.2}",
                                    profile.sampling.temperature, profile.sampling.top_p,
                                    profile.sampling.top_k, profile.sampling.repeat_penalty);
                                eprintln!("  Thinking: {}", profile.opens_with_thinking);
                                eprintln!("  Stop:     {:?}", profile.stop_tokens.iter().map(|s| s.1).collect::<Vec<_>>());
                            }
                            "system" => {
                                if system_prompt.is_empty() {
                                    eprintln!("[system = profile default]");
                                    eprintln!("  {}", profile.default_system);
                                } else {
                                    eprintln!("{}", system_prompt);
                                }
                            }
                            "history" => {
                                if conversation.is_empty() {
                                    eprintln!("[no conversation yet]");
                                } else {
                                    eprintln!("─ Conversation ({} turns) ─", conversation.len());
                                    for (i, (role, content)) in conversation.iter().enumerate() {
                                        let preview = truncate_str(content, 70);
                                        eprintln!("  {}. [{}] {}", i + 1, role, preview);
                                    }
                                }
                            }
                            "stats" => {
                                let avg_speed = if total_time > 0.0 {
                                    total_tokens as f64 / total_time
                                } else { 0.0 };
                                eprintln!("  Turns:         {}", turn_count);
                                eprintln!("  Total tokens:  {}", total_tokens);
                                eprintln!("  Total time:    {:.2}s", total_time);
                                eprintln!("  Avg speed:     {:.2} tok/s", avg_speed);
                                let peak = get_peak_rss_mb();
                                eprintln!("  Peak RAM:      {}", format_rss(peak));
                                eprintln!("  Conversation:  {} turns in context", conversation.len());
                            }
                            _ => {
                                eprintln!("Unknown /show target: {}  (info, profile, system, history, stats)", parts[1]);
                            }
                        }
                    }
                    continue;
                }
                "/temp" => {
                    // Short alias for /set temp.
                    if parts.len() < 2 {
                        eprintln!("temperature: {:.2}  /temp <f>", temp);
                    } else if let Ok(v) = parts[1].trim().parse::<f32>() {
                        temp = v;
                        eprintln!("[temperature = {:.2}]", temp);
                    } else {
                        eprintln!("Invalid: {}", parts[1]);
                    }
                    continue;
                }
                "/info" => {
                    eprintln!("  Model:    {}", model_name);
                    eprintln!("  Arch:     {}", info.architecture);
                    eprintln!("  Layers:   {}", info.total_layers);
                    eprintln!("  Hidden:   {}", info.hidden_size);
                    eprintln!("  Size:     {:.1} MB", file_mb);
                    let peak = get_peak_rss_mb();
                    eprintln!("  Peak RAM: {}", format_rss(peak));
                    continue;
                }
                "/stats" | "/usage" => {
                    let avg_speed = if total_time > 0.0 {
                        total_tokens as f64 / total_time
                    } else { 0.0 };
                    eprintln!("  Turns:       {}", turn_count);
                    eprintln!("  Tokens out:  {}", total_tokens);
                    eprintln!("  Time:        {:.2}s", total_time);
                    eprintln!("  Avg speed:   {:.2} tok/s", avg_speed);
                    let peak = get_peak_rss_mb();
                    eprintln!("  Peak RAM:    {}", format_rss(peak));
                    continue;
                }
                "/source" => {
                    // Manage model source directories (persisted in ~/.config/leafcutter).
                    use leafcutter::config;
                    match parts.get(1).copied() {
                        None | Some("list") => {
                            for d in resolve_models_dirs() {
                                eprintln!("  {}", d.display());
                            }
                        }
                        Some("add") => {
                            if let Some(dir) = parts.get(2) {
                                let expanded = shellexpand::tilde(dir).as_ref().to_string();
                                if config::add_model_dir(&expanded) {
                                    eprintln!("[added source: {}]", expanded);
                                } else {
                                    eprintln!("[already in sources: {}]", expanded);
                                }
                            } else {
                                eprintln!("Usage: /source add <dir>");
                            }
                        }
                        Some("remove") => {
                            if let Some(dir) = parts.get(2) {
                                let expanded = shellexpand::tilde(dir).as_ref().to_string();
                                if config::remove_model_dir(&expanded) {
                                    eprintln!("[removed source: {}]", expanded);
                                } else {
                                    eprintln!("[not in sources: {}]", expanded);
                                }
                            } else {
                                eprintln!("Usage: /source remove <dir>");
                            }
                        }
                        Some(other) => {
                            eprintln!("Unknown /source op: {}  (add <dir> | remove <dir> | <list>)", other);
                        }
                    }
                    continue;
                }
                _ => {
                    eprintln!("Unknown command: {}  (try /help)", cmd);
                    continue;
                }
            }
        }

        // --- Build multi-turn prompt from conversation history ---
        conversation.push(("user".into(), input.clone()));

        let prompt_text = if let Some(ref file) = gguf {
            // Always use the profile-based renderer (render_chat_prompt) for
            // multi-turn REPL. The GGUF's embedded Jinja chat_template was
            // designed for single-turn Ollama-style API calls and primes
            // reasoning models (e.g. Ornith) to think verbosely with
            // markdown bullets before answering, which floods stderr in the
            // REPL. The profile renderer emits an open ChatML/Llama3/
            // Ministral turn that matches Ollama's `run` behavior.
            let prof = resolve_profile(&file.metadata, None);
            render_chat_prompt(&prof, &system_prompt, &conversation)
        } else {
            input.clone()
        };

        let tokens = engine.tokenize(&prompt_text, true);
        if tokens.is_empty() {
            eprintln!("[tokenization failed — no tokenizer available]");
            // Roll back the conversation push.
            conversation.pop();
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
        let stop_token_ids: Vec<usize> = profile
            .stop_tokens
            .iter()
            .map(|s| s.0)
            .collect();
        let mut in_thinking = profile.opens_with_thinking;
        let mut thinking_tail = String::new();
        let debug_chunks = std::env::var("LEAFCUTTER_CHUNK_DEBUG").is_ok();
        let generated_ids = engine.generate_streaming_with_stops(
            &tokens,
            max_tokens,
            temp,
            top_p,
            &stop_token_ids,
            |id, chunk| {
                if debug_chunks {
                    eprintln!("[chunk] id={} surface={:?} in_thinking={}", id, chunk, in_thinking);
                }
                if in_thinking {
                    thinking_tail.push_str(chunk);
                    // Check for the full `</think>` marker BEFORE trimming
                    // (it is 8 chars; the trimmed tail is only 7).
                    if let Some(pos) = thinking_tail.find("</think>") {
                        let (pre, rest) = thinking_tail.split_at(pos);
                        if !pre.is_empty() {
                            eprint!("{}", dim_purple(pre));
                        }
                        thinking_tail = rest["</think>".len()..].to_string();
                        if std::env::var("LEAFCUTTER_CHUNK_DEBUG").is_ok() {
                            eprintln!("[gen-marker-set] thinking_tail now={:?}", thinking_tail);
                        }
                        in_thinking = false;
                        eprintln!();
                        let _ = io::stdout().flush();
                    } else {
                        // Emit all but the last 7 chars (they can't be part
                        // of a partial `</think>`), keeping the boundary.
                        let raw_keep = thinking_tail.len().saturating_sub(7);
                        let keep = floor_char_boundary(&thinking_tail, raw_keep);
                        if keep > 0 {
                            let (emit, rest) = thinking_tail.split_at(keep);
                            eprint!("{}", dim_purple(emit));
                            thinking_tail = rest.to_string();
                        }
                    }
                    let _ = io::stdout().flush();
                    return true;
                }
                if !thinking_tail.is_empty() {
                    eprint!("{}", dim_purple(&thinking_tail));
                    thinking_tail.clear();
                    eprintln!();
                }
                eprint!("{}", gold(chunk));
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

        // Rolling stats.
        total_tokens += gen_tokens;
        total_time += gen_elapsed.as_secs_f64();
        turn_count += 1;

        let peak_rss = get_peak_rss_mb();
        let cur_rss = get_current_rss_mb();
        eprintln!("{}", dim_purple("─────────────────────────────────────────────────"));
        eprintln!("{} {} {} {} {} {} {}",
            dim_purple(&truncate_str(&model_name, 28)),
            dim_purple("|"),
            gold(&format!("out={}", gen_tokens)),
            dim_purple("|"),
            gold(&format!("{:.2}s", gen_elapsed.as_secs_f64())),
            dim_purple("|"),
            gold(&format!("{:.2} tok/s  RAM {} (peak {})",
                tok_per_sec,
                format_rss(cur_rss),
                format_rss(peak_rss),
            )),
        );
        eprintln!();
        conversation.push(("assistant".into(), generated_text));
    }
}

/// Print the /help screen.
fn print_help(temp: f32, top_p: f32, max_tokens: usize, profile: &leafcutter::profiles::ModelProfile) {
    eprintln!();
    eprintln!("{}", gold("  Available Commands:"));
    eprintln!("  {}  {}", dim_purple("/set <key> <val>"), "Set parameter (temp, top_p, max, system)");
    eprintln!("  {}      {}", dim_purple("/show <target>"), "Show info, profile, system, history, stats");
    eprintln!("  {}         {}", dim_purple("/temp <f>"), "Set temperature (alias for /set temp)");
    eprintln!("  {}            {}", dim_purple("/info"), "Show loaded model info");
    eprintln!("  {}           {}", dim_purple("/stats"), "Show rolling session statistics");
    eprintln!("  {}           {}", dim_purple("/clear"), "Clear conversation + flush caches");
    eprintln!("  {}          {}", dim_purple("/source"), "List model source dirs (/source add <dir>)");
    eprintln!("  {}        {}", dim_purple("/help, /?"), "Show this help");
    eprintln!("  {}    {}", dim_purple("/bye, /quit"), "Exit");
    eprintln!();
    eprintln!("{}", gold("  Current Settings:"));
    eprintln!("  {}  {}", dim_purple("temperature :"), gold(&format!("{:.2}", temp)));
    eprintln!("  {}        {}", dim_purple("top_p :"), gold(&format!("{:.2}", top_p)));
    eprintln!("  {}   {}", dim_purple("max_tokens :"), gold(&format!("{}", max_tokens)));
    eprintln!("  {}      {} {}",
        dim_purple("profile :"),
        gold(profile.name),
        dim_purple(&format!("({})", profile.description)),
    );
    eprintln!();
}

/// Truncate a string to `max` chars, appending "..." if truncated.
fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else if max <= 3 {
        "...".to_string()
    } else {
        let cut = floor_char_boundary(s, max - 3);
        format!("{}...", &s[..cut])
    }
}

/// Return the largest `i <= at_or_before` that lands on a UTF-8 char boundary
/// in `s`.  Avoids panics from `split_at` / `&s[..i]` when `i` falls inside a
/// multi-byte character (e.g. emoji, accented Latin, CJK).
fn floor_char_boundary(s: &str, at_or_before: usize) -> usize {
    if at_or_before >= s.len() {
        return s.len();
    }
    let mut i = at_or_before;
    // Walk back until we find a non-continuation byte (UTF-8 continuation
    // bytes are 0b10xxxxxx = 0x80..0xC0).  We never need more than 3 steps
    // since the longest UTF-8 sequence is 4 bytes.
    while i > 0 && (s.as_bytes()[i] & 0xC0) == 0x80 {
        i -= 1;
    }
    i
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
        let dirs = resolve_models_dirs();
        let models = scan_models(&dirs);
        if models.is_empty() {
            eprintln!("No model specified and no .gguf models found.");
            eprintln!("Usage: leafcutter serve --model <path>");
            eprintln!("Or: leafcutter source add <dir>  to point at your models");
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

    eprintln!("{}", gold("🌿 LeafcutterLLM — Serve Mode: native-streaming"));
    eprintln!("   {}  {}", dim_purple("Model:"), model_path);
    eprintln!("   {}   {}:{}", dim_purple("Host:"), host, port);
    eprintln!();

    let engine: Arc<dyn LeafcutterEngine> = match NativeStreamingEngine::load(model_path) {
        Ok(e) => {
            eprintln!("{}", gold("✅ Native Streaming Engine loaded (low RAM mode)"));
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

    eprintln!("{}", gold(&format!("🌿 LeafcutterLLM v0.9.5 — Server Mode: {}", engine_type)));
    eprintln!("   {}  {}", dim_purple("Model:"), model_path);
    eprintln!("   {}   {}", dim_purple("Host:"), host);

    let engine: Arc<dyn LeafcutterEngine> = if engine_type == "native-streaming" {
        match NativeStreamingEngine::load(model_path) {
            Ok(e) => {
                eprintln!("{}", gold("✅ Native Streaming Engine loaded (low RAM mode)"));
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
                eprintln!("{}", gold("✅ llama.cpp FFI Engine loaded (Full load mode)"));
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
    tf_check: bool,
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

    if tf_check {
        run_tf_check_ffi(&model, model_path, prompt, system, raw);
        return;
    }

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
fn run_tf_check_ffi(model: &LlamaModel, _model_path: &PathBuf, prompt: &str, system: &str, raw: bool) {
    // Teacher-forcing oracle via the llama.cpp FFI backend. Prefill the
    // prompt, then feed a known reference continuation token-by-token and
    // report the fraction where the model's top-1 prediction matched the
    // reference. A low match rate flags a broken engine/quant, not a bad
    // answer — this is a sanity gate, kimi's "tiny-model oracle" idea.
    let mut ctx = match LlamaContext::new(model, 4096, 4) {
        Ok(c) => c,
        Err(e) => { eprintln!("❌ Failed to create context: {}", e); std::process::exit(1); }
    };
    let prompt_text = if raw {
        prompt.to_string()
    } else if system.is_empty() {
        prompt.to_string()
    } else {
        format!("{system}\n\n{prompt}")
    };
    let tokens = ctx.tokenize(&prompt_text, true, true);
    let reference = ctx.tokenize(" one two three four five six seven eight nine ten.", true, true);

    let mut matched = 0usize;
    let mut total = 0usize;
    for i in 0..reference.len() {
        let mut all = tokens.clone();
        all.extend_from_slice(&reference[..i]);
        match ctx.forward(&all) {
            Ok(logits) => {
                if let Some(pred) = argmax_top(&logits, model.n_vocab() as usize) {
                    total += 1;
                    if pred == reference[i] as usize {
                        matched += 1;
                    }
                }
            }
            Err(e) => { eprintln!("⚠️  forward failed at step {}: {}", i, e); break; }
        }
    }
    report_tf_check("ffi", matched, total);
    backend_free();
    if total > 0 && (matched as f64 / total as f64) < 0.5 {
        std::process::exit(1);
    }
}

/// Teacher-forcing oracle for the native engine. Prefill the prompt, then
/// force the reference continuation through and count top-1 matches.
fn run_tf_check_native(
    _engine: &mut leafcutter::inference::engine::Engine,
    model_path: &PathBuf,
    prompt: &str,
    system: &str,
    raw: bool,
) {
    use leafcutter::inference::engine::Engine;
    let mut engine = Engine::load(model_path.to_str().unwrap()).expect("load failed");
    let prompt_text = if raw {
        prompt.to_string()
    } else if system.is_empty() {
        prompt.to_string()
    } else {
        format!("{system}\n\n{prompt}")
    };
    let tokens = engine.tokenize(&prompt_text, true);
    let reference = engine.tokenize(" one two three four five six seven eight nine ten.", true);

    let mut matched = 0usize;
    let mut total = 0usize;
    for i in 0..reference.len() {
        let mut all = tokens.clone();
        all.extend_from_slice(&reference[..i]);
        let logits = engine.forward(&all);
        total += 1;
        if argmax(&logits) == Some(reference[i]) {
            matched += 1;
        }
    }
    report_tf_check("native", matched, total);
    if total > 0 && (matched as f64 / total as f64) < 0.5 {
        std::process::exit(1);
    }
}

fn argmax(logits: &[f32]) -> Option<usize> {
    if logits.is_empty() { return None; }
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    Some(best)
}

#[cfg(feature = "llama-ffi")]
fn argmax_top(logits: &[f32], vocab: usize) -> Option<usize> {
    argmax(&logits[..logits.len().min(vocab)])
}

fn report_tf_check(engine_name: &str, matched: usize, total: usize) {
    let pct = if total > 0 { 100.0 * matched as f64 / total as f64 } else { 0.0 };
    eprintln!("🌿 Teacher-forcing oracle ({} engine):", engine_name);
    eprintln!("   reference: ' one two three four five six seven eight nine ten.'");
    eprintln!("   top-1 match: {}/{} ({:.1}%)", matched, total, pct);
    if total > 0 && pct < 50.0 {
        eprintln!("⚠️  Match rate below 50% — engine or quant may be broken.");
    }
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
    tf_check: bool,
) {
    use leafcutter::inference::engine::Engine;

    let debug = std::env::var("LEAFCUTTER_DEBUG").map(|v| v == "1").unwrap_or(false);

    if debug {
        eprintln!("🌿 LeafcutterLLM Native Engine (no llama.cpp FFI)");
        eprintln!("   Model: {}", model_path.display());
    }

    let mut engine = match Engine::load(model_path.to_str().unwrap()) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    if tf_check {
        run_tf_check_native(&mut engine, model_path, prompt, system, raw);
        return;
    }

    // Resolve the per-architecture profile (Ollama Modelfile-style), which
    // supplies the chat template (with <think> for reasoning models), stop
    // tokens, and default sampling parameters.
    let gguf = GGUFile::open(model_path.to_str().unwrap()).ok();
    let profile = resolve_profile(
        &gguf.as_ref().map(|f| &f.metadata).cloned().unwrap_or_default(),
        None,
    );

    // Build the prompt.  --raw skips the chat template entirely (raw text
    // continuation); otherwise use the profile's chat template so reasoning
    // models like Ornith receive their system prompt + <think> opener.
    let prompt_text = if raw {
        prompt.to_string()
    } else {
        let history: Vec<(String, String)> = vec![("user".into(), prompt.to_string())];
        render_chat_prompt(&profile, system, &history)
    };

    let tokens = engine.tokenize(&prompt_text, true);
    if tokens.is_empty() {
        eprintln!("❌ No tokenizer available (GGUF lacks tokenizer.ggml.tokens metadata).");
        std::process::exit(1);
    }
    if debug {
        eprintln!("📝 Prompt tokens: {}", tokens.len());
    }

    let info = engine.info();
    if debug {
        eprintln!("   Arch: {}  Layers: {}  Hidden: {}", info.architecture, info.total_layers, info.hidden_size);
    }

    // Stream the response, showing the thinking block (💭) then the answer,
    // and stopping at the profile's stop tokens.  `</think>` may arrive as
    // the special token 248069 OR as raw byte pieces (`</`, `think`, `>`),
    // so we detect the closing marker in the *surface text* — a tiny tail
    // buffer that can hold a partial `</think>` across chunk boundaries.
    let stop_token_ids: Vec<usize> = profile
        .stop_tokens
        .iter()
        .map(|s| s.0)
        .collect();
    let mut in_thinking = profile.opens_with_thinking;
    let mut thinking_prefix_shown = false;
    let mut thinking_tail = String::new(); // may hold a partial `</think>`
    let top_p = 0.9;
    let generated = engine.generate_streaming_with_stops(
        &tokens,
        max_tokens,
        temperature,
        top_p,
        &stop_token_ids,
        |_id, chunk| {
            if in_thinking {
                thinking_tail.push_str(chunk);
                // Check for the full `</think>` marker BEFORE trimming
                // (it is 8 chars; the trimmed tail is only 7).
                if let Some(pos) = thinking_tail.find("</think>") {
                    let (pre, rest) = thinking_tail.split_at(pos);
                    if !pre.is_empty() {
                        if !thinking_prefix_shown {
                            eprint!("💭");
                            thinking_prefix_shown = true;
                        }
                        eprint!("{}", pre);
                    }
                    // Drop the marker; the answer resumes on the next line.
                    thinking_tail = rest["</think>".len()..].to_string();
                    in_thinking = false;
                    eprintln!();
                    let _ = io::stdout().flush();
                } else {
                    // Emit all but the last 7 chars (they can't be part of a
                    // partial `</think>`), keeping the boundary.
                    // SAFETY: the 7-byte window can straddle a multi-byte
                    // UTF-8 character (e.g. CJK or emoji), so we floor to the
                    // nearest char boundary to avoid a panic on `split_at`.
                    let keep = floor_char_boundary(&thinking_tail, thinking_tail.len().saturating_sub(7));
                    if keep > 0 {
                        let (emit, rest) = thinking_tail.split_at(keep);
                        if !thinking_prefix_shown {
                            eprint!("💭");
                            thinking_prefix_shown = true;
                        }
                        eprint!("{}", emit);
                        thinking_tail = rest.to_string();
                    }
                }
                let _ = io::stdout().flush();
                return true;
            }
            if !thinking_tail.is_empty() {
                print!("{}", thinking_tail);
                thinking_tail.clear();
            }
            print!("{}", chunk);
            let _ = io::stdout().flush();
            true
        },
    );
    println!();
    let _ = io::stdout().flush();

    if debug {
        eprintln!("[generate] {} tokens out", generated.len());
    }
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
            let is_gguf = path.extension().and_then(|s| s.to_str()) == Some("gguf");
            let is_st_dir =
                path.is_dir() && leafcutter::detect::looks_like_safetensors_dir(&path);
            if is_gguf || is_st_dir {
                found = true;
                let size = if is_st_dir {
                    leafcutter::detect::model_dir_size(&path)
                } else {
                    std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0)
                };
                let kind = if is_st_dir { "[safetensors]" } else { "" };
                println!(
                    "  {:<44} {} {}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    format_size(size),
                    kind
                );
            }
        }
    }

    if !found {
        eprintln!("No models found in {}", dir.display());
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

/// Get current RSS in bytes from /proc/self/status (Linux only).
/// VmRSS is the resident set size right now — the honest "how much RAM
/// am I using this moment" number, comparable to Ollama/llama.cpp output.
fn get_current_rss_bytes() -> u64 {
    if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
        for line in contents.lines() {
            if line.starts_with("VmRSS:") {
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

/// Get current RSS, return as MB for display
fn get_current_rss_mb() -> u64 {
    get_current_rss_bytes() / (1024 * 1024)
}

/// Format RSS for the stats line — shows "123 MB" or "1.2 GB"
fn format_rss(mb: u64) -> String {
    if mb >= 1024 {
        format!("{:.1} GB", mb as f64 / 1024.0)
    } else {
        format!("{} MB", mb)
    }
}
