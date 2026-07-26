//! leaf — terminal chat REPL for LeafcutterLLM (Ollama-style).
//!
//! Usage:
//!   leaf list                          List .gguf models in the models directory
//!   leaf run <model-name>             Start a chat session with a model
//!   leaf run <model-name> --temp 0.7  Override temperature
//!   leaf run <path/to/model.gguf>     Run a model by direct path
//!
//! In-session commands:
//!   /bye   or  /quit   End the session
//!   /clear            Clear conversation history (fresh context)
//!   /temp <f>         Set temperature for next turns
//!   /help             Show in-session commands
//!
//! Environment:
//!   LEAF_MODELS_DIR   Override the models directory (default: ./models, then ~/Downloads/models)
//!   LEAFCUTTER_PREFETCH=0    Disable async layer prefetch
//!   LEAFCUTTER_ANTIDOOM=0    Disable anti-doom loop detection
//!   LEAFCUTTER_PROFILE       Print per-token timing
//!   LEAFCUTTER_PROFILE_BLOCKS Print per-tensor parse time at load
//!
//! All engine optimizations carry through: async layer prefetch,
//! anti-doom loop detection, zero-copy mmap, SIMD matmul.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_help_top();
        std::process::exit(0);
    }

    match args[1].as_str() {
        "list" => cmd_list(),
        "run" => {
            if args.len() < 3 {
                eprintln!("Usage: leaf run <model-name-or-path> [--temp <f>] [--top-p <f>] [--max-tokens <n>]");
                std::process::exit(1);
            }
            let model_arg = &args[2];
            let mut temp = 0.7f32;
            let mut top_p = 0.9f32;
            let mut max_tokens = 512usize;
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--temp" | "-t" => {
                        if i + 1 < args.len() {
                            temp = args[i + 1].parse().unwrap_or(0.7);
                            i += 2;
                        } else { i += 1; }
                    }
                    "--top-p" | "-p" => {
                        if i + 1 < args.len() {
                            top_p = args[i + 1].parse().unwrap_or(0.9);
                            i += 2;
                        } else { i += 1; }
                    }
                    "--max-tokens" | "-n" => {
                        if i + 1 < args.len() {
                            max_tokens = args[i + 1].parse().unwrap_or(512);
                            i += 2;
                        } else { i += 1; }
                    }
                    "--help" | "-h" => {
                        eprintln!("Usage: leaf run <model-name-or-path> [--temp <f>] [--top-p <f>] [--max-tokens <n>]");
                        std::process::exit(0);
                    }
                    _ => { eprintln!("Unknown flag: {}", args[i]); i += 1; }
                }
            }
            cmd_run(model_arg, temp, top_p, max_tokens);
        }
        "help" | "--help" | "-h" => print_help_top(),
        "version" | "--version" | "-V" => {
            println!("leaf {} — LeafcutterLLM chat REPL", env!("CARGO_PKG_VERSION"));
        }
        _ => {
            eprintln!("Unknown command: '{}'\n", args[1]);
            print_help_top();
            std::process::exit(1);
        }
    }
}

fn print_help_top() {
    eprintln!("leaf — terminal chat REPL for LeafcutterLLM\n");
    eprintln!("Commands:");
    eprintln!("  leaf list                         List available GGUF models");
    eprintln!("  leaf run <model> [opts]           Start chatting with a model");
    eprintln!("  leaf help                         Show this help");
    eprintln!("  leaf version                      Show version\n");
    eprintln!("Options for 'leaf run':");
    eprintln!("  --temp <f>       Sampling temperature (default 0.7)");
    eprintln!("  --top-p <f>      Top-p nucleus sampling (default 0.9)");
    eprintln!("  --max-tokens <n> Max tokens per response (default 512)\n");
    eprintln!("In-session commands: /bye  /quit  /clear  /temp <f>  /help\n");
    eprintln!("Environment:");
    eprintln!("  LEAF_MODELS_DIR   Override models directory (default: ./models, then ~/Downloads/models)");
    eprintln!("  See README for LEAFCUTTER_* tuning knobs.\n");
}

// ── model discovery ─────────────────────────────────────────────

/// Resolve the models directory: env override > ./models > ~/Downloads/models.
fn models_dir() -> PathBuf {
    if let Ok(d) = std::env::var("LEAF_MODELS_DIR") {
        return PathBuf::from(d);
    }
    let local = PathBuf::from("models");
    if local.is_dir() {
        return local;
    }
    if let Some(home) = dirs_home() {
        let dl = home.join("Downloads").join("models");
        if dl.is_dir() {
            return dl;
        }
    }
    // Fall back to local even if it doesn't exist — the error message
    // will tell the user to create it.
    local
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// One discovered model file.
struct ModelEntry {
    name: String,
    path: PathBuf,
    size_mb: u64,
}

/// Scan the models directory for *.gguf files. Returns entries sorted by
/// name (case-insensitive). Silently skips unreadable files.
fn scan_models(dir: &Path) -> Vec<ModelEntry> {
    let mut entries = Vec::new();
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return entries,
    };
    for e in read.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("gguf") {
            continue;
        }
        let size_mb = e.metadata().map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        entries.push(ModelEntry { name, path, size_mb });
    }
    entries.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    entries
}

/// Resolve a user-provided model argument to a concrete path.  Accepts:
///   - absolute/relative path to a .gguf file (used directly)
///   - a bare name like "Ministral-3-3B-Instruct-2512-Q4_K_M" (matched
///     case-insensitively against file_stems in the models dir)
///   - a partial substring match (first hit wins)
///   - if still nothing and the models dir has exactly one .gguf, use it
fn resolve_model(arg: &str) -> Result<PathBuf, String> {
    // Direct path?
    let p = PathBuf::from(arg);
    if p.is_file() {
        return Ok(p);
    }

    // Scan and match by name.
    let dir = models_dir();
    let models = scan_models(&dir);

    // Exact stem match (case-insensitive).
    let arg_lower = arg.to_lowercase();
    if let Some(m) = models.iter().find(|m| m.name.to_lowercase() == arg_lower) {
        return Ok(m.path.clone());
    }
    // Substring match.
    if let Some(m) = models.iter().find(|m| m.name.to_lowercase().contains(&arg_lower)) {
        return Ok(m.path.clone());
    }
    // Last resort: single model auto-pick.
    if models.len() == 1 {
        return Ok(models[0].path.clone());
    }

    let mut msg = format!("Model '{}' not found.\n", arg);
    if models.is_empty() {
        msg.push_str(&format!("No .gguf files in '{}'.\n", dir.display()));
        msg.push_str("Download a GGUF model and place it there, or pass a full path.");
    } else {
        msg.push_str("Available models:\n");
        for m in &models {
            msg.push_str(&format!("  {} ({} MB)\n", m.name, m.size_mb));
        }
    }
    Err(msg)
}

// ── commands ─────────────────────────────────────────────────────

fn cmd_list() {
    let dir = models_dir();
    let models = scan_models(&dir);
    if models.is_empty() {
        eprintln!("No .gguf models found in {}", dir.display());
        eprintln!("\nDownload a model and place it there, or set LEAF_MODELS_DIR.");
        return;
    }
    println!("{:<50} {:>10}  {}", "NAME", "SIZE", "PATH");
    println!("{}", "-".repeat(80));
    for m in &models {
        println!("{:<50} {:>6} MB  {}", m.name, m.size_mb, m.path.display());
    }
}

fn cmd_run(model_arg: &str, mut temp: f32, top_p: f32, max_tokens: usize) {
    let path = match resolve_model(model_arg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let size_mb = std::fs::metadata(&path)
        .map(|m| m.len() / (1024 * 1024))
        .unwrap_or(0);

    // Loading message goes to stderr so streaming stdout stays clean.
    eprintln!("Loading {} ({} MB) ...", name, size_mb);
    let t0 = std::time::Instant::now();

    let path_str = path.to_string_lossy().to_string();
    let mut engine = match leafcutter::inference::engine::Engine::load(&path_str) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!("Loaded in {:.2}s", t0.elapsed().as_secs_f64());

    // Print a short banner.
    eprintln!("\nChat with {} (temp={:.1}, top_p={:.1}, max_tokens={})", name, temp, top_p, max_tokens);
    eprintln!("Type /bye to exit, /clear to reset context, /help for commands.\n");

    let mut conversation: Vec<(String, String)> = Vec::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Prompt.
        print!(">>> ");
        let _ = stdout.flush();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() || input.is_empty() {
            break;
        }
        let trimmed = input.trim();

        if trimmed.is_empty() {
            continue;
        }

        // In-session commands.
        match trimmed {
            "/bye" | "/quit" | "/exit" => break,
            "/help" => {
                eprintln!("\nCommands:");
                eprintln!("  /bye  /quit  /exit   End session");
                eprintln!("  /clear               Clear conversation history");
                eprintln!("  /temp <f>            Set temperature (current: {:.1})", temp);
                eprintln!("  /tokens <n>          Set max tokens per reply (current: {})", max_tokens);
                eprintln!();
                continue;
            }
            "/clear" => {
                conversation.clear();
                eprintln!("[context cleared]\n");
                continue;
            }
            s if s.starts_with("/temp ") => {
                if let Ok(v) = s[6..].trim().parse::<f32>() {
                    temp = v;
                    eprintln!("[temperature set to {:.1}]\n", temp);
                } else {
                    eprintln!("Usage: /temp <number>\n");
                }
                continue;
            }
            s if s.starts_with("/tokens ") => {
                if let Ok(_v) = s[8..].trim().parse::<usize>() {
                    eprintln!("[max tokens: {} — restart for it to take effect]\n", _v);
                }
                continue;
            }
            s if s.starts_with('/') => {
                eprintln!("Unknown command: {} (try /help)\n", s);
                continue;
            }
            _ => {}
        }

        // Build the full conversation context.  We use the engine's
        // format_chat_prompt which auto-detects the chat template
        // (Llama3 / ChatML / Mistral / Gemma).
        conversation.push(("user".to_string(), trimmed.to_string()));

        let mut full_prompt = String::new();
        for (role, text) in &conversation {
            if role == "user" {
                full_prompt.push_str(&engine.format_chat_prompt("", text));
            } else {
                // Assistant turns — append raw text after the user turns;
                // the chat template wraps the latest user turn.  For now
                // we only send the most recent user turn and rely on the
                // engine's single-turn template.  Multi-turn context
                // building is a TODO once the template system supports
                // arbitrary message lists.
                full_prompt.push_str(text);
            }
        }

        // Tokenize and generate with streaming.
        let tokens = engine.tokenize(&full_prompt, true);
        if tokens.is_empty() {
            eprintln!("[error: tokenization returned empty — is the tokenizer loaded?]\n");
            continue;
        }

        eprintln!(); // blank line before output
        let _ = stdout.flush();

        let t_gen = std::time::Instant::now();
        let mut token_count = 0usize;

        let _generated = engine.generate_streaming_with(
            &tokens,
            max_tokens,
            temp,
            top_p,
            |_id: usize, surface: &str| {
                print!("{}", surface);
                let _ = io::stdout().flush();
                token_count += 1;
                true // keep going
            },
        );

        let elapsed = t_gen.elapsed().as_secs_f64();
        eprintln!();
        if elapsed > 0.0 && token_count > 0 {
            eprintln!(
                "\n[{} tokens in {:.1}s  |  {:.2} tok/s]",
                token_count,
                elapsed,
                token_count as f64 / elapsed
            );
        }
        eprintln!();
    }

    eprintln!("\nbye.");
}

// End of leaf.rs
