//! `leafcutter launch <app>` — Ollama-style app launcher.
//!
//! Modeled on `ollama launch` (see cmd/launch in the ollama tree): the
//! leafcutter server is a *persistent* daemon that launch never tears down
//! (unlike a naive spawn-and-kill wrapper). The one managed integration is
//! **cynapse** (ManagedSingleModel, the Hermes pattern): launch rewrites its
//! `config.yaml` so `provider = leafcutter`, `model = <abs path>`,
//! `leafcutter_path` set, then launches the cynapse binary. Cynapse spawns
//! its own leafcutter server (see crates/cynapse-core/src/llm/leafcutter.rs),
//! so no server is started here.
//!
//! Custom apps registered with `leafcutter app add` get a generic launch:
//! start a leafcutter server if they need one, export `LEAFCUTTER_BASE_URL`,
//! spawn the command.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::config;

/// A leafcutter-model descriptor used when writing app configs.
pub struct LaunchModel {
    pub name: String,
    pub path: String,
}

/// Request describing one `leafcutter launch <app>` invocation.
#[derive(Debug, Default)]
pub struct LaunchRequest {
    pub app: String,
    pub model_override: Option<String>,
    /// Re-run configuration even if it already matches (equivalent to the
    /// interactive launcher selecting a different model).
    pub force_configure: bool,
    /// Configure the app but do not launch it.
    pub configure_only: bool,
    /// Restore an app config back to its pre-launch state.
    pub restore: bool,
    /// Extra arguments passed through to the app after `--`.
    pub extra_args: Vec<String>,
    /// Auto-approve install/configuration prompts.
    pub yes: bool,
}

/// Resolve the model a launch should use:
///   1. explicit `--model`
///   2. the app's persisted last model
///   3. the global last-used model
///   4. the largest model on disk
/// Returns an absolute GGUF path.
pub fn resolve_model(req: &LaunchRequest, models: &[String]) -> Result<String, String> {
    let candidate = req
        .model_override
        .clone()
        .or_else(|| config::app_model(&req.app))
        .or_else(|| config::load().last_model);

    if let Some(name) = candidate {
        if let Some(p) = find_model_path(&name, models) {
            return Ok(p);
        }
        if req.model_override.is_some() {
            return Err(format!("Model '{}' not found.", name));
        }
    }

    // Fall back to the largest model.
    if let Some(p) = models.last() {
        return Ok(p.clone());
    }
    Err("No models found. Point the tool at your models with: leafcutter source add <dir>".into())
}

/// Resolve a model name/partial name to an absolute path within `models`.
fn find_model_path(name: &str, models: &[String]) -> Option<String> {
    // Direct path to a .gguf file.
    let p = PathBuf::from(shellexpand::tilde(name).as_ref());
    let is_gguf = p.extension().and_then(|s| s.to_str()) == Some("gguf");
    if p.exists() && is_gguf {
        return Some(p.to_string_lossy().into_owned());
    }
    // Index number (e.g. "0", "1")?
    if let Ok(idx) = name.parse::<usize>() {
        if idx < models.len() {
            return Some(models[idx].clone());
        }
    }
    let needle = name.to_lowercase();
    // Exact basename match, then substring.
    for m in models {
        let base = Path::new(m)
            .file_name()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if base == needle {
            return Some(m.clone());
        }
    }
    for m in models {
        if m.to_lowercase().contains(&needle) {
            return Some(m.clone());
        }
    }
    None
}

/// The public entry point: run one `leafcutter launch <app>`.
pub fn launch(req: &LaunchRequest, models: &[String]) -> Result<(), String> {
    // 1. Resolve the integration (custom config entry first, built-ins next).
    let entry = config::resolve_app(&req.app)
        .ok_or_else(|| unknown_app_message(&req.app))?;

    let integration = match integration_for(&req.app) {
        Some(i) => i,
        None => return launch_generic(&entry, req, models),
    };

    // 2. --restore: undo a launch-managed config rewrite.
    if req.restore {
        integration.restore(&req.app)?;
        return Ok(());
    }

    // 3. Resolve the model.
    let model_path = resolve_model(req, models)?;
    let model = LaunchModel {
        name: Path::new(&model_path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| model_path.clone()),
        path: model_path.clone(),
    };

    // 3b. Config fingerprint: if the saved model for this app changed on disk
    // (re-downloaded / re-quantized), warn so the user knows the remembered
    // model may no longer match what launch configured before. The fingerprint
    // is recorded at the end of a successful configure.
    let fingerprint = model_fingerprint(&model_path);
    if let (Some(saved_fp), Some(current)) = (config::app_model_fp(&req.app), fingerprint.clone()) {
        if saved_fp != current {
            eprintln!(
                "⚠️  Model '{}' changed on disk since last launch (fingerprint mismatch).",
                model_path
            );
        }
    }

    // 4. Ensure the leafcutter server is up (persistent daemon — never killed
    //    here). Cynapse spawns its own server, so it opts out.
    let server = if integration.needs_server() {
        Some(ensure_server(&model.path)?)
    } else {
        None
    };

    // 5. Configure the app to point at leafcutter + the chosen model.
    integration.configure(&req.app, &model, server.as_deref())?;

    // 5b. Remember the model (+ its config fingerprint) for next launch.
    config::set_app_model(&req.app, &model.path, fingerprint.as_deref());

    // 6. Optionally stop after configuring.
    if req.configure_only {
        return Ok(());
    }

    // 7. Launch the app, attached to our stdio.
    integration.run(&req.app, &model, server.as_deref(), &req.extra_args)
}

/// Generic launch for user-registered apps: ensure a server, inject
/// `LEAFCUTTER_BASE_URL`/`LEAFCUTTER_MODEL`, spawn the command.
fn launch_generic(entry: &config::AppEntry, req: &LaunchRequest, models: &[String]) -> Result<(), String> {
    let model_path = resolve_model(req, models)?;
    let server = if entry.needs_server {
        Some(ensure_server(&model_path)?)
    } else {
        None
    };

    if req.restore {
        // Generic apps have no managed config to restore.
        return Ok(());
    }
    if req.configure_only {
        eprintln!("🌿 '{}' configured.", req.app);
        return Ok(());
    }

    let mut cmd = Command::new(&entry.command);
    cmd.args(&entry.args);
    cmd.env("LEAFCUTTER_MODEL", &model_path);
    if let Some(url) = server {
        cmd.env("LEAFCUTTER_BASE_URL", url);
    }
    for (k, v) in &entry.env {
        cmd.env(k, v);
    }
    cmd.args(&req.extra_args);
    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = cmd
        .spawn()
        .map_err(|e| format!("Failed to launch '{}': {}", req.app, e))?
        .wait()
        .map_err(|e| format!("Error waiting for '{}': {}", req.app, e))?;

    if !status.success() {
        return Err(format!("'{}' exited with {}", req.app, status));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration registry
// ─────────────────────────────────────────────────────────────────────────────

/// One launch-managed application.
trait Integration {
    fn display_name(&self) -> &'static str;
    /// Whether a persistent leafcutter server must be running first.
    fn needs_server(&self) -> bool;
    /// Locate the app's config file (may not exist yet).
    fn config_path(&self, app: &str) -> PathBuf;
    /// Write the launch-managed config pointing at leafcutter + model.
    fn configure(&self, app: &str, model: &LaunchModel, server_url: Option<&str>) -> Result<(), String>;
    /// Undo a previous `configure` from the launch backup.
    fn restore(&self, app: &str) -> Result<(), String>;
    /// Launch the app process (attached).
    fn run(&self, app: &str, model: &LaunchModel, server_url: Option<&str>, extra: &[String]) -> Result<(), String>;
}

fn integration_for(name: &str) -> Option<Box<dyn Integration>> {
    match name {
        "cynapse" => Some(Box::new(Cynapse)),
        _ => None,
    }
}

fn unknown_app_message(app: &str) -> String {
    let mut msg = format!("Unknown app: '{}'\n\nKnown apps:", app);
    for (name, entry) in config::builtin_apps() {
        let server = if entry.needs_server { " (server)" } else { "" };
        msg.push_str(&format!("\n  - {}{}", name, server));
    }
    msg.push_str("\n\nRegister your own with: leafcutter app add <name> --command <cmd> [--needs-server]");
    msg
}

// ─────────────────────────────────────────────────────────────────────────────
// cynapse — ManagedSingleModel (config.yaml rewrite)
// ─────────────────────────────────────────────────────────────────────────────

struct Cynapse;

impl Integration for Cynapse {
    fn display_name(&self) -> &'static str { "Cynapse" }
    fn needs_server(&self) -> bool { false }
    fn config_path(&self, _app: &str) -> PathBuf {
        // cynapse reads ./config.yaml first, then ~/.cynapse/config.yaml.
        if Path::new("config.yaml").exists() {
            PathBuf::from("config.yaml")
        } else if let Some(home) = std::env::var_os("HOME") {
            PathBuf::from(home).join(".cynapse").join("config.yaml")
        } else {
            PathBuf::from("config.yaml")
        }
    }

    fn configure(&self, app: &str, model: &LaunchModel, _server: Option<&str>) -> Result<(), String> {
        let path = self.config_path(app);

        // Back up the pre-launch config for --restore.
        if path.exists() {
            config::save_launch_backup(app, &path);
        }

        // Minimal YAML edit: set provider + model + leafcutter_path under `llm:`.
        let text = if path.exists() {
            std::fs::read_to_string(&path).map_err(|e| format!("read {}: {}", path.display(), e))?
        } else {
            String::new()
        };
        let mut out = rewrite_cynapse_yaml(&text, &model.path, &current_exe()?)?;

        // Make sure the llm section is present and indented properly.
        let llm = "llm:\n    provider: leafcutter\n";
        if out.trim().is_empty() {
            out = format!("{llm}    model: {}\n    leafcutter_path: {}\n", model.path, current_exe()?);
        }
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(&path, out).map_err(|e| format!("write {}: {}", path.display(), e))?;

        eprintln!(
            "🌿 {} configured to use leafcutter (model: {})",
            self.display_name(),
            model.path
        );
        Ok(())
    }

    fn restore(&self, app: &str) -> Result<(), String> {
        let path = self.config_path(app);
        if config::restore_launch_backup(app, &path) {
            eprintln!("🌿 {} config restored.", self.display_name());
        } else {
            eprintln!("No launch backup found for '{}'.", app);
        }
        Ok(())
    }

    fn run(&self, _app: &str, _model: &LaunchModel, _server: Option<&str>, extra: &[String]) -> Result<(), String> {
        // cynapse spawns its own leafcutter server; just launch the binary.
        let bin = find_binary("cynapse").ok_or_else(|| {
            "cynapse binary not found in PATH. Install cynapse-rs first.".to_string()
        })?;
        let status = Command::new(&bin)
            .args(extra)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Failed to launch cynapse: {}", e))?
            .wait()
            .map_err(|e| format!("Error waiting for cynapse: {}", e))?;
        if !status.success() {
            return Err(format!("cynapse exited with {}", status));
        }
        Ok(())
    }
}

/// Edit a cynapse config.yaml `llm:` block in place (provider, model,
/// leafcutter_path). Preserves every other section and key. Indentation
/// follows the sample config (4 spaces under `llm:`).
fn rewrite_cynapse_yaml(text: &str, model_path: &str, leafcutter_path: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut in_llm = false;
    let mut wrote = false;
    for line in text.lines() {
        if line.trim_start().is_empty() || line.trim_start().starts_with('#') {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if indent == 0 {
            in_llm = line.trim_end().trim_end_matches(':') == "llm";
        }
        if in_llm && indent > 0 && !wrote {
            // Inject our managed keys at the top of the llm block.
            out.push_str(&format!("    provider: leafcutter\n"));
            out.push_str(&format!("    model: {}\n", model_path));
            out.push_str(&format!("    leafcutter_path: {}\n", leafcutter_path));
            wrote = true;
        }
        let key = line.trim().split(':').next().unwrap_or("");
        let managed = in_llm && matches!(key, "provider" | "model" | "leafcutter_path" | "llama_server_path");
        if managed {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    if !in_llm && !wrote {
        // No llm section at all — caller prepends one.
        out.push_str("llm:\n    provider: leafcutter\n");
        out.push_str(&format!("    model: {}\n", model_path));
        out.push_str(&format!("    leafcutter_path: {}\n", leafcutter_path));
        wrote = true;
    } else if !wrote {
        out.push_str(&format!("    model: {}\n", model_path));
        out.push_str(&format!("    leafcutter_path: {}\n", leafcutter_path));
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistent leafcutter server (the "daemon")
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_PORT: u16 = 8081;

/// Ensure a leafcutter server is running on DEFAULT_PORT. If none is healthy,
/// spawn `leafcutter serve --model <path> --port 8081` detached and wait for
/// `/health`. The server is *not* torn down when the app exits — it is a
/// persistent daemon (like `ollama serve`). Returns the base URL.
fn ensure_server(model_path: &str) -> Result<String, String> {
    let base_url = format!("http://127.0.0.1:{}", DEFAULT_PORT);
    if server_healthy(DEFAULT_PORT) {
        eprintln!("🌿 leafcutter server already running on port {}.", DEFAULT_PORT);
        return Ok(base_url);
    }

    let self_exe = current_exe()?;
    eprintln!("🌿 Starting leafcutter server on port {} ...", DEFAULT_PORT);
    let mut child = Command::new(&self_exe)
        .args(["serve", "--model", model_path, "--port", &DEFAULT_PORT.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to start leafcutter server: {}", e))?;

    // Wait up to 5 minutes for model load.
    let deadline = Instant::now() + Duration::from_secs(300);
    while Instant::now() < deadline {
        if server_healthy(DEFAULT_PORT) {
            eprintln!("🌿 leafcutter server ready: {}", base_url);
            return Ok(base_url);
        }
        if let Some(status) = child.try_wait().map_err(|e| format!("server poll error: {}", e))? {
            return Err(format!("leafcutter server exited early with {}", status));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err(format!("leafcutter server did not become healthy on port {}.", DEFAULT_PORT))
}

fn server_healthy(port: u16) -> bool {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    let addr = match format!("127.0.0.1:{}", port).parse() {
        Ok(a) => a,
        Err(_) => return false,
    };
    match TcpStream::connect_timeout(&addr, Duration::from_millis(500)) {
        Ok(mut stream) => {
            let _ = stream.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
            let mut buf = [0u8; 32];
            match stream.read(&mut buf) {
                Ok(_) => {
                    let head = String::from_utf8_lossy(&buf);
                    head.contains("200 OK") || head.contains("200 ok")
                }
                Err(_) => false,
            }
        }
        Err(_) => false,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn current_exe() -> Result<String, String> {
    std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .map_err(|e| format!("cannot locate leafcutter binary: {}", e))
}

fn find_binary(name: &str) -> Option<PathBuf> {
    for dir in std::env::var("PATH").unwrap_or_default().split(':') {
        let candidate = PathBuf::from(dir).join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Config fingerprint of a GGUF model file (stable across metadata order).
/// None if the file isn't a readable GGUF.
fn model_fingerprint(path: &str) -> Option<String> {
    crate::model::gguf::GGUFile::open(path).ok().map(|f| f.fingerprint())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cynapse_rewrite_updates_existing_llm_block() {
        let input = "gateway:\n    address: 0.0.0.0:8080\nllm:\n    provider: ollama\n    model: qwen-bench:latest\n    max_tokens: 1024\nmemory:\n    persona_path: ./persona\n";
        let out = rewrite_cynapse_yaml(input, "/mnt/big.gguf", "/usr/local/bin/leafcutter").unwrap();
        assert!(out.contains("provider: leafcutter"));
        assert!(out.contains("model: /mnt/big.gguf"));
        assert!(out.contains("leafcutter_path: /usr/local/bin/leafcutter"));
        assert!(out.contains("max_tokens: 1024"));
        assert!(out.contains("address: 0.0.0.0:8080"));
        assert!(out.contains("persona_path: ./persona"));
        assert!(!out.contains("qwen-bench"));
    }

    #[test]
    fn cynapse_rewrite_adds_llm_when_missing() {
        let out = rewrite_cynapse_yaml("memory:\n    persona_path: ./persona\n", "/m.gguf", "/leafcutter").unwrap();
        assert!(out.contains("llm:"));
        assert!(out.contains("provider: leafcutter"));
        assert!(out.contains("model: /m.gguf"));
        assert!(out.contains("persona_path: ./persona"));
    }

    #[test]
    fn find_model_path_handles_name_and_index() {
        let models = vec!["/m/ornith-1.0-9b-Q4_K_M.gguf".to_string(), "/m/mini.gguf".to_string()];
        assert_eq!(find_model_path("ornith", &models).as_deref(), Some("/m/ornith-1.0-9b-Q4_K_M.gguf"));
        assert_eq!(find_model_path("1", &models).as_deref(), Some("/m/mini.gguf"));
        assert_eq!(find_model_path("/m/mini.gguf", &models).as_deref(), Some("/m/mini.gguf"));
        assert_eq!(find_model_path("zzz", &models), None);
    }
}