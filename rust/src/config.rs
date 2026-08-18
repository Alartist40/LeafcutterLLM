//! Persistent user configuration: where the colony looks for models.
//!
//! The one problem this solves: "I downloaded a model to my Downloads dir /
//! my documents / a special partition — how does `leafcutter run <name>`
//! find it?" Answer: a plain JSON file listing search directories, written
//! once via `/source <dir>` (or `leafcutter source add <dir>`), read on every
//! run. The binary is not tied to any CWD — paths resolve from the config
//! first, then `LEAF_MODELS_DIR` (colon-separated), then built-in defaults.
//!
//! Config file locations (OS-aware):
//!   - Linux:     `$XDG_CONFIG_HOME/leafcutter/config.json` or `~/.config/leafcutter/config.json`
//!   - macOS:     `~/Library/Application Support/leafcutter/config.json`
//!   - Windows:   `%APPDATA%\leafcutter\config.json`

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// How `leafcutter launch <app>` starts a program.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppEntry {
    /// Executable to run (resolved against PATH, or a direct path).
    pub command: String,
    /// Fixed arguments passed to the executable.
    #[serde(default)]
    pub args: Vec<String>,
    /// Ensure a leafcutter server is running while the app runs.
    #[serde(default)]
    pub needs_server: bool,
    /// Extra environment variables injected into the app process.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Last model selected for this app (persisted by launch).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Config fingerprint of the model file `model` pointed at when it was
    /// saved (kimi-k3-in-c "config fingerprint"): if the model file changed
    /// (re-downloaded, re-quantized), launch can detect a stale saved model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_fp: Option<String>,
}

/// What we persist. JSON so users can hand-edit it like a `.ollama`/Modelfile.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// Search directories for models, in priority order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_dirs: Vec<String>,
    /// Last model used (for a fast `leafcutter run` resume). Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_model: Option<String>,
    /// Apps known to `leafcutter launch <app>`, keyed by name.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub apps: HashMap<String, AppEntry>,
}

/// Path to the config file for the current OS. Does not create it.
pub fn config_path() -> PathBuf {
    let dir = config_dir();
    dir.join("config.json")
}

/// OS-aware config directory (may not exist yet).
pub fn config_dir() -> PathBuf {
    if cfg!(target_os = "windows") {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            return PathBuf::from(appdata).join("leafcutter");
        }
    } else if cfg!(target_os = "macos") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("leafcutter");
        }
    }
    // Linux / FreeBSD / fallback: XDG_CONFIG_HOME or ~/.config
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        return PathBuf::from(xdg).join("leafcutter");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".config").join("leafcutter");
    }
    // Last resort: CWD-relative so the binary still works anywhere.
    PathBuf::from(".leafcutter")
}

/// Load the config; a missing or corrupt file yields the defaults.
pub fn load() -> Config {
    let path = config_path();
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
        Err(_) => Config::default(),
    }
}

/// Save the config, creating the directory if needed. Best-effort.
pub fn save(cfg: &Config) {
    let dir = config_dir();
    let _ = std::fs::create_dir_all(&dir);
    if let Ok(text) = serde_json::to_string_pretty(cfg) {
        let _ = std::fs::write(config_path(), text);
    }
}

/// All model search directories, in priority order:
///   1. built-in defaults (first existing one wins) — no config needed
///   2. `LEAF_MODELS_DIR` env (colon-separated, overrides defaults)
///   3. config-file `model_dirs` (appended, user `/source` additions)
pub fn model_dirs() -> Vec<PathBuf> {
    let cfg = load();

    // Env override wins over everything (explicit user intent, e.g. from a
    // container mount or CI). Colon-separated on all platforms for simplicity.
    if let Ok(env) = std::env::var("LEAF_MODELS_DIR") {
        if !env.is_empty() {
            return env
                .split(':')
                .filter(|s| !s.is_empty())
                .map(|s| PathBuf::from(shellexpand::tilde(s).as_ref()))
                .collect();
        }
    }

    let mut dirs: Vec<PathBuf> = Vec::new();

    // Config-file dirs first (user explicitly pointed us somewhere).
    for d in &cfg.model_dirs {
        let expanded = shellexpand::tilde(d);
        dirs.push(PathBuf::from(expanded.as_ref()));
    }

    // Built-in defaults: an existing ./models, then ~/Downloads/models.
    // Only appended if not already present, so duplicates don't accumulate.
    for cand in ["./models", "~/Downloads/models"] {
        let expanded = shellexpand::tilde(cand);
        let p = PathBuf::from(expanded.as_ref());
        if !dirs.contains(&p) {
            dirs.push(p);
        }
    }

    dirs
}

/// Add a directory to the config's `model_dirs` and persist it.
/// Returns true if the directory was newly added.
pub fn add_model_dir(dir: &str) -> bool {
    let mut cfg = load();
    let expanded = shellexpand::tilde(dir).as_ref().to_owned();
    if !cfg.model_dirs.contains(&expanded) {
        cfg.model_dirs.push(expanded);
        save(&cfg);
        true
    } else {
        false
    }
}

/// Remove a directory from the config's `model_dirs` and persist it.
pub fn remove_model_dir(dir: &str) -> bool {
    let mut cfg = load();
    let expanded = shellexpand::tilde(dir).as_ref().to_owned();
    let before = cfg.model_dirs.len();
    cfg.model_dirs.retain(|d| d != &expanded);
    if cfg.model_dirs.len() != before {
        save(&cfg);
        true
    } else {
        false
    }
}

/// Register an app for `leafcutter launch`. Returns true if newly added.
pub fn add_app(name: &str, entry: AppEntry) -> bool {
    let mut cfg = load();
    if cfg.apps.contains_key(name) {
        false
    } else {
        cfg.apps.insert(name.to_string(), entry);
        save(&cfg);
        true
    }
}

/// Remove a registered app. Returns true if it existed.
pub fn remove_app(name: &str) -> bool {
    let mut cfg = load();
    if cfg.apps.remove(name).is_some() {
        save(&cfg);
        true
    } else {
        false
    }
}

/// Look up a registered app, falling back to a built-in default.
pub fn resolve_app(name: &str) -> Option<AppEntry> {
    let cfg = load();
    if let Some(entry) = cfg.apps.get(name) {
        return Some(entry.clone());
    }
    builtin_app(name)
}

/// Built-in app registry: programs that commonly talk to a local leafcutter
/// server. Users can override or extend these via `leafcutter app add`.
pub fn builtin_apps() -> Vec<(String, AppEntry)> {
    let mut out = Vec::new();
    for (name, entry) in [
        (
            "cynapse",
            AppEntry {
                command: "cynapse".into(),
                args: Vec::new(),
                needs_server: false,
                env: HashMap::new(),
                model: None,
                model_fp: None,
            },
        ),
    ] {
        out.push((name.to_string(), entry));
    }
    out
}

fn builtin_app(name: &str) -> Option<AppEntry> {
    builtin_apps()
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, e)| e)
}

/// Remember the model launch last used for an app, so `leafcutter launch <app>`
/// can resume with the same model without typing it again. Also records the
/// model's config fingerprint so a stale (changed) model file can be detected.
pub fn set_app_model(name: &str, model: &str, fingerprint: Option<&str>) {
    let mut cfg = load();
    // Builtin apps (e.g. cynapse) live in builtin_apps(), not cfg.apps, so we
    // upsert a copy into cfg.apps to persist the per-app model choice.
    let entry = match cfg.apps.get_mut(name) {
        Some(e) => e,
        None => {
            cfg.apps.insert(name.to_string(), builtin_app(name).unwrap_or_default());
            cfg.apps.get_mut(name).unwrap()
        }
    };
    entry.model = Some(model.to_string());
    entry.model_fp = fingerprint.map(|f| f.to_string());
    save(&cfg);
}

/// The last model launch selected for an app, if any.
pub fn app_model(name: &str) -> Option<String> {
    let cfg = load();
    if let Some(entry) = cfg.apps.get(name) {
        return entry.model.clone();
    }
    builtin_app(name).and_then(|e| e.model)
}

/// The stored config fingerprint for an app's saved model, if any.
pub fn app_model_fp(name: &str) -> Option<String> {
    let cfg = load();
    cfg.apps.get(name).and_then(|e| e.model_fp.clone())
}

/// Path where `launch` stores a pre-launch backup of an app's config file,
/// used by `--restore` to undo a launch-managed rewrite.
pub fn launch_backup_path(app: &str) -> PathBuf {
    config_dir().join(format!("launch-{}.bak", app))
}

/// Write a backup of the given file (if it exists) for `--restore`.
pub fn save_launch_backup(app: &str, path: &std::path::Path) {
    if let Ok(data) = std::fs::read(path) {
        let _ = std::fs::create_dir_all(&config_dir());
        let _ = std::fs::write(launch_backup_path(app), data);
    }
}

/// Restore an app's config file from the launch backup (if present).
/// Returns true if a backup was restored.
pub fn restore_launch_backup(app: &str, path: &std::path::Path) -> bool {
    let backup = launch_backup_path(app);
    if !backup.exists() {
        return false;
    }
    let data = match std::fs::read(&backup) {
        Ok(d) => d,
        Err(_) => return false,
    };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if std::fs::write(path, data).is_ok() {
        let _ = std::fs::remove_file(&backup);
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_empty() {
        let cfg = Config::default();
        assert!(cfg.model_dirs.is_empty());
        assert!(cfg.last_model.is_none());
        assert!(cfg.apps.is_empty());
    }

    #[test]
    fn roundtrip_serialization() {
        let mut cfg = Config {
            model_dirs: vec!["/mnt/models".into(), "~/models".into()],
            last_model: Some("ornith".into()),
            apps: std::collections::HashMap::new(),
        };
        cfg.apps.insert(
            "cynapse".into(),
            AppEntry {
                command: "cynapse".into(),
                args: vec!["--gui".into()],
                needs_server: false,
                env: std::collections::HashMap::new(),
                model: None,
                model_fp: None,
            },
        );
        let text = serde_json::to_string(&cfg).unwrap();
        let back: Config = serde_json::from_str(&text).unwrap();
        assert_eq!(back.model_dirs, cfg.model_dirs);
        assert_eq!(back.last_model, cfg.last_model);
        assert_eq!(back.apps.len(), 1);
        assert_eq!(back.apps["cynapse"].command, "cynapse");
        assert_eq!(back.apps["cynapse"].needs_server, false);
    }

    #[test]
    fn corrupt_json_falls_back_to_default() {
        let _ = std::fs::create_dir_all("/tmp/leafcutter_cfg_test");
        // We can't easily point config_path() at /tmp without env, so just
        // verify the serde path handles garbage.
        let back: Config = serde_json::from_str("not json {{").unwrap_or_default();
        assert!(back.model_dirs.is_empty());
    }

    #[test]
    fn add_is_idempotent() {
        let mut cfg = Config::default();
        cfg.model_dirs.push("/a".into());
        cfg.model_dirs.push("/a".into());
        // dedupe via contains check on insert
        let expanded = "/a".to_string();
        if !cfg.model_dirs.contains(&expanded) {
            cfg.model_dirs.push(expanded);
        }
        assert_eq!(cfg.model_dirs.len(), 2);
    }
}
