//! Llama.cpp bridge — delegates inference for unsupported architectures
//!
//! When Leafcutter encounters a model it cannot run natively (e.g. Qwen3.5
//! with SSM layers), it transparently spawns llama-server and forwards
//! requests.  This gives users a working system TODAY while native support
//! is built in `leafcutter advanced/`.
//!
//! Usage:
//!   1. Install llama.cpp and build `llama-server`
//!   2. Point Leafcutter at any GGUF file
//!   3. If native support is missing, bridge auto-spawns llama-server
//!
//! The bridge exposes the same `/generate` and `/health` endpoints as the
//! native Axum server, so clients never need to change.

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Holds the llama-server child process and its configuration.
pub struct LlamaBridge {
    /// Path to the llama-server binary
    pub server_bin: String,
    /// Path to the GGUF model file
    pub model_path: String,
    /// Port llama-server listens on
    pub port: u16,
    /// Number of threads for llama-server
    pub threads: usize,
    /// Context size
    pub ctx_size: usize,
    /// The running child process (if any)
    child: Option<Child>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LlamaGenerateRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub n_predict: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct LlamaGenerateResponse {
    pub content: String,
}

fn default_max_tokens() -> usize { 256 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.9 }

impl LlamaBridge {
    /// Create a new bridge configuration (does not start the server yet).
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            server_bin: "llama-server".to_string(),
            model_path: model_path.into(),
            port: 8082, // one port above native Rust server
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            ctx_size: 4096,
            child: None,
        }
    }

    /// Attempt to locate llama-server binary in common paths.
    pub fn with_auto_detected_binary(mut self) -> Self {
        let candidates = [
            "llama-server",
            "/usr/local/bin/llama-server",
            "/usr/bin/llama-server",
            "/opt/llama.cpp/llama-server",
            "./llama.cpp/llama-server",
            "../llama.cpp/llama-server",
        ];
        for c in &candidates {
            if which::which(c).is_ok() || std::path::Path::new(c).exists() {
                self.server_bin = c.to_string();
                break;
            }
        }
        self
    }

    /// Start llama-server as a child process.
    pub fn start(&mut self) -> Result<(), BridgeError> {
        if self.child.is_some() {
            return Ok(());
        }

        let args = vec![
            "-m".to_string(), self.model_path.clone(),
            "--port".to_string(), self.port.to_string(),
            "-t".to_string(), self.threads.to_string(),
            "-c".to_string(), self.ctx_size.to_string(),
            "--host".to_string(), "127.0.0.1".to_string(),
        ];

        println!("🌉 Starting llama-server bridge on port {} ...", self.port);
        println!("   {} {}", self.server_bin, args.join(" "));

        let child = Command::new(&self.server_bin)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| BridgeError::SpawnFailed(e.to_string()))?;

        self.child = Some(child);

        // Wait a moment for server to bind
        std::thread::sleep(Duration::from_millis(1500));

        // Quick health check
        if !self.is_healthy() {
            return Err(BridgeError::ServerNotResponding);
        }

        println!("✅ llama-server bridge ready on http://127.0.0.1:{}", self.port);
        Ok(())
    }

    /// Check if llama-server is responding.
    pub fn is_healthy(&self) -> bool {
        let url = format!("http://127.0.0.1:{}/health", self.port);
        if let Ok(resp) = ureq::get(&url).timeout(Duration::from_secs(2)).call() {
            return resp.status() == 200;
        }
        false
    }

    /// Generate text via llama-server's `/completion` endpoint.
    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32, top_p: f32) -> Result<String, BridgeError> {
        let url = format!("http://127.0.0.1:{}/completion", self.port);

        let req_body = serde_json::json!({
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": false,
        });

        let json_str = serde_json::to_string(&req_body)
            .map_err(|e| BridgeError::RequestFailed(e.to_string()))?;

        let resp = ureq::post(&url)
            .timeout(Duration::from_secs(300))
            .set("Content-Type", "application/json")
            .send_string(&json_str)
            .map_err(|e| BridgeError::RequestFailed(e.to_string()))?;

        let json: serde_json::Value = resp.into_json()
            .map_err(|e| BridgeError::ParseFailed(e.to_string()))?;

        let content = json.get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(content)
    }

    /// Stop the llama-server child process.
    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
            println!("🌉 llama-server bridge stopped.");
        }
    }
}

impl Drop for LlamaBridge {
    fn drop(&mut self) {
        self.stop();
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("Failed to spawn llama-server: {0}")]
    SpawnFailed(String),
    #[error("llama-server started but is not responding to health checks")]
    ServerNotResponding,
    #[error("HTTP request to llama-server failed: {0}")]
    RequestFailed(String),
    #[error("Failed to parse llama-server response: {0}")]
    ParseFailed(String),
}

// ---------------------------------------------------------------------------
// Integration helper: Engine wrapper that falls back to bridge
// ---------------------------------------------------------------------------

use crate::inference::engine::Engine;
use crate::tokenizer::GgufBpeTokenizer;

/// A unified inference backend that tries native Rust first,
/// then falls back to the llama.cpp bridge.
pub struct HybridEngine {
    pub native: Option<Engine>,
    pub bridge: Option<LlamaBridge>,
    pub model_path: String,
    /// Tokenizer for the native path. Built from the GGUF embedded vocab
    /// at load time so we never fall back to byte-level tokenization.
    tokenizer: Option<GgufBpeTokenizer>,
}

impl HybridEngine {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Try native Rust engine first
        match Engine::load(path) {
            Ok(engine) => {
                println!("✅ Native Rust engine loaded: {} layers", engine.config.num_hidden_layers);
                // Build a real tokenizer from the GGUF embedded vocab so the
                // native generate path never uses byte-level fallback.
                let tokenizer = GgufBpeTokenizer::from_gguf(path);
                if tokenizer.is_none() {
                    eprintln!("⚠️  No GGUF-embedded tokenizer found for {}; native path will not work correctly", path);
                }
                return Ok(Self {
                    native: Some(engine),
                    bridge: None,
                    model_path: path.to_string(),
                    tokenizer,
                });
            }
            Err(e) => {
                println!("⚠️  Native engine cannot run this model: {}", e);
                println!("   Falling back to llama.cpp bridge...");
            }
        }

        // Fall back to bridge
        let mut bridge = LlamaBridge::new(path).with_auto_detected_binary();
        bridge.start()?;

        Ok(Self {
            native: None,
            bridge: Some(bridge),
            model_path: path.to_string(),
            tokenizer: None,
        })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f32, top_p: f32) -> String {
        if let Some(engine) = &mut self.native {
            // Native path: use the GGUF-embedded tokenizer, NOT byte casting.
            // If no tokenizer is available, return an explicit error — do not
            // silently corrupt the input by treating each byte as a token ID.
            let tokenizer = match &self.tokenizer {
                Some(t) => t,
                None => {
                    eprintln!("❌ No tokenizer available for native path; cannot tokenize input");
                    return "[Error: no tokenizer available — cannot use native generate without a GGUF tokenizer]".to_string();
                }
            };
            let tokens = tokenizer.encode(prompt);
            let generated = engine.generate(&tokens, max_tokens, temperature, top_p);
            tokenizer.decode(&generated)
        } else if let Some(bridge) = &self.bridge {
            // Bridge path
            match bridge.generate(prompt, max_tokens, temperature, top_p) {
                Ok(text) => text,
                Err(e) => {
                    eprintln!("Bridge generation failed: {}", e);
                    format!("[Error: {}]", e)
                }
            }
        } else {
            "[Error: no engine loaded]".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_config() {
        let bridge = LlamaBridge::new("/tmp/test.gguf");
        assert_eq!(bridge.port, 8082);
        assert!(!bridge.is_healthy()); // nothing running
    }
}
