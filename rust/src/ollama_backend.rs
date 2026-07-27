//! Ollama HTTP API backend.
//!
//! Routes inference through a locally-running `ollama serve` instance,
//! giving us Ollama-quality generation without the layer-streaming memory
//! benefit but with full template/sampling accuracy.
//!
//! This is the "ground truth" path — if native Leafcutter output diverges
//! from Ollama output with the same prompt, the bug is in our forward
//! pass.  If they match, the divergence is in our chat template or
//! sampling.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};

#[derive(Clone)]
pub struct OllamaClient {
    pub host: String,
    pub model: String,
    pub http_client: ureq::Agent,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    /// Enable thinking/reasoning stream for Ornith, Qwen3 thinking,
    /// DeepSeek-R1, etc.  When true, Ollama splits the response into
    /// `content` (final answer) and `thinking` (reasoning trace).
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
    options: ChatOptions<'a>,
}

#[derive(Debug, Serialize)]
struct ChatOptions<'a> {
    temperature: f32,
    top_p: f32,
    top_k: i32,
    num_predict: i32,
    stop: &'a [String],
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Per-token thinking/reasoning stream.  Ollama sends this as a
    /// delta — each chunk has only the NEW token, not the accumulated
    /// thinking text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    model: String,
    message: ChatMessage,
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
}

impl OllamaClient {
    pub fn new(host: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            host: host.into(),
            model: model.into(),
            http_client: ureq::AgentBuilder::new()
                .timeout_read(std::time::Duration::from_secs(120))
                .timeout_write(std::time::Duration::from_secs(60))
                .build(),
        }
    }

    /// Generate a single non-streaming response.
    pub fn chat(
        &self,
        messages: &[ChatMessage],
        temperature: f32,
        top_p: f32,
        top_k: i32,
        max_tokens: i32,
        stop: &[String],
    ) -> Result<ChatResponse, String> {
        let req = ChatRequest {
            model: &self.model,
            messages,
            stream: false,
            think: Some(true),
            options: ChatOptions {
                temperature,
                top_p,
                top_k,
                num_predict: max_tokens,
                stop,
            },
        };
        let url = format!("{}/api/chat", self.host);
        let resp = self
            .http_client
            .post(&url)
            .set("Content-Type", "application/json")
            .send_json(&req)
            .map_err(|e| format!("HTTP error: {e}"))?;
        let body = resp.into_string().map_err(|e| e.to_string())?;
        serde_json::from_str::<ChatResponse>(&body).map_err(|e| format!("JSON error: {e} ({body})"))
    }

    /// Generate with streaming output.  Calls `on_token` for each streamed
    /// content chunk (decoded JSON messages from `/api/chat`).
    ///
    /// Implementation: Ollama sends newline-delimited JSON.  We buffer
    /// the entire NDJSON stream into a String and then split it into
    /// lines — simpler than chunked parsing and ureq's stream parsing
    /// has been flaky in older versions.
    pub fn chat_streaming<F>(
        &self,
        messages: &[ChatMessage],
        temperature: f32,
        top_p: f32,
        top_k: i32,
        max_tokens: i32,
        stop: &[String],
        mut on_token: F,
    ) -> Result<(), String>
    where
        F: FnMut(&str, Option<&str>) -> bool,
    {
        let req = ChatRequest {
            model: &self.model,
            messages,
            stream: true,
            think: Some(true),
            options: ChatOptions {
                temperature,
                top_p,
                top_k,
                num_predict: max_tokens,
                stop,
            },
        };
        let url = format!("{}/api/chat", self.host);
        let resp = self
            .http_client
            .post(&url)
            .set("Content-Type", "application/json")
            .send_json(&req)
            .map_err(|e| format!("HTTP error: {e}"))?;
        // Buffer the whole NDJSON body — Ollama streams token-by-token
        // but the body is small (a few KB) for typical max_predict.
        let body = resp.into_string().map_err(|e| e.to_string())?;
        if std::env::var("LEAFCUTTER_OLLAMA_DEBUG").is_ok() {
            eprintln!("[ollama-debug] body bytes: {}", body.len());
            eprintln!("[ollama-debug] first line: {}", body.lines().next().unwrap_or(""));
        }
        for line in body.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<ChatResponse>(line) {
                Ok(r) => {
                    if std::env::var("LEAFCUTTER_OLLAMA_DEBUG").is_ok() {
                        eprintln!(
                            "[ollama-debug] chunk: done={} content={:?} thinking={:?}",
                            r.done, r.message.content, r.message.thinking
                        );
                    }
                    if r.done {
                        return Ok(());
                    }
                    let continue_emit =
                        on_token(&r.message.content, r.message.thinking.as_deref());
                    if !continue_emit {
                        return Ok(());
                    }
                }
                Err(e) => {
                    eprintln!("[ollama-stream] parse error: {e} on line: {}", &line[..line.len().min(80)]);
                }
            }
        }
        Ok(())
    }
}
