//! HTTP API server using Axum — Direct llama.cpp FFI backend
//!
//! Uses the existing `llama_ffi` module (already built and tested).
//! The server returns decoded text so clients don't need their own tokenizer.
//!
//! Endpoints:
//!   GET  /health                — health check
//!   POST /generate              — text generation (returns text + tokens for compat)
//!   POST /v1/chat/completions   — OpenAI-compatible chat API

use axum::{
    routing::{get, post},
    extract::State,
    Json, Router,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use crate::llama_ffi::{backend_init, backend_free, LlamaModel, LlamaContext};

// ---------------------------------------------------------------------------
// /generate endpoint — matches old LeafcutterLLM API but NOW returns text
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> usize { 256 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.9 }

/// Response format — NOW includes `text` so clients don't need a tokenizer.
/// `tokens` kept for backward compatibility but `text` is the primary field.
#[derive(Serialize, Deserialize)]
pub struct GenerateResponse {
    pub id: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub text: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tokens: Vec<usize>,
    pub took_ms: i64,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub error: String,
}

// ---------------------------------------------------------------------------
// /health endpoint
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

// ---------------------------------------------------------------------------
// /v1/chat/completions endpoint (OpenAI-compatible)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

// ---------------------------------------------------------------------------
// Engine wrapper — thin state around LlamaModel (context created per request)
// ---------------------------------------------------------------------------

pub struct FfiEngine {
    model: LlamaModel,
    ctx_size: u32,
    threads: i32,
}

impl FfiEngine {
    pub fn load(path: &str) -> Result<Self, String> {
        backend_init();
        let model = LlamaModel::load(std::path::Path::new(path), 0)?;
        Ok(Self {
            model,
            ctx_size: 4096,
            threads: 4,
        })
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<(String, Vec<usize>), String> {
        // Fresh context per request — avoids KV cache contamination
        let mut ctx = LlamaContext::new(&self.model, self.ctx_size, self.threads)
            .map_err(|e| format!("Failed to create context: {}", e))?;

        let prompt_tokens = ctx.tokenize(prompt, true, true);
        if prompt_tokens.is_empty() {
            return Err("Empty prompt after tokenization".to_string());
        }

        let eos = self.model.eos_token();
        let generated = ctx.generate(&prompt_tokens, max_tokens, temperature, eos);

        let text: String = generated.iter()
            .map(|&t| ctx.token_to_piece(t))
            .collect();

        let tokens: Vec<usize> = generated.iter().map(|&t| t as usize).collect();
        Ok((text, tokens))
    }
}

pub type SharedEngine = Arc<FfiEngine>;

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

pub async fn generate_handler(
    State(engine): State<SharedEngine>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let start = Instant::now();
    let id = format!("req-{}", start.elapsed().as_nanos());

    let (text, tokens) = tokio::task::spawn_blocking(move || {
        engine.generate(&req.prompt, req.max_tokens, req.temperature)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Task panic: {}", e)))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(GenerateResponse {
        id,
        text,
        tokens,
        took_ms: start.elapsed().as_millis() as i64,
        error: String::new(),
    }))
}

pub async fn health_handler(_state: State<SharedEngine>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: "leafcutter-ffi v0.9.0".to_string(),
    })
}

pub async fn chat_completions_handler(
    State(engine): State<SharedEngine>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    let prompt = req.messages.iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let (text, _tokens) = tokio::task::spawn_blocking(move || {
        engine.generate(&prompt, req.max_tokens, req.temperature)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Task panic: {}", e)))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let resp = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: "stop".to_string(),
        }],
    };

    Ok(Json(resp))
}

pub fn create_app(engine: SharedEngine) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/generate", post(generate_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(engine)
}

pub async fn run_server(engine: SharedEngine, port: u16) {
    let app = create_app(engine);
    let addr = format!("0.0.0.0:{}", port);
    println!("🚀 Leafcutter FFI server listening on http://{}", addr);
    println!("   GET  /health");
    println!("   POST /generate");
    println!("   POST /v1/chat/completions");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
