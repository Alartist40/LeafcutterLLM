//! HTTP API server using Axum — Direct llama.cpp FFI backend OR Native Streaming
//!
//! Native Streaming works without any FFI. FfiEngine requires llama-ffi.

use axum::{
    middleware,
    routing::{get, post},
    extract::State,
    Json, Router,
    http::{StatusCode, HeaderMap},
    body::Body,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "llama-ffi")]
use crate::llama_ffi::{backend_init, LlamaModel, LlamaContext};
use crate::inference::engine::Engine as NativeEngine;
use crate::tokenizer::GgufBpeTokenizer;
use tokio_stream::StreamExt;

// ---------------------------------------------------------------------------
// Types
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
}

fn default_max_tokens() -> usize { 256 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.9 }

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

#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub engine: String,
}

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
// Unified Engine Trait
// ---------------------------------------------------------------------------

pub trait LeafcutterEngine: Send + Sync {
    fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32, top_p: f32) -> Result<(String, Vec<usize>), String>;
    fn name(&self) -> &str;
    fn max_seq_len(&self) -> usize;
}

// ---------------------------------------------------------------------------
// FFI Engine (Standard llama.cpp behavior)
// ---------------------------------------------------------------------------

#[cfg(feature = "llama-ffi")]
pub struct FfiEngine {
    model: LlamaModel,
    ctx_size: u32,
    threads: i32,
}

#[cfg(feature = "llama-ffi")]
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
}

#[cfg(feature = "llama-ffi")]
impl LeafcutterEngine for FfiEngine {
    fn name(&self) -> &str { "llama-ffi" }
    fn max_seq_len(&self) -> usize {
        self.model.n_ctx_train().max(1) as usize
    }
    fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32, top_p: f32) -> Result<(String, Vec<usize>), String> {
        let mut ctx = LlamaContext::new(&self.model, self.ctx_size, self.threads)
            .map_err(|e| format!("Failed to create context: {}", e))?;

        let prompt_tokens = ctx.tokenize(prompt, true, true);
        if prompt_tokens.is_empty() { return Err("Empty prompt".to_string()); }

        let eos = self.model.eos_token();
        // top_p is accepted by the API for OpenAI compatibility but the FFI
        // binding doesn't expose llama.cpp's sampler chain. llama.cpp's
        // internal sampler uses top_p=0.95. If the caller passes a non-default
        // top_p, surface it as a warning so the user knows it's ignored.
        if (top_p - 0.95).abs() > 0.01 && (top_p - 0.9).abs() > 0.01 {
            eprintln!("⚠️  top_p={} requested but FFI engine uses llama.cpp's internal top_p=0.95; value ignored", top_p);
        }
        let _ = top_p;
        let generated = ctx.generate(&prompt_tokens, max_tokens, temperature, eos);

        let text: String = generated.iter().map(|&t| ctx.token_to_piece(t)).collect();
        let tokens: Vec<usize> = generated.iter().map(|&t| t as usize).collect();
        Ok((text, tokens))
    }
}

// ---------------------------------------------------------------------------
// Native Streaming Engine (The "Leafcutter" Magic)
// ---------------------------------------------------------------------------

pub struct NativeStreamingEngine {
    engine: std::sync::Mutex<NativeEngine>,
    tokenizer: Arc<dyn crate::tokenizer::BaseTokenizer + Send + Sync>,
}

impl NativeStreamingEngine {
    pub fn load(path: &str) -> Result<Self, String> {
        let engine = NativeEngine::load(path).map_err(|e| e.to_string())?;
        let tokenizer = GgufBpeTokenizer::from_gguf(path)
            .ok_or_else(|| "No tokenizer found in GGUF".to_string())?;

        Ok(Self {
            engine: std::sync::Mutex::new(engine),
            tokenizer: Arc::new(tokenizer),
        })
    }
}

impl LeafcutterEngine for NativeStreamingEngine {
    fn name(&self) -> &str { "native-streaming" }
    fn max_seq_len(&self) -> usize {
        self.engine.lock().ok().map(|e| e.config.max_seq_len).unwrap_or(4096)
    }
    fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32, top_p: f32) -> Result<(String, Vec<usize>), String> {
        let mut engine = self.engine.lock().map_err(|_| "Engine lock poisoned".to_string())?;
        let tokens = self.tokenizer.encode(prompt);

        let generated = engine.generate(&tokens, max_tokens, temperature, top_p);
        let text = self.tokenizer.decode(&generated);

        Ok((text, generated))
    }
}

pub type SharedEngine = Arc<dyn LeafcutterEngine>;

// Auth: DISABLED by default. Set LEAFCUTTER_API_KEY env var to enable.
// Setting it to any non-empty string requires that value on every request.
// An empty or unset env var means no auth — the server runs open (intended
// for local-loopback development). Pair with `--host` binding to 127.0.0.1.

// ---------------------------------------------------------------------------
// Auth middleware
// ---------------------------------------------------------------------------

async fn auth_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let key = std::env::var("LEAFCUTTER_API_KEY").unwrap_or_default();
    if key.is_empty() {
        // No API key configured — auth disabled (default, loopback-only safe).
        return next.run(req).await;
    }
    match req.headers().get("X-API-Key") {
        Some(v) if v.to_str().map(|s| s == key).unwrap_or(false) => next.run(req).await,
        _ => axum::response::Response::builder()
            .status(StatusCode::UNAUTHORIZED)
            .body(Body::from("Missing or invalid X-API-Key header"))
            .unwrap(),
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

pub async fn generate_handler(
    State(engine): State<SharedEngine>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let start = Instant::now();
    // Use wall-clock nanos for uniqueness — start.elapsed() here is ~0
    // and would produce duplicate IDs across concurrent requests.
    let id = format!(
        "req-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    let capped_max = engine.max_seq_len().min(req.max_tokens);

    let (text, out_tokens) = tokio::task::spawn_blocking(move || {
        engine.generate(&req.prompt, capped_max, req.temperature, req.top_p)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Task panic: {}", e)))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let truncated = capped_max < req.max_tokens;
    let text = if truncated { format!("{}[truncated]", text) } else { text };

    Ok(Json(GenerateResponse {
        id,
        text,
        tokens: out_tokens,
        took_ms: start.elapsed().as_millis() as i64,
        error: String::new(),
    }))
}

pub async fn health_handler(State(engine): State<SharedEngine>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: format!("v{}", env!("CARGO_PKG_VERSION")),
        engine: engine.name().to_string(),
    })
}

pub async fn chat_completions_handler(
    State(engine): State<SharedEngine>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    use axum::response::IntoResponse;
    let prompt = req.messages.iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let capped_max = engine.max_seq_len().min(req.max_tokens);

    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(32);
        let model_name = req.model.clone();
        let req_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        tokio::task::spawn_blocking(move || {
            if let Ok((text, _)) = engine.generate(&prompt, capped_max, req.temperature, req.top_p) {
                for chunk in text.as_bytes().chunks(16) {
                    if let Ok(s) = std::str::from_utf8(chunk) {
                        let json_chunk = serde_json::json!({
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": s },
                                "finish_reason": null
                            }]
                        });
                        let _ = tx.blocking_send(json_chunk.to_string());
                    }
                }
            }
            let _ = tx.blocking_send("[DONE]".to_string());
        });

        use axum::response::sse::Event;
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
            .map(|data| Ok::<_, std::convert::Infallible>(Event::default().data(data)));
        return Ok(axum::response::Sse::new(stream).into_response());
    }

    let (text, _tokens) = tokio::task::spawn_blocking(move || {
        engine.generate(&prompt, capped_max, req.temperature, req.top_p)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Task panic: {}", e)))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let truncated = capped_max < req.max_tokens;
    let text = if truncated { format!("{}[truncated]", text) } else { text };

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

    Ok(Json(resp).into_response())
}

pub fn create_app(engine: SharedEngine) -> Router {
    let key = std::env::var("LEAFCUTTER_API_KEY").unwrap_or_default();
    if !key.is_empty() {
        println!("🔐 Auth enabled — send X-API-Key header on all requests");
    } else {
        println!("🔓 Auth disabled (LEAFCUTTER_API_KEY not set) — server is open");
    }
    Router::new()
        .route("/health", get(health_handler))
        .route("/generate", post(generate_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(engine)
        .layer(middleware::from_fn(auth_middleware))
}

pub async fn run_server(engine: SharedEngine, port: u16, host: &str) {
    let app = create_app(engine);
    let addr = format!("{}:{}", host, port);
    println!("🚀 Leafcutter server listening on http://{}", addr);

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("❌ Failed to bind to {}: {}", addr, e);
            std::process::exit(1);
        }
    };
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("❌ Server error: {}", e);
        std::process::exit(1);
    }
}