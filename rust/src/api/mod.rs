//! HTTP API server using Axum
//!
//! Now uses `HybridEngine` which tries native Rust first, then falls back
//! to llama-server via the bridge for unsupported architectures.
//!
//! Endpoints:
//!   GET  /health                — health check
//!   POST /generate              — Leafcutter native text generation
//!   POST /v1/chat/completions   — OpenAI-compatible chat API

use axum::{
    routing::{get, post},
    extract::State,
    Json, Router,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::bridge::HybridEngine;

// ---------------------------------------------------------------------------
// /generate endpoint
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
    pub text: String,
    pub tokens: Vec<usize>,
    pub took_ms: u64,
    pub backend: String,
}

// ---------------------------------------------------------------------------
// /health endpoint
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub backend: String,
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
// Router
// ---------------------------------------------------------------------------

pub type SharedEngine = Arc<Mutex<HybridEngine>>;

pub async fn generate_handler(
    State(engine): State<SharedEngine>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let start = Instant::now();
    let mut engine = engine.lock().map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Lock error: {}", e)))?;
    let backend = if engine.native.is_some() { "native" } else { "bridge" };
    let text = engine.generate(&req.prompt, req.max_tokens, req.temperature, req.top_p);
    let tokens: Vec<usize> = text.bytes().map(|b| b as usize).collect();

    Ok(Json(GenerateResponse {
        text,
        tokens,
        took_ms: start.elapsed().as_millis() as u64,
        backend: backend.to_string(),
    }))
}

pub async fn health_handler(State(engine): State<SharedEngine>) -> Json<HealthResponse> {
    let engine = engine.lock().unwrap();
    let backend = if engine.native.is_some() { "native" } else { "bridge" };

    Json(HealthResponse {
        status: "ok".to_string(),
        version: "0.9.0-hybrid".to_string(),
        backend: backend.to_string(),
    })
}

pub async fn chat_completions_handler(
    State(engine): State<SharedEngine>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, String)> {
    let mut eng = engine.lock().map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Lock error: {}", e)))?;

    // Build prompt from messages
    let prompt = req.messages.iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let content = eng.generate(&prompt, req.max_tokens, req.temperature, req.top_p);

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
                content,
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
    println!("🚀 Leafcutter Hybrid server listening on http://{}", addr);
    println!("   GET  /health");
    println!("   POST /generate");
    println!("   POST /v1/chat/completions");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request;
    use tower::util::ServiceExt;

    fn dummy_engine() -> SharedEngine {
        Arc::new(Mutex::new(HybridEngine {
            native: None,
            bridge: None,
            model_path: "dummy".to_string(),
        }))
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = create_app(dummy_engine());
        let response = app
            .oneshot(Request::builder().uri("/health").body(axum::body::Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_chat_completions_endpoint() {
        let app = create_app(dummy_engine());
        let req_body = serde_json::json!({
            "model": "leafcutter-test",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 10,
        });
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/chat/completions")
                    .method("POST")
                    .header("Content-Type", "application/json")
                    .body(axum::body::Body::from(serde_json::to_string(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let resp: ChatCompletionResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp.choices[0].message.role, "assistant");
        assert_eq!(resp.object, "chat.completion");
    }
}
