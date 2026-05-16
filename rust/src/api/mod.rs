//! HTTP API server using Axum

use axum::{
    routing::{get, post},
    extract::State,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::inference::engine::Engine;

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

#[derive(Serialize)]
pub struct GenerateResponse {
    pub text: String,
    pub tokens: Vec<usize>,
    pub took_ms: u64,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

pub type SharedEngine = Arc<Mutex<Engine>>;

pub async fn generate_handler(
    State(engine): State<SharedEngine>,
    Json(req): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let start = Instant::now();

    let mut engine = engine.lock().unwrap();

    // Simple character-level tokenization for demo
    let tokens: Vec<usize> = req.prompt.bytes().map(|b| b as usize).collect();
    let generated = engine.generate(&tokens, req.max_tokens, req.temperature, req.top_p);

    let text = String::from_utf8_lossy(
        &generated.iter().map(|&t| t as u8).collect::<Vec<u8>>()
    ).to_string();

    Json(GenerateResponse {
        text,
        tokens: generated,
        took_ms: start.elapsed().as_millis() as u64,
    })
}

pub async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: "0.8.0-rust",
    })
}

pub fn create_app(engine: SharedEngine) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/generate", post(generate_handler))
        .with_state(engine)
}

pub async fn run_server(engine: SharedEngine, port: u16) {
    let app = create_app(engine);
    let addr = format!("0.0.0.0:{}", port);
    println!("🚀 Leafcutter Rust server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
