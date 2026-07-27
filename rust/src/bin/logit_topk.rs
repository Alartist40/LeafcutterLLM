//! Diagnostic: run a forward pass through native Engine, print top-K logits.
//!
//! Compares against Ollama's HTTP `/api/generate` for the same prompt to
//! find where native diverges from ground truth.
//!
//! Usage: cargo run --release --bin logit_topk -- <gguf> <prompt> [top_k=10]

use leafcutter::inference::engine::Engine;
use std::env;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <gguf> <prompt> [top_k=10]", args[0]);
        std::process::exit(1);
    }
    let gguf_path = &args[1];
    let prompt = &args[2];
    let top_k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);

    eprintln!("[logit_topk] GGUF: {}", gguf_path);
    eprintln!("[logit_topk] Prompt: {:?}", prompt);

    // Load engine
    let mut engine = Engine::load(gguf_path).expect("Failed to load engine");
    let info = engine.info();
    eprintln!(
        "[logit_topk] Arch: {}  Layers: {}  Hidden: {}",
        info.architecture, info.total_layers, info.hidden_size
    );

    // Tokenize (engine wraps the GGUF tokenizer)
    let tokens = engine.tokenize(prompt, true);
    eprintln!("[logit_topk] Prompt tokens: {}", tokens.len());
    eprintln!("[logit_topk] Token IDs: {:?}", &tokens[..tokens.len().min(60)]);
    eprintln!();

    // Run forward pass
    let logits = match engine.forward_native(&tokens) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[logit_topk] Forward pass failed: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!("[logit_topk] Logits vec length: {}", logits.len());

    // Find top-K by logit value
    let mut indexed: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("[logit_topk] Top-{} native predictions:", top_k);
    // Decode each top-K id by tokenizing it through our BPE-aware tokenizer
    // (cheaper than loading the full vocab list).
    for (rank, (id, logit)) in indexed.iter().take(top_k).enumerate() {
        // Single-token decode: feed the id through decode().  For multi-token
        // IDs this would give garbled output, but top-K predictions are
        // usually single tokens.  Fall back to id display if decode is empty.
        let probe = vec![*id];
        let surface = engine
            .decode(&probe)
            .trim()
            .to_string();
        let surface_display = if surface.is_empty() {
            format!("<id:{}>", id)
        } else {
            format!("{:?}", surface)
        };
        eprintln!(
            "  #{}: id={:>7}  logit={:>10.4}  surface={}",
            rank + 1,
            id,
            logit,
            surface_display
        );
    }
    eprintln!();

    // Now query Ollama for the same prompt (no chat template — raw text)
    eprintln!("[logit_topk] Querying Ollama for ground truth...");
    let ollama_output = Command::new("curl")
        .args([
            "-s",
            "-X",
            "POST",
            "http://127.0.0.1:11434/api/generate",
            "-H",
            "Content-Type: application/json",
            "-d",
            &format!(
                "{{\"model\":\"ornith:9b\",\"prompt\":{},\"raw\":true,\"stream\":false,\"options\":{{\"num_predict\":1,\"temperature\":0,\"top_k\":{}}}}}",
                serde_json::to_string(prompt).unwrap_or_else(|_| format!("{:?}", prompt)),
                top_k
            ),
        ])
        .output();
    match ollama_output {
        Ok(out) if out.status.success() => {
            let body = String::from_utf8_lossy(&out.stdout);
            eprintln!("[logit_topk] Ollama raw response:");
            eprintln!("{}", body);
        }
        Ok(out) => {
            eprintln!(
                "[logit_topk] Ollama curl failed: stderr={}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Err(e) => {
            eprintln!("[logit_topk] Failed to run curl: {}", e);
        }
    }
}
