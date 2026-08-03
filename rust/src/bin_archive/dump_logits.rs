//! dump_logits — diagnostic binary that runs a single forward pass on a
//! fixed prompt and prints the raw token IDs so we can see whether the model
//! is producing sane logits even when the streamed text looks incoherent.
//!
//! Usage:
//!     cargo run --release --bin dump_logits -- <model-path>
//!
//! Mirrors the load/tokenize/generate pattern in `main.rs::cmd_run` but with
//! `max_tokens = 1` so only the *first* sampled token is observed for each of
//! the two diagnostic prompts ("Hello" and "Hello world"; "What is 2+2?"; "The quick brown fox jumps"; "software engineering").
//!
//! NOTE: `generate_streaming_with` clears `kv_cache` / `ssm_cache` /
//! `deltanet_cache` and resets `seq_offset` at the top of each call (see
//! engine.rs ~L695-698), so running the two prompts back-to-back is safe —
//! no stale state leaks between passes.

use std::io::Write;

use leafcutter::inference::engine::Engine;

fn main() {
    let model_arg = std::env::args().nth(1);
    let model_path = match model_arg {
        Some(p) => p,
        None => {
            eprintln!("Usage: dump_logits <model-path>");
            eprintln!();
            eprintln!("Dumps token IDs for the single-token forward pass of");
            eprintln!("\"Hello\" and \"Hello world\" — useful for checking");
            eprintln!("whether a model's first-token logits are coherent.");
            std::process::exit(2);
        }
    };

    // Sanity check: fail early with a clear message if the file is missing,
    // mirroring cmd_run's find_model guard but without the auto-discovery.
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Model '{}' not found.", model_path);
        std::process::exit(1);
    }

    let mut engine = match Engine::load(&model_path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    let info = engine.info();
    eprintln!("Leaf: {}", model_path);
    eprintln!(
        "Arch: {}  Layers: {}  Hidden: {}",
        info.architecture, info.total_layers, info.hidden_size
    );
    eprintln!("─────────────────────────────────────────────────");

    // Two fixed prompts, deterministic sampling params so the only variable is
    // the model's logits. temp=0.0 collapses top-p sampling to greedy argmax,
    // which is exactly what we want for a "is the argmax sane?" probe.
    const TEMP: f32 = 0.0;
    const TOP_P: f32 = 1.0;

    run_probe(&mut engine, "Hello", TEMP, TOP_P);
    println!();
    run_probe(&mut engine, "Hello world", TEMP, TOP_P);
    println!();
    run_probe(&mut engine, "What is 2+2?", TEMP, TOP_P);
    println!();
    run_probe(&mut engine, "software engineering", TEMP, TOP_P);
}

/// Tokenize `prompt`, run a single forward pass (`max_tokens = 1`), and print
/// the raw input IDs plus the first generated token ID. The on_token callback
/// also captures the decoded chunk so we can see what the model "thinks" it
/// produced, not just the numeric id.
fn run_probe(engine: &mut Engine, prompt: &str, temp: f32, top_p: f32) {
    println!("=== Prompt: {:?} ===", prompt);

    let tokens = engine.tokenize(prompt, /* add_special */ true);
    if tokens.is_empty() {
        eprintln!(
            "[tokenization failed — no tokenizer available for {:?}]",
            prompt
        );
        return;
    }

    println!("Input token IDs ({}): {:?}", tokens.len(), tokens);

    let mut emitted: Option<String> = None;
    let generated_ids = engine.generate_streaming_with(
        &tokens,
        /* max_tokens */ 1,
        temp,
        top_p,
        |_id: usize, chunk: &str| {
            // Capture the decoded text for the (single) generated token so we
            // can compare id ↔ rendered char(s). Always return true; the
            // cap of max_tokens=1 means the loop exits regardless.
            emitted = Some(chunk.to_string());
            true
        },
    );

    // generated_ids holds every sampled token; with max_tokens=1 that is
    // exactly the first-token prediction we want to inspect.
    println!("First generated token ID: {:?}", generated_ids);
    match emitted {
        Some(ref s) => println!("Decoded first token: {:?}", s),
        None => println!("Decoded first token: <no callback fired — likely EOS or empty logits>"),
    }
}
