// Debug test to trip a single RMSNorm call against a known input.
// Goal: see whether the math is correct or whether the explosion
//        happens *inside* RMSNorm itself.
//
// Run with:
//   cargo test --release --test debug_gemma_rmsnorm -- --nocapture

use leafcutter::inference::gemma::gemma_rms_norm;
use leafcutter::inference::engine::Engine;
use leafcutter::tokenizer::GgufTokenizer;

#[test]
#[ignore = "manual debug test against local GGUF file"]
fn debug_rmsnorm_input() {
    let model_path = std::env::var("LEAFCUTTER_MODEL").unwrap_or_else(|_| {
        "/home/xander/Downloads/models/gemma-4-12b-it-Q4_K_M.gguf".to_string()
    });
    let prompt = std::env::var("LEAFCUTTER_PROMPT").unwrap_or_else(|_| "Hello".to_string());

    eprintln!("Loading model: {model_path}");
    let mut engine = Engine::load(&model_path).expect("Failed to load model");
    eprintln!("Loaded. hidden_size={}, vocab={}", engine.config.hidden_size, engine.config.vocab_size);

    // Encode via the engine's tokenizer (try FFI first, then native rust).
    let mut tokens = engine.tokenize(&prompt, true);
    if tokens.is_empty() {
        eprintln!("Engine.tokenize returned empty — using rust GgufTokenizer fallback.");
        let tok = GgufTokenizer::from_gguf(&model_path).expect("tokenizer load failed");
        tokens = tok.encode(&prompt, true);
    }
    eprintln!("Tokenized: ids={:?}", tokens);

    // Embed → row of hidden_size per token.
    let hidden = engine.embed_lookup_mmap(&tokens).expect("embed failed");
    let scale = (engine.config.hidden_size as f32).sqrt();
    let hidden_scaled = {
        let mut d = hidden.data.clone();
        for v in d.iter_mut() { *v *= scale; }
        leafcutter::model::tensor::Tensor::from_vec(d, hidden.shape.clone())
    };
    let l2_pre = hidden_scaled.data.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let max_pre = hidden_scaled.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_pre = hidden_scaled.data.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!("emb(scaled)  L2={l2_pre:.3}  min={min_pre:.4}  max={max_pre:.4}");

    // Load layer 0 weights — pull out attn_norm.weight.
    let mut layer = engine.model.load_layer(0).expect("layer 0");
    let w = layer
        .get("input_layernorm.weight")
        .or_else(|| layer.get("attn_norm.weight"))
        .expect("missing attn_norm.weight");
    let norm_inplace = {
        let mut w_owned = w.clone();
        w_owned.materialize_data();
        w_owned
    };
    let w_l2 = norm_inplace.data.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let w_n = norm_inplace.data.len();
    let w_first6: Vec<f32> = norm_inplace.data.iter().take(6).cloned().collect();
    eprintln!(
        "attn_norm.weight n={w_n} L2={w_l2:.3} head={:?}",
        w_first6
    );

    // Cross-check: list attn_norm.weight metadata via public API.
    eprintln!("--- tensor metadata ---");
    if let Some(info) = engine
        .model
        .file
        .get_tensor_info("blk.0.attn_norm.weight")
    {
        eprintln!(
            "blk.0.attn_norm.weight  typ={} dims={:?} offset={}",
            info.typ, info.dimensions, info.offset
        );
    } else {
        eprintln!("blk.0.attn_norm.weight NOT in raw GGUF");
    }

    // Apply RMSNorm with the same eps the engine uses.
    let out = gemma_rms_norm(&hidden_scaled, &norm_inplace, engine.gemma_norm_eps);
    let l2_post = out.data.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let max_post = out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_post = out.data.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!(
        "rms_norm(out) L2={l2_post:.3}  min={min_post:.4}  max={max_post:.4}  ratio_post/pre={}",
        l2_post / l2_pre
    );
}
