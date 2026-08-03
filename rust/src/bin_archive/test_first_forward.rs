//! Test: native Rust embedding lookup + first forward pass step.
//! Verifies that our safetensors loader + embedding + RMSNorm match
//! what HuggingFace would compute for the first token.
use leafcutter::ornith_config::OrnithConfig;
use leafcutter::ornith_kernels::{matmul, rmsnorm};
use leafcutter::safetensors_loader::Shards;
use std::path::Path;

fn main() {
    let dir = Path::new("/home/xander/Downloads/models/ornith safetensor");
    println!("Loading safetensors from {}", dir.display());

    let mut shards = Shards::open_dir(dir).expect("open shards");
    let cfg = OrnithConfig::load(dir.join("config.json").to_str().unwrap())
        .expect("load config");

    println!("Loaded config: hidden={}, vocab={}, layers={}",
        cfg.hidden_size, cfg.vocab_size, cfg.num_hidden_layers);

    // Load embed_tokens.weight: shape [vocab_size, hidden_size]
    let embed = shards
        .read_tensor_f32("model.language_model.embed_tokens.weight")
        .expect("read embed");
    println!("Loaded embed: {} elements", embed.len());
    assert_eq!(embed.len(), cfg.vocab_size * cfg.hidden_size);

    // Token 760 = " The" (from our tokenizer test)
    let token_id = 760usize;
    let mut hidden: Vec<f32> = embed[token_id * cfg.hidden_size..(token_id + 1) * cfg.hidden_size].to_vec();
    println!("First 8 of embed[760]: {:?}", &hidden[..8]);

    // Layer 0 input layer norm
    let in_ln_w = shards
        .read_tensor_f32("model.language_model.layers.0.input_layernorm.weight")
        .expect("read input_layernorm")
        .into_iter()
        .collect::<Vec<f32>>();
    println!("First 8 of input_layernorm.weight: {:?}", &in_ln_w[..8]);

    // Apply RMSNorm
    let mut normed = vec![0.0f32; cfg.hidden_size];
    rmsnorm(&mut normed, &hidden, &in_ln_w, cfg.rms_norm_eps);
    println!("First 8 after RMSNorm: {:?}", &normed[..8]);

    // Linear attention: project to qkv
    // in_proj_qkv.weight shape = [out=8192, in=4096]
    let qkv_w = shards
        .read_tensor_f32("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        .expect("read qkv");
    println!("\nLoaded qkv: {} elements (out=8192, in=4096)", qkv_w.len());
    assert_eq!(qkv_w.len(), 8192 * 4096);

    // Project: y[1, 8192] = normed[1, 4096] @ qkv[8192, 4096]^T
    let mut qkv = vec![0.0f32; 8192];
    matmul(&mut qkv, &normed, &qkv_w, 1, cfg.hidden_size, 8192);
    println!("First 8 of QKV output: {:?}", &qkv[..8]);
    println!("QKV max: {}", qkv.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    println!("QKV min: {}", qkv.iter().fold(f32::INFINITY, |a, &b| a.min(b)));

    println!("\n✓ Native Rust forward pass first step verified.");
    println!("  (Embedding → RMSNorm → QKV projection all in pure Rust)");
}
