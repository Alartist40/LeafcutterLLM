use leafcutter::inference::engine::Engine;
use leafcutter::inference::attention::{attention_forward, AttentionParams};
use leafcutter::cache::KVCache;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: test_iq4_nl_attn <model.gguf>");
    let mut engine = Engine::load(&path).expect("Failed to load engine");
    let tokens = vec![9906usize];

    let mut hidden = engine.embed_lookup_mmap(&tokens).expect("embed_lookup_mmap failed");
    println!("embed: min={:.4} max={:.4} mean={:.6}",
        hidden.data.iter().cloned().fold(f32::INFINITY, f32::min),
        hidden.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        hidden.data.iter().sum::<f32>() / hidden.data.len() as f32);

    let layer_idx = 0;
    let weights = engine.model.load_layer(layer_idx).unwrap();

    let pre_norm_weight = weights.get("input_layernorm.weight")
        .or_else(|| weights.get("attn_norm.weight"))
        .expect("Missing pre-norm");
    let pre_norm = hidden.rms_norm(pre_norm_weight, 1e-5);
    println!("pre_norm: min={:.4} max={:.4} mean={:.6}",
        pre_norm.data.iter().cloned().fold(f32::INFINITY, f32::min),
        pre_norm.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        pre_norm.data.iter().sum::<f32>() / pre_norm.data.len() as f32);

    // Test q_proj matmul directly
    let q_proj = weights.get("self_attn.q_proj.weight")
        .or_else(|| weights.get("attn_q.weight"))
        .expect("Missing q_proj");
    let q = pre_norm.matmul(q_proj);
    println!("q_proj output: min={:.4} max={:.4} mean={:.6}",
        q.data.iter().cloned().fold(f32::INFINITY, f32::min),
        q.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        q.data.iter().sum::<f32>() / q.data.len() as f32);

    let mut kv_cache = KVCache::new(1);
    let attn_out = attention_forward(&pre_norm, &weights, &engine.attn_params, &mut kv_cache, layer_idx, 0);
    println!("attn_out: min={:.4} max={:.4} mean={:.6}",
        attn_out.data.iter().cloned().fold(f32::INFINITY, f32::min),
        attn_out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        attn_out.data.iter().sum::<f32>() / attn_out.data.len() as f32);
}
