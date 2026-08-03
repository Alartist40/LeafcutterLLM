use leafcutter::inference::engine::Engine;
use leafcutter::inference::attention::{apply_rotary_emb, attention_forward};
use leafcutter::cache::KVCache;
use leafcutter::model::tensor::Tensor;

fn print_stats(name: &str, data: &[f32]) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let abs_mean = data.iter().map(|x| x.abs()).sum::<f32>() / data.len() as f32;
    println!("{}: min={:.6} max={:.6} mean={:.8} abs_mean={:.6}", name, min, max, mean, abs_mean);
}

fn main() {
    let path = std::env::args().nth(1).expect("Usage: compare_layer0 <model.gguf>");
    let mut engine = Engine::load(&path).expect("Failed to load engine");
    let tokens = vec![9906usize];

    let mut hidden = engine.embed_lookup_mmap(&tokens).expect("embed_lookup_mmap failed");
    println!("rope_theta = {}", engine.attn_params.rope_theta); print_stats("embed", &hidden.data);

    let layer_idx = 0;
    let weights = engine.model.load_layer(layer_idx).unwrap();

    let pre_norm_weight = weights.get("input_layernorm.weight")
        .or_else(|| weights.get("attn_norm.weight"))
        .expect("Missing pre-norm");
    let pre_norm = hidden.rms_norm(pre_norm_weight, 1e-5);
    print_stats("pre_norm", &pre_norm.data);

    // QKV projections directly
    let q_proj = weights.get("self_attn.q_proj.weight")
        .or_else(|| weights.get("attn_q.weight"))
        .expect("Missing q_proj");
    let k_proj = weights.get("self_attn.k_proj.weight")
        .or_else(|| weights.get("attn_k.weight"))
        .expect("Missing k_proj");
    let v_proj = weights.get("self_attn.v_proj.weight")
        .or_else(|| weights.get("attn_v.weight"))
        .expect("Missing v_proj");

    let q = pre_norm.matmul(q_proj);
    let k = pre_norm.matmul(k_proj);
    let v = pre_norm.matmul(v_proj);
    print_stats("q_proj", &q.data);
    print_stats("k_proj", &k.data);
    print_stats("v_proj", &v.data);

    // RoPE
    let mut q_tensor = Tensor::from_vec(q.data.clone(), vec![1, engine.attn_params.num_heads, engine.attn_params.head_dim]);
    let mut k_tensor = Tensor::from_vec(k.data.clone(), vec![1, engine.attn_params.num_kv_heads, engine.attn_params.kv_head_dim]);
    apply_rotary_emb(&mut q_tensor, 1, engine.attn_params.num_heads, engine.attn_params.head_dim, engine.attn_params.rope_dim, engine.attn_params.rope_theta, 0);
    apply_rotary_emb(&mut k_tensor, 1, engine.attn_params.num_kv_heads, engine.attn_params.kv_head_dim, engine.attn_params.rope_dim, engine.attn_params.rope_theta, 0);
    println!("q after rope: min={:.6} max={:.6}", q_tensor.data.iter().cloned().fold(f32::INFINITY, f32::min), q_tensor.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!("k after rope: min={:.6} max={:.6}", k_tensor.data.iter().cloned().fold(f32::INFINITY, f32::min), k_tensor.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Full attention forward
    let mut kv_cache = KVCache::new(1);
    let attn_out = attention_forward(&pre_norm, &weights, &engine.attn_params, &mut kv_cache, layer_idx, 0);
    print_stats("attn_out (after o_proj)", &attn_out.data);

    // Residual
    hidden = hidden.add(&attn_out);
    print_stats("hidden after attn residual", &hidden.data);

    // Post-norm
    let post_norm_weight = weights.get("post_attention_layernorm.weight")
        .or_else(|| weights.get("ffn_norm.weight"))
        .expect("Missing post-norm");
    let post_norm = hidden.rms_norm(post_norm_weight, 1e-5);
    print_stats("post_norm", &post_norm.data);

    // FFN
    let ffn_out = Engine::ffn_forward(&post_norm, &weights).expect("ffn_forward failed");
    print_stats("ffn_out", &ffn_out.data);

    // Residual
    hidden = hidden.add(&ffn_out);
    print_stats("hidden after ffn residual", &hidden.data);
}
