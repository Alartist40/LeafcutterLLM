use leafcutter::inference::engine::Engine;

fn print_stats(label: &str, data: &[f32]) {
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let abs_mean: f32 = data.iter().map(|&x| x.abs()).sum::<f32>() / data.len() as f32;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("{} | mean={:.6} | abs_mean={:.6} | std={:.6} | min={:.6} | max={:.6}",
        label, mean, abs_mean, std, min, max);
}

fn main() {
    let path = std::env::args().nth(1).expect("Usage: debug_layer0 <model.gguf>");
    let mut engine = Engine::load(&path).unwrap();
    let tokens = vec![17usize, 10, 17, 28];
    let hidden = engine.embed_lookup_mmap(&tokens).expect("embed_lookup_mmap failed");
    print_stats("embed", &hidden.data);
    
    let layer_weights = engine.model.load_layer(0).unwrap();
    let pre_norm_weight = layer_weights.get("input_layernorm.weight")
        .or_else(|| layer_weights.get("attn_norm.weight"))
        .expect("Missing pre-norm");
    let pre_norm = hidden.rms_norm(pre_norm_weight, engine.config.norm_eps);
    print_stats("pre_norm", &pre_norm.data);
    
    let deltanet_out = leafcutter::inference::deltanet::deltanet_forward(
        &pre_norm, &layer_weights, &engine.deltanet_params, &mut engine.deltanet_cache, 0);
    print_stats("deltanet_out", &deltanet_out.data);
    
    let hidden_after = hidden.add(&deltanet_out);
    print_stats("hidden_after_delta", &hidden_after.data);
    
    let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
        .or_else(|| layer_weights.get("ffn_norm.weight"))
        .expect("Missing post-norm");
    let post_norm = hidden_after.rms_norm(post_norm_weight, engine.config.norm_eps);
    print_stats("post_norm", &post_norm.data);
    
    let ffn_out = leafcutter::inference::engine::Engine::ffn_forward(&post_norm, &layer_weights).expect("ffn_forward failed");
    print_stats("ffn_out", &ffn_out.data);
    
    let hidden_final = hidden_after.add(&ffn_out);
    print_stats("hidden_final", &hidden_final.data);
}
