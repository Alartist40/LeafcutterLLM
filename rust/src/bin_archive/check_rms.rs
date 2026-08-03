use leafcutter::inference::engine::Engine;

fn rms(data: &[f32]) -> f32 {
    let mean_sq = data.iter().map(|v| v * v).sum::<f32>() / data.len() as f32;
    mean_sq.sqrt()
}

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let mut engine = Engine::load(path).unwrap();
    let tokens = vec![9906usize];
    
    let mut hidden = engine.embed_lookup_mmap(&tokens).expect("embed_lookup_mmap failed");
    println!("embed RMS: {:.4}", rms(&hidden.data));
    
    for layer_idx in 0..engine.config.num_hidden_layers {
        let layer_weights = engine.model.load_layer(layer_idx).unwrap();
        
        let pre_norm_weight = layer_weights.get("input_layernorm.weight")
            .or_else(|| layer_weights.get("attn_norm.weight"))
            .expect("Missing pre-norm");
        let pre_norm = hidden.rms_norm(pre_norm_weight, 1e-5);
        println!("Layer {} pre_norm RMS: {:.4}", layer_idx, rms(&pre_norm.data));
        
        let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
            .or_else(|| layer_weights.get("ffn_norm.weight"))
            .expect("Missing post-norm");
        let post_norm = hidden.rms_norm(post_norm_weight, 1e-5);
        println!("Layer {} post_norm RMS: {:.4}", layer_idx, rms(&post_norm.data));
        
        let has_standard_attn = layer_weights.contains_key("self_attn.q_proj.weight")
            || layer_weights.contains_key("attn_q.weight");
        if has_standard_attn {
            let attn_out = leafcutter::inference::attention::attention_forward(
                &pre_norm, &layer_weights, &engine.attn_params, &mut engine.kv_cache, layer_idx, engine.seq_offset);
            println!("Layer {} attn_out RMS: {:.4}", layer_idx, rms(&attn_out.data));
            hidden = hidden.add(&attn_out);
        }
        
        let ffn_out = leafcutter::inference::engine::Engine::ffn_forward(&post_norm, &layer_weights).expect("ffn_forward failed");
        println!("Layer {} ffn_out RMS: {:.4}", layer_idx, rms(&ffn_out.data));
        hidden = hidden.add(&ffn_out);
        
        println!("Layer {} hidden RMS: {:.4}", layer_idx, rms(&hidden.data));
    }
}
