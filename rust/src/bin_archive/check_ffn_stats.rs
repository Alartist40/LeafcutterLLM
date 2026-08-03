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
    let path = std::env::args().nth(1).unwrap_or_else(|| "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf".to_string());
    let mut engine = Engine::load(&path).unwrap();
    let tokens = vec![17usize, 10, 17, 28];
    let mut hidden = engine.embed_lookup_mmap(&tokens).expect("embed_lookup_mmap failed");
    
    for layer_idx in 0..engine.config.num_hidden_layers {
        let layer_weights = engine.model.load_layer(layer_idx).unwrap();
        
        let pre_norm_weight = layer_weights.get("input_layernorm.weight")
            .or_else(|| layer_weights.get("attn_norm.weight"))
            .expect("Missing pre-norm");
        let pre_norm = hidden.rms_norm(pre_norm_weight, engine.config.norm_eps);
        
        let has_standard_attn = layer_weights.contains_key("self_attn.q_proj.weight")
            || layer_weights.contains_key("attn_q.weight");
        let has_deltanet = layer_weights.contains_key("ssm_alpha.weight")
            || layer_weights.contains_key("self_attn.qkv_proj.weight");
        let has_ssm = layer_weights.contains_key("ssm_out.weight")
            && !has_deltanet;
        
        if has_deltanet {
            let deltanet_out = leafcutter::inference::deltanet::deltanet_forward(
                &pre_norm, &layer_weights, &engine.deltanet_params, &mut engine.deltanet_cache, layer_idx);
            hidden = hidden.add(&deltanet_out);
        } else if has_standard_attn {
            let attn_out = leafcutter::inference::attention::attention_forward(
                &pre_norm, &layer_weights, &engine.attn_params, &mut engine.kv_cache, layer_idx, engine.seq_offset);
            hidden = hidden.add(&attn_out);
        } else if has_ssm {
            let ssm_out = leafcutter::inference::ssm::ssm_forward(
                &pre_norm, &layer_weights, &engine.ssm_config, &mut engine.ssm_cache, layer_idx);
            hidden = hidden.add(&ssm_out);
        }
        
        print_stats(&format!("LAYER {:2} post-attn", layer_idx), &hidden.data);
        
        let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
            .or_else(|| layer_weights.get("ffn_norm.weight"))
            .expect("Missing post-norm");
        let post_norm = hidden.rms_norm(post_norm_weight, engine.config.norm_eps);
        let ffn_out = leafcutter::inference::engine::Engine::ffn_forward(&post_norm, &layer_weights).expect("ffn_forward failed");
        
        print_stats(&format!("LAYER {:2} ffn_out", layer_idx), &ffn_out.data);
        
        hidden = hidden.add(&ffn_out);
        print_stats(&format!("LAYER {:2} post-ffn", layer_idx + 1), &hidden.data);
        
        if layer_idx >= 5 {
            break;
        }
    }
}
