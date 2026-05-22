use leafcutter::inference::engine::Engine;
use leafcutter::model::tensor::Tensor;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf".to_string());
    let mut engine = Engine::load(&path).unwrap();
    let tokens = vec![9906usize];
    
    let mut hidden = engine.embed_lookup_mmap(&tokens);
    println!("embed: min={:.4} max={:.4} mean={:.6}", 
        hidden.data.iter().cloned().fold(f32::INFINITY, f32::min),
        hidden.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        hidden.data.iter().sum::<f32>() / hidden.data.len() as f32);
    
    for layer_idx in 0..engine.config.num_hidden_layers {
        let layer_weights = engine.model.load_layer(layer_idx).unwrap();
        
        let pre_norm_weight = layer_weights.get("input_layernorm.weight")
            .or_else(|| layer_weights.get("attn_norm.weight"))
            .expect("Missing pre-norm");
        let pre_norm = hidden.rms_norm(pre_norm_weight, 1e-5);
        
        let has_standard_attn = layer_weights.contains_key("self_attn.q_proj.weight")
            || layer_weights.contains_key("attn_q.weight");
        
        if has_standard_attn {
            let attn_out = leafcutter::inference::attention::attention_forward(
                &pre_norm, &layer_weights, &engine.attn_params, &mut engine.kv_cache, layer_idx, engine.seq_offset);
            hidden = hidden.add(&attn_out);
            println!("Layer {} attn_out: min={:.4} max={:.4} mean={:.6}", layer_idx,
                attn_out.data.iter().cloned().fold(f32::INFINITY, f32::min),
                attn_out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                attn_out.data.iter().sum::<f32>() / attn_out.data.len() as f32);
        }
        
        let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
            .or_else(|| layer_weights.get("ffn_norm.weight"))
            .expect("Missing post-norm");
        let post_norm = hidden.rms_norm(post_norm_weight, 1e-5);
        
        let ffn_out = leafcutter::inference::engine::Engine::ffn_forward(&post_norm, &layer_weights);
        hidden = hidden.add(&ffn_out);
        
        println!("Layer {} ffn_out: min={:.4} max={:.4} mean={:.6} | hidden: min={:.4} max={:.4} mean={:.6}", layer_idx,
            ffn_out.data.iter().cloned().fold(f32::INFINITY, f32::min),
            ffn_out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            ffn_out.data.iter().sum::<f32>() / ffn_out.data.len() as f32,
            hidden.data.iter().cloned().fold(f32::INFINITY, f32::min),
            hidden.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            hidden.data.iter().sum::<f32>() / hidden.data.len() as f32);
    }
    
    let final_norm = engine.special_weights.get("model.norm.weight").unwrap();
    hidden = hidden.rms_norm(final_norm, 1e-5);
    println!("final_norm: min={:.4} max={:.4} mean={:.6}", 
        hidden.data.iter().cloned().fold(f32::INFINITY, f32::min),
        hidden.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        hidden.data.iter().sum::<f32>() / hidden.data.len() as f32);
}
