use leafcutter::inference::engine::Engine;
use leafcutter::model::tensor::Tensor;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "../models/Qwen3.5-0.8B-Q4_0.gguf".to_string());
    let mut engine = Engine::load(&path).expect("Failed to load engine");
    let tokens = vec![17usize, 10, 17, 28];
    
    let mut hidden = engine.embed_lookup_mmap(&tokens).expect("embed_lookup_mmap failed");
    
    // Dump embedding (last token)
    let emb_last = &hidden.data[(tokens.len()-1) * hidden.shape[1]..];
    println!("EMBED {:?}", emb_last);
    
    for layer_idx in 0..engine.config.num_hidden_layers {
        let layer_weights = engine.model.load_layer(layer_idx).unwrap();
        let pre_norm_weight = layer_weights.get("input_layernorm.weight")
            .or_else(|| layer_weights.get("attn_norm.weight"))
            .expect("Missing pre-norm");
        let normed = hidden.rms_norm(pre_norm_weight, engine.config.norm_eps);
        
        let has_standard_attn = layer_weights.contains_key("self_attn.q_proj.weight")
            || layer_weights.contains_key("attn_q.weight");
        let has_deltanet = layer_weights.contains_key("ssm_alpha.weight")
            || layer_weights.contains_key("self_attn.qkv_proj.weight");
        
        let layer_out = if has_standard_attn {
            let mut kv_cache = leafcutter::cache::KVCache::new(1);
            leafcutter::inference::attention::attention_forward(
                &normed, &layer_weights, &engine.attn_params, &mut kv_cache, layer_idx, 0)
        } else if has_deltanet {
            let mut cache = leafcutter::cache::deltanet_state::DeltaNetStateCache::new();
            leafcutter::inference::deltanet::deltanet_forward(
                &normed, &layer_weights, &engine.deltanet_params, &mut cache, layer_idx)
        } else {
            Tensor::from_vec(vec![0.0f32; hidden.data.len()], hidden.shape.clone())
        };
        
        let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
            .or_else(|| layer_weights.get("ffn_norm.weight"))
            .expect("Missing post-norm");
        let post_norm = hidden.rms_norm(post_norm_weight, engine.config.norm_eps);
        let ffn_out = leafcutter::inference::engine::Engine::ffn_forward(&post_norm, &layer_weights).expect("ffn_forward failed");
        
        hidden = hidden.add(&layer_out);
        hidden = hidden.add(&ffn_out);
        
        let last = &hidden.data[(tokens.len()-1) * hidden.shape[1]..];
        println!("LAYER{} {:?}", layer_idx, last);
    }
    
    let final_norm = engine.special_weights.get("model.norm.weight").unwrap();
    hidden = hidden.rms_norm(final_norm, engine.config.norm_eps);
    let last = &hidden.data[(tokens.len()-1) * hidden.shape[1]..];
    println!("FINAL {:?}", last);
}
