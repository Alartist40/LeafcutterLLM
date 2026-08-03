use leafcutter::inference::engine::Engine;
use leafcutter::model::tensor::Tensor;

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn read_layer(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect(&format!("Failed to read {}", path));
    let mut vec = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        vec.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    vec
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "../models/Qwen3.5-0.8B-Q4_0.gguf".to_string());
    let mut engine = Engine::load(&path).expect("Failed to load engine");
    let tokens = vec![17usize, 10, 17, 28];
    
    let mut hidden = engine.embed_lookup_mmap(&tokens).expect("embed_lookup_mmap failed");
    let native_emb = hidden.data[(tokens.len()-1) * hidden.shape[1]..].to_vec();
    let hf_emb = read_layer("../hf_layer_00.bin");
    println!("EMBED  | cos_sim={:.6} | max_diff={:.6}", cos_sim(&native_emb, &hf_emb), max_diff(&native_emb, &hf_emb));
    
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
        
        let native = hidden.data[(tokens.len()-1) * hidden.shape[1]..].to_vec();
        let hf = read_layer(&format!("../hf_layer_{:02}.bin", layer_idx + 1));
        println!("LAYER{:2} | cos_sim={:.6} | max_diff={:.6} | type={}", 
            layer_idx, cos_sim(&native, &hf), max_diff(&native, &hf),
            if has_standard_attn { "ATTN" } else { "DELTA" });
    }
    
    let final_norm = engine.special_weights.get("model.norm.weight").unwrap();
    hidden = hidden.rms_norm(final_norm, engine.config.norm_eps);
    let native = hidden.data[(tokens.len()-1) * hidden.shape[1]..].to_vec();
    let hf = read_layer("../hf_layer_24.bin");
    println!("FINAL  | cos_sim={:.6} | max_diff={:.6}", cos_sim(&native, &hf), max_diff(&native, &hf));
    
    // Compare logits
    let hf_logits = read_layer("../hf_logits.bin");
    let native_logits = engine.forward(&tokens);
    let mut max_logit_diff = 0.0f32;
    let mut logit_cos_sim_num = 0.0f32;
    let mut logit_norm_a = 0.0f32;
    let mut logit_norm_b = 0.0f32;
    for i in 0..hf_logits.len() {
        max_logit_diff = max_logit_diff.max((native_logits[i] - hf_logits[i]).abs());
        logit_cos_sim_num += native_logits[i] * hf_logits[i];
        logit_norm_a += native_logits[i] * native_logits[i];
        logit_norm_b += hf_logits[i] * hf_logits[i];
    }
    println!("LOGITS | cos_sim={:.6} | max_diff={:.6}", 
        logit_cos_sim_num / (logit_norm_a.sqrt() * logit_norm_b.sqrt()), max_logit_diff);
}
