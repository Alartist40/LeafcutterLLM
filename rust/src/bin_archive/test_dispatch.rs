fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "../models/Qwen3.5-2B-BF16.gguf".to_string());
    let engine = leafcutter::inference::engine::Engine::load(&path).expect("Failed to load engine");
    
    for layer_idx in 0..engine.config.num_hidden_layers.min(4) {
        let weights = engine.model.load_layer(layer_idx).unwrap();
        let has_q = weights.contains_key("self_attn.q_proj.weight");
        let has_attn_q = weights.contains_key("attn_q.weight");
        let has_ssm_alpha = weights.contains_key("ssm_alpha.weight");
        let has_qkv = weights.contains_key("self_attn.qkv_proj.weight");
        let has_ssm_out = weights.contains_key("ssm_out.weight");
        
        let has_standard_attn = has_q || has_attn_q;
        let has_deltanet = has_ssm_alpha || has_qkv;
        let has_ssm = has_ssm_out && !has_deltanet;
        
        println!("layer {}: has_q={}, has_attn_q={}, has_ssm_alpha={}, has_qkv={}, has_ssm_out={}", 
            layer_idx, has_q, has_attn_q, has_ssm_alpha, has_qkv, has_ssm_out);
        println!("  -> has_standard_attn={}, has_deltanet={}, has_ssm={}", 
            has_standard_attn, has_deltanet, has_ssm);
    }
}
