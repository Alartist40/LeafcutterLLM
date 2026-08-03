use leafcutter::inference::engine::Engine;
use leafcutter::model::tensor::Tensor;

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "../models/Qwen3.5-2B-BF16.gguf".to_string());
    let mut engine = Engine::load(&path).expect("Failed to load engine");
    let tokens = vec![17usize, 10, 17, 28];
    
    let mut hidden = engine.embed_lookup_mmap(&tokens).expect("embed_lookup_mmap failed");
    let layer_weights = engine.model.load_layer(0).unwrap();
    let pre_norm_weight = layer_weights.get("input_layernorm.weight").unwrap();
    let normed = hidden.rms_norm(pre_norm_weight, engine.config.norm_eps);
    
    // Test deltanet_forward
    let mut cache1 = leafcutter::cache::deltanet_state::DeltaNetStateCache::new();
    let out_deltanet = leafcutter::inference::deltanet::deltanet_forward(
        &normed, &layer_weights, &engine.deltanet_params, &mut cache1, 0);
    
    // Test ssm_forward
    let mut cache2 = leafcutter::cache::ssm_state::SSMStateCache::new();
    let out_ssm = leafcutter::inference::ssm::ssm_forward(
        &normed, &layer_weights, &engine.ssm_config, &mut cache2, 0);
    
    println!("deltanet shape: {:?}", out_deltanet.shape);
    println!("ssm shape: {:?}", out_ssm.shape);
    
    let cs = cos_sim(&out_deltanet.data, &out_ssm.data);
    let md: f32 = out_deltanet.data.iter().zip(out_ssm.data.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("cos_sim={:.6} max_diff={:.6}", cs, md);
}
