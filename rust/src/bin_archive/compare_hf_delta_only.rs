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
    let layer_weights = engine.model.load_layer(0).unwrap();
    let pre_norm_weight = layer_weights.get("input_layernorm.weight").unwrap();
    let normed = hidden.rms_norm(pre_norm_weight, engine.config.norm_eps);
    
    let mut cache = leafcutter::cache::deltanet_state::DeltaNetStateCache::new();
    let delta_out = leafcutter::inference::deltanet::deltanet_forward(
        &normed, &layer_weights, &engine.deltanet_params, &mut cache, 0);
    
    let native = delta_out.data[(tokens.len()-1) * delta_out.shape[1]..].to_vec();
    
    // Compute HF reference for layer 0 delta-only output
    // HF hidden_states[1] - hidden_states[0] = layer 0 output (including residual)
    // But we need just the DeltaNet output, which is (hidden_states[1] - hidden_states[0]) - ffn_out
    
    let hf_emb = read_layer("../hf_layer_00.bin");
    let hf_after_layer = read_layer("../hf_layer_01.bin");
    
    // For layer 0, we need to subtract embed and then separate delta from ffn
    // Actually, HF layer 0 output = delta_out + ffn_out (both with residuals)
    // So hf_after_layer - hf_emb = delta_out + ffn_out
    
    // Let's just compare delta_out + normed_embed vs HF after layer (approximate)
    let mut combined = vec![0.0f32; native.len()];
    for i in 0..native.len() {
        combined[i] = native[i] + normed.data[(tokens.len()-1) * normed.shape[1] + i];
    }
    
    println!("Native delta_out + normed vs HF after_layer_0:");
    println!("  cos_sim={:.6} max_diff={:.6}", cos_sim(&combined, &hf_after_layer), max_diff(&combined, &hf_after_layer));
    
    // Also compare just delta_out (before residual)
    // We don't have HF delta_out directly, but we can approximate:
    // HF delta_out ≈ hf_after_layer - hf_emb - ffn_out
    // For a rough check, let's just compare magnitudes
    println!("Native delta_out max={:.4} mean_abs={:.4}", 
        native.iter().cloned().fold(0.0f32, f32::max),
        native.iter().map(|x| x.abs()).sum::<f32>() / native.len() as f32);
}
