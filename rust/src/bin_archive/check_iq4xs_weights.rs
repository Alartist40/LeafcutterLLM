use leafcutter::model::loader::GGUFModel;

fn main() {
    let model = GGUFModel::load("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
    
    // Find an IQ4_XS layer
    for layer_idx in 0..28 {
        let weights = model.load_layer(layer_idx).unwrap();
        if let Some(q_proj) = weights.get("self_attn.q_proj.weight") {
            if q_proj.data.len() > 0 { // dequantized (IQ4_XS fallback)
                let min = q_proj.data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = q_proj.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mean = q_proj.data.iter().sum::<f32>() / q_proj.data.len() as f32;
                let abs_mean = q_proj.data.iter().map(|v| v.abs()).sum::<f32>() / q_proj.data.len() as f32;
                println!("Layer {} q_proj (dequantized): shape={:?}, min={:.4}, max={:.4}, mean={:.6}, abs_mean={:.6}", 
                    layer_idx, q_proj.shape, min, max, mean, abs_mean);
            }
        }
    }
}
