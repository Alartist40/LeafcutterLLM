use leafcutter::model::loader::GGUFModel;

fn main() {
    let model = GGUFModel::load("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
    for layer_idx in 0..3 {
        let weights = model.load_layer(layer_idx).unwrap();
        if let Some(w) = weights.get("input_layernorm.weight") {
            let min = w.data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = w.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = w.data.iter().sum::<f32>() / w.data.len() as f32;
            println!("Layer {} input_layernorm: min={:.4} max={:.4} mean={:.4}", layer_idx, min, max, mean);
        }
        if let Some(w) = weights.get("post_attention_layernorm.weight") {
            let min = w.data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = w.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = w.data.iter().sum::<f32>() / w.data.len() as f32;
            println!("Layer {} post_attention_layernorm: min={:.4} max={:.4} mean={:.4}", layer_idx, min, max, mean);
        }
    }
    let special = model.load_special().unwrap();
    if let Some(w) = special.get("model.norm.weight") {
        let min = w.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = w.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = w.data.iter().sum::<f32>() / w.data.len() as f32;
        println!("Final norm: min={:.4} max={:.4} mean={:.4}", min, max, mean);
    }
}
