use leafcutter::model::loader::GGUFModel;
use leafcutter::model::tensor::Tensor;

fn main() {
    let model = GGUFModel::load("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
    let weights = model.load_layer(0).unwrap();
    
    let hidden_size = 3072;
    let x = Tensor::from_vec(vec![0.1f32; hidden_size], vec![1, hidden_size]);
    let weight = weights.get("input_layernorm.weight").unwrap();
    
    let out = x.rms_norm(weight, 1e-5);
    
    // Reference RMS norm
    let mean_sq: f32 = x.data.iter().map(|&v| v * v).sum::<f32>() / hidden_size as f32;
    let scale = 1.0 / (mean_sq + 1e-5).sqrt();
    let ref_data: Vec<f32> = x.data.iter().zip(weight.data.iter())
        .map(|(&x, &w)| x * scale * w)
        .collect();
    
    println!("out   min={:.4} max={:.4} mean={:.6}", 
        out.data.iter().cloned().fold(f32::INFINITY, f32::min),
        out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        out.data.iter().sum::<f32>() / out.data.len() as f32);
    println!("ref   min={:.4} max={:.4} mean={:.6}", 
        ref_data.iter().cloned().fold(f32::INFINITY, f32::min),
        ref_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        ref_data.iter().sum::<f32>() / ref_data.len() as f32);
    
    let mut max_diff = 0.0f32;
    for i in 0..out.data.len() {
        let diff = (out.data[i] - ref_data[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("max_diff: {:.6}", max_diff);
}
