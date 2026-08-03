use leafcutter::inference::engine::Engine;
use leafcutter::model::loader::GGUFModel;
use leafcutter::model::tensor::Tensor;

fn main() {
    let model = GGUFModel::load("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
    let weights = model.load_layer(0).unwrap();
    
    let hidden_size = 3072;
    let x = Tensor::from_vec(vec![0.1f32; hidden_size], vec![1, hidden_size]);
    
    let ffn_out = Engine::ffn_forward(&x, &weights).expect("ffn_forward failed");
    
    // Reference FFN: down(silu(x @ gate) * (x @ up))
    let gate = weights.get("mlp.gate_proj.weight").unwrap();
    let up = weights.get("mlp.up_proj.weight").unwrap();
    let down = weights.get("mlp.down_proj.weight").unwrap();
    
    let gate_proj = x.matmul(gate);
    let up_proj = x.matmul(up);
    let activated: Vec<f32> = gate_proj.data.iter().zip(up_proj.data.iter())
        .map(|(&g, &u)| g * (1.0 / (1.0 + (-g).exp())) * u)
        .collect();
    let fused = Tensor::from_vec(activated, gate_proj.shape.clone());
    let ref_out = fused.matmul(down);
    
    println!("ffn_out  min={:.4} max={:.4} mean={:.6}", 
        ffn_out.data.iter().cloned().fold(f32::INFINITY, f32::min),
        ffn_out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        ffn_out.data.iter().sum::<f32>() / ffn_out.data.len() as f32);
    println!("ref_out  min={:.4} max={:.4} mean={:.6}", 
        ref_out.data.iter().cloned().fold(f32::INFINITY, f32::min),
        ref_out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        ref_out.data.iter().sum::<f32>() / ref_out.data.len() as f32);
    
    let mut max_diff = 0.0f32;
    for i in 0..ffn_out.data.len() {
        let diff = (ffn_out.data[i] - ref_out.data[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("max_diff: {:.6}", max_diff);
}
