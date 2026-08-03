use leafcutter::inference::engine::Engine;
use leafcutter::model::loader::GGUFModel;
use leafcutter::model::tensor::Tensor;
use leafcutter::kernels;

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|&v| v*v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w.iter()).map(|(&a, &b)| a * s * b).collect()
}

fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

fn main() {
    let path = "/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf";
    let model = GGUFModel::load(path).unwrap();

    // Use layer 0. Get the FFN weights.
    let w_gate = model.file.get_tensor_info("blk.0.ffn_gate.weight").unwrap();
    let w_up = model.file.get_tensor_info("blk.0.ffn_up.weight").unwrap();
    let w_down = model.file.get_tensor_info("blk.0.ffn_down.weight").unwrap();
    eprintln!("FFN shapes: gate={:?} up={:?} down={:?}", w_gate.dimensions, w_up.dimensions, w_down.dimensions);

    // Load layer 0 weights
    let layer = model.load_layer(0).unwrap();
    let gate_t = layer.get("mlp.gate_proj.weight").unwrap();
    let up_t = layer.get("mlp.up_proj.weight").unwrap();
    let down_t = layer.get("mlp.down_proj.weight").unwrap();
    eprintln!("Engine FFN tensor shapes: gate={:?} up={:?} down={:?}", gate_t.shape, up_t.shape, down_t.shape);

    // Create a simple input vector (post-norm of layer 0)
    // For now, just create a small vec and run engine ffn_forward
    let hidden_size = 4096;
    let intermediate = gate_t.shape[1]; // assume [in, out] for FFN
    eprintln!("intermediate size: {}", intermediate);

    // Build a simple test input: all 0.1
    let x: Vec<f32> = (0..hidden_size).map(|i| 0.1 + 0.001 * (i as f32)).collect();
    let xt = Tensor::from_vec(x.clone(), vec![1, hidden_size]);

    // Run engine FFN
    let engine_ffn_out = Engine::ffn_forward(&xt, &layer).expect("ffn");
    eprintln!("Engine FFN output l2 = {:.4}", engine_ffn_out.data.iter().map(|v| v*v).sum::<f32>().sqrt());

    // Reference: do it manually
    // Gate = x @ gate_w (matmul), SiLU, * up = (x @ up_w), then @ down_w
    // We need to dequantize the FFN weights. They are Q6_K based on earlier check.
    // For the reference, just verify the engine produces reasonable output:
    let max = engine_ffn_out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = engine_ffn_out.data.iter().cloned().fold(f32::INFINITY, f32::min);
    let nan = engine_ffn_out.data.iter().filter(|&&v| v.is_nan()).count();
    eprintln!("Engine FFN output: min={:.4} max={:.4} nan={}", min, max, nan);
}
