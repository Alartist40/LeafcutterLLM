use leafcutter::model::loader::GGUFModel;
use leafcutter::model::tensor::Tensor;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let model = GGUFModel::load(path).expect("Failed to load model");
    
    let weights = model.load_layer(0).expect("Failed to load layer");
    let qtensor = weights.get("mlp.down_proj.weight").expect("Missing down");
    println!("Shape: {:?}", qtensor.shape);
    
    let m = 1;
    let k = qtensor.shape[0];
    let n = qtensor.shape[1];
    let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001 - 0.5).collect();
    let atensor = Tensor::from_vec(a.clone(), vec![m, k]);
    
    let c1 = atensor.matmul(qtensor);
    
    let raw = model.file.get_tensor_raw("blk.0.ffn_down.weight").expect("Missing tensor");
    let info = model.file.get_tensor_info("blk.0.ffn_down.weight").expect("Missing info");
    let count: usize = info.dimensions.iter().map(|&d| d as usize).product();
    let mut deq = vec![0.0f32; count];
    leafcutter::kernels::dequantize_q6_k(raw, &mut deq);
    let dtensor = Tensor::from_vec(deq, qtensor.shape.clone());
    let c2 = atensor.matmul(&dtensor);
    
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for i in 0..c1.data.len() {
        let diff = (c1.data[i] - c2.data[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    println!("Max diff: {} at index {} (c1={}, c2={})", max_diff, max_idx, c1.data[max_idx], c2.data[max_idx]);
    println!("c1 mean abs: {:.6}", c1.data.iter().map(|x| x.abs()).sum::<f32>() / c1.data.len() as f32);
    println!("c2 mean abs: {:.6}", c2.data.iter().map(|x| x.abs()).sum::<f32>() / c2.data.len() as f32);
}
