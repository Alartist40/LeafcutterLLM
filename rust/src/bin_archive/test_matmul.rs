use leafcutter::model::tensor::Tensor;

fn main() {
    // Exact Llama output projection size: [1, 3072] @ [3072, 128256]
    let a = Tensor::from_vec(vec![0.1f32; 1 * 3072], vec![1, 3072]);
    let b = Tensor::from_vec(vec![0.01f32; 3072 * 128256], vec![3072, 128256]);
    println!("Starting large matmul...");
    let c = a.matmul(&b);
    println!("Done. shape={:?}, max={}", c.shape, c.data.iter().fold(0.0f32, |a, &b| a.max(b)));
}
