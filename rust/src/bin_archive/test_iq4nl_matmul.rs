use leafcutter::model::tensor::Tensor;

fn main() {
    // Small IQ4_NL matmul test
    let a = Tensor::from_vec(vec![0.1f32; 1 * 4096], vec![1, 4096]);
    
    // Create a fake IQ4_NL weight [4096, 4096]
    let block_count = (4096 * 4096) / 32;
    let bytes_needed = block_count * 18; // IQ4_NL block size
    let raw = vec![0u8; bytes_needed];
    let q4 = leafcutter::kernels::iq4_nl::Matrix {
        rows: 4096, cols: 4096,
        blocks: leafcutter::kernels::iq4_nl::blocks_from_bytes(&raw),
    };
    let b = Tensor::from_iq4_nl_only(q4, vec![4096, 4096]);
    
    println!("a shape={:?}, b shape={:?}", a.shape, b.shape);
    
    let c = a.matmul(&b);
    println!("c shape={:?}, mean={:.6}, max_abs={:.6}", c.shape, 
        c.data.iter().sum::<f32>() / c.data.len() as f32,
        c.data.iter().map(|&v| v.abs()).fold(0.0f32, f32::max));
}
