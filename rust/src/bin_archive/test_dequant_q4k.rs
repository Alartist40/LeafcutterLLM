use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = GGUFile::open(&path).unwrap();
    
    let info = file.get_tensor_info("token_embd.weight").unwrap();
    let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    let cols = dims[0];
    let rows = dims[1];
    let raw = file.get_tensor_raw("token_embd.weight").unwrap();
    
    // Dequantize using matrix method
    let q4k = leafcutter::kernels::q4_k::Matrix {
        rows,
        cols,
        blocks: leafcutter::kernels::q4_k::blocks_from_bytes(raw),
    };
    let full_deq = q4k.dequantize(); // [rows, cols] row-major
    
    // Dequantize using per-row method
    let mut row_deq = vec![0.0f32; cols];
    file.get_tensor_row_f32_into("token_embd.weight", 0, &mut row_deq).unwrap();
    
    // Compare first row
    let max_diff: f32 = row_deq.iter().zip(full_deq[..cols].iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let dot: f32 = row_deq.iter().zip(full_deq[..cols].iter()).map(|(a, b)| a * b).sum();
    let norm_a: f32 = row_deq.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = full_deq[..cols].iter().map(|x| x * x).sum::<f32>().sqrt();
    
    println!("Q4_K row 0 dequant: max_diff={:.6} cos_sim={:.6}", max_diff, dot / (norm_a * norm_b));
}
