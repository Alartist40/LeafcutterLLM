use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: test <model.gguf>");
    let file = GGUFile::open(&path).expect("Failed to open GGUF");

    let info = file.get_tensor_info("blk.0.attn_q.weight").expect("Missing tensor");
    let raw = file.get_tensor_raw("blk.0.attn_q.weight").expect("Missing raw data");

    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    let q4 = leafcutter::kernels::iq4_nl::Matrix {
        rows: shape[0],
        cols: shape[1],
        blocks: leafcutter::kernels::iq4_nl::blocks_from_bytes(raw),
    };

    // Random input
    let mut input = vec![0.0f32; shape[0]];
    for i in 0..shape[0] {
        input[i] = ((i as f32) * 0.01 - 0.5).sin() * 0.1;
    }

    // Path 1: iq4_nl_matmul
    let mut output1 = vec![0.0f32; shape[1]];
    leafcutter::kernels::iq4_nl_gemm::iq4_nl_matmul(&input, &q4, &mut output1, 1, shape[0], shape[1]);

    // Path 2: dequantize then f32 matmul
    let b_f32 = q4.dequantize();
    let mut output2 = vec![0.0f32; shape[1]];
    leafcutter::kernels::simd::simd_matmul(&input, &b_f32, &mut output2, 1, shape[0], shape[1]);

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for i in 0..shape[1] {
        let diff = (output1[i] - output2[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    println!("Shape: {:?}", shape);
    println!("Max diff: {} at index {}", max_diff, max_diff_idx);
    println!("Output1[{}] = {}, Output2[{}] = {}", max_diff_idx, output1[max_diff_idx], max_diff_idx, output2[max_diff_idx]);

    let mean_abs1 = output1.iter().map(|x| x.abs()).sum::<f32>() / output1.len() as f32;
    let mean_abs2 = output2.iter().map(|x| x.abs()).sum::<f32>() / output2.len() as f32;
    println!("Mean abs: path1={}, path2={}", mean_abs1, mean_abs2);
}
