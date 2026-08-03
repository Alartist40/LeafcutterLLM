use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: test <model.gguf>");
    let file = GGUFile::open(&path).expect("Failed to open GGUF");

    for name in &["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"] {
        let info = file.get_tensor_info(name).expect("Missing tensor");
        let raw = file.get_tensor_raw(name).expect("Missing raw data");

        let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
        let q4 = leafcutter::kernels::iq4_nl::Matrix {
            rows: shape[0],
            cols: shape[1],
            blocks: leafcutter::kernels::iq4_nl::blocks_from_bytes(raw),
        };

        let mut input = vec![0.0f32; shape[0]];
        input[0] = 1.0;
        let mut output = vec![0.0f32; shape[1]];
        leafcutter::kernels::iq4_nl_gemm::iq4_nl_matmul(&input, &q4, &mut output, 1, shape[0], shape[1]);

        let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = output.iter().sum::<f32>() / output.len() as f32;
        let abs_mean = output.iter().map(|x| x.abs()).sum::<f32>() / output.len() as f32;
        println!("{}: shape={:?} min={:.6} max={:.6} mean={:.8} abs_mean={:.6}", name, shape, min, max, mean, abs_mean);
    }
}
