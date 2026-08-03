use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: test_iq4_nl_real <model.gguf>");
    let file = GGUFile::open(&path).expect("Failed to open GGUF");

    let info = file.get_tensor_info("blk.0.attn_q.weight").expect("Missing tensor");
    let raw = file.get_tensor_raw("blk.0.attn_q.weight").expect("Missing raw data");

    println!("Tensor: blk.0.attn_q.weight");
    println!("  dims: {:?}", info.dimensions);
    println!("  typ: {}", info.typ);
    println!("  raw_len: {}", raw.len());

    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    let q4 = leafcutter::kernels::iq4_nl::Matrix {
        rows: shape[0],
        cols: shape[1],
        blocks: leafcutter::kernels::iq4_nl::blocks_from_bytes(raw),
    };

    println!("  matrix rows={}, cols={}, blocks={}", q4.rows, q4.cols, q4.blocks.len());

    // Check first few block scales
    println!("  First 5 block scales: {:?}", q4.blocks[..5].iter().map(|b| b.scale).collect::<Vec<_>>());

    // Dequantize first block
    let mut deq = [0.0f32; 32];
    q4.blocks[0].dequantize(&mut deq);
    println!("  First block dequantized: {:?}", &deq[..10]);

    // Do a small matmul: input [1, 3072] with first element = 1.0, rest = 0.0
    let mut input = vec![0.0f32; 3072];
    input[0] = 1.0;
    let mut output = vec![0.0f32; 3072];

    leafcutter::kernels::iq4_nl_gemm::iq4_nl_matmul(&input, &q4, &mut output, 1, 3072, 3072);

    let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = output.iter().sum::<f32>() / output.len() as f32;
    println!("  Matmul output: min={}, max={}, mean={}", min, max, mean);
    println!("  First 10 output values: {:?}", &output[..10]);
}
