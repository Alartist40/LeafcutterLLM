use leafcutter::model::gguf::GGUFile;
use leafcutter::kernels::q4_0::Matrix as Q4Matrix;
use leafcutter::kernels::q4_0::blocks_from_bytes;
use leafcutter::kernels::int8_gemm::q4_0_matmul_transposed_b;

fn main() {
    let model_path = std::env::args().nth(1)
        .unwrap_or_else(|| "../models/Qwen3.5-0.8B-Q4_0.gguf".to_string());

    // Load actual input from native dump
    let input_bytes = std::fs::read("native_l0_input_pre_norm_4096.bin").expect("read input");
    let input_f32: Vec<f32> = input_bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    println!("Input len: {} (expected 4096 = 4*1024)", input_f32.len());

    // Load actual native output
    let native_out_bytes = std::fs::read("native_l0_qkv_proj_s4.bin").expect("read native output");
    let native_out: Vec<f32> = native_out_bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    println!("Native output len: {} (expected 24576 = 4*6144)", native_out.len());

    // Load Q4_0 weight
    let file = GGUFile::open(&model_path).expect("open gguf");
    let info = file.get_tensor_info("blk.0.attn_qkv.weight").expect("find");
    let raw_bytes = file.get_tensor_raw("blk.0.attn_qkv.weight").expect("read");
    let inner = info.dimensions[0] as usize;
    let outer = info.dimensions[1] as usize;

    let q4mat = Q4Matrix {
        rows: outer,
        cols: inner,
        blocks: blocks_from_bytes(raw_bytes),
    };

    // Reference f32 matmul using dequantized weight
    let weight_f32 = q4mat.dequantize();
    let m = 4;
    let k = inner;
    let n = outer;
    let mut ref_out = vec![0.0f32; m * n];
    for i in 0..m {
        for o in 0..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += input_f32[i * k + kk] * weight_f32[o * k + kk];
            }
            ref_out[i * n + o] = sum;
        }
    }

    // Fast Q4_0 matmul
    let mut fast_out = vec![0.0f32; m * n];
    q4_0_matmul_transposed_b(&input_f32, &q4mat, &mut fast_out, m, k, n);

    // Compare ref vs native
    let mae_ref_native = ref_out.iter().zip(native_out.iter()).map(|(a,b)| (a-b).abs()).sum::<f32>() / ref_out.len() as f32;
    let cos_ref_native = {
        let mut dot = 0.0f32; let mut a_sq = 0.0f32; let mut b_sq = 0.0f32;
        for i in 0..ref_out.len() { dot += ref_out[i] * native_out[i]; a_sq += ref_out[i] * ref_out[i]; b_sq += native_out[i] * native_out[i]; }
        dot / (a_sq.sqrt() * b_sq.sqrt() + 1e-10)
    };
    println!("\nRef vs Native: MAE={:.6} CosSim={:.6}", mae_ref_native, cos_ref_native);

    // Compare ref vs fast
    let mae_ref_fast = ref_out.iter().zip(fast_out.iter()).map(|(a,b)| (a-b).abs()).sum::<f32>() / ref_out.len() as f32;
    let cos_ref_fast = {
        let mut dot = 0.0f32; let mut a_sq = 0.0f32; let mut b_sq = 0.0f32;
        for i in 0..ref_out.len() { dot += ref_out[i] * fast_out[i]; a_sq += ref_out[i] * ref_out[i]; b_sq += fast_out[i] * fast_out[i]; }
        dot / (a_sq.sqrt() * b_sq.sqrt() + 1e-10)
    };
    println!("Ref vs Fast:   MAE={:.6} CosSim={:.6}", mae_ref_fast, cos_ref_fast);

    // Compare fast vs native
    let mae_fast_native = fast_out.iter().zip(native_out.iter()).map(|(a,b)| (a-b).abs()).sum::<f32>() / fast_out.len() as f32;
    let cos_fast_native = {
        let mut dot = 0.0f32; let mut a_sq = 0.0f32; let mut b_sq = 0.0f32;
        for i in 0..fast_out.len() { dot += fast_out[i] * native_out[i]; a_sq += fast_out[i] * fast_out[i]; b_sq += native_out[i] * native_out[i]; }
        dot / (a_sq.sqrt() * b_sq.sqrt() + 1e-10)
    };
    println!("Fast vs Native: MAE={:.6} CosSim={:.6}", mae_fast_native, cos_fast_native);
}
