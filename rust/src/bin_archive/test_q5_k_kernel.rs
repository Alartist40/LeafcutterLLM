use leafcutter::model::gguf::GGUFile;
use leafcutter::kernels::q5_k::Matrix as Q5KMatrix;
use leafcutter::kernels::q5_k::blocks_from_bytes;
use leafcutter::kernels::q5_k_gemm::q5_k_matmul_transposed_b;

fn main() {
    let path = std::env::args().nth(1).expect("Usage");
    let file = GGUFile::open(&path).expect("open");
    let info = file.get_tensor_info("blk.0.ssm_out.weight").expect("find");
    let raw = file.get_tensor_raw("blk.0.ssm_out.weight").expect("read");
    let inner = info.dimensions[0] as usize;
    let outer = info.dimensions[1] as usize;
    println!("ssm_out.weight: inner={}, outer={}, typ={}", inner, outer, info.typ);

    let q5mat = Q5KMatrix {
        rows: outer,
        cols: inner,
        blocks: blocks_from_bytes(raw),
    };
    println!("Q5KMatrix: rows={}, cols={}, blocks_per_row={}", q5mat.rows, q5mat.cols, q5mat.blocks_per_row());

    let weight_f32 = q5mat.dequantize();
    let m = 4;
    let k = inner;
    let n = outer;
    
    // Synthetic input
    let input_f32: Vec<f32> = (0..m*k).map(|i| ((i % 7) as f32 - 3.0) / 10.0).collect();
    
    // Reference
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
    
    // Fast
    let mut fast_out = vec![0.0f32; m * n];
    q5_k_matmul_transposed_b(&input_f32, &q5mat, &mut fast_out, m, k, n);
    
    let mae = ref_out.iter().zip(fast_out.iter()).map(|(a,b)| (a-b).abs()).sum::<f32>() / ref_out.len() as f32;
    let cos = {
        let mut dot = 0.0f32; let mut a_sq = 0.0f32; let mut b_sq = 0.0f32;
        for i in 0..ref_out.len() { dot += ref_out[i] * fast_out[i]; a_sq += ref_out[i] * ref_out[i]; b_sq += fast_out[i] * fast_out[i]; }
        dot / (a_sq.sqrt() * b_sq.sqrt() + 1e-10)
    };
    println!("m={}: MAE={:.6} CosSim={:.6}", m, mae, cos);
}
