use leafcutter::model::gguf::GGUFile;
use leafcutter::kernels::q4_0::Matrix as Q4Matrix;
use leafcutter::kernels::q4_0::blocks_from_bytes;
use leafcutter::kernels::int8_gemm::q4_0_matmul_transposed_b;

fn main() {
    let model_path = std::env::args().nth(1)
        .unwrap_or_else(|| "../models/Qwen3.5-0.8B-Q4_0.gguf".to_string());

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

    // Test with m=1 and m=4
    for m in [1, 4] {
        let input_f32: Vec<f32> = (0..m*inner).map(|i| ((i % 7) as f32 - 3.0) / 10.0).collect();
        
        // Reference f32 matmul
        let weight_f32 = q4mat.dequantize();
        let mut ref_out = vec![0.0f32; m * outer];
        for i in 0..m {
            for o in 0..outer {
                let mut sum = 0.0;
                for k in 0..inner {
                    sum += input_f32[i * inner + k] * weight_f32[o * inner + k];
                }
                ref_out[i * outer + o] = sum;
            }
        }
        
        // Fast Q4_0 matmul
        let mut fast_out = vec![0.0f32; m * outer];
        q4_0_matmul_transposed_b(&input_f32, &q4mat, &mut fast_out, m, inner, outer);
        
        let mae = ref_out.iter().zip(fast_out.iter()).map(|(a,b)| (a-b).abs()).sum::<f32>() / ref_out.len() as f32;
        let cos_sim = {
            let mut dot = 0.0f32;
            let mut a_sq = 0.0f32;
            let mut b_sq = 0.0f32;
            for i in 0..ref_out.len() {
                dot += ref_out[i] * fast_out[i];
                a_sq += ref_out[i] * ref_out[i];
                b_sq += fast_out[i] * fast_out[i];
            }
            dot / (a_sq.sqrt() * b_sq.sqrt() + 1e-10)
        };
        
        println!("m={}: MAE={:.6} CosSim={:.6}", m, mae, cos_sim);
    }
}
