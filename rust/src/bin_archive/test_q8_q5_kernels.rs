use leafcutter::model::gguf::GGUFile;
use leafcutter::kernels::q8_0::Matrix as Q8Matrix;
use leafcutter::kernels::int8_gemm::q8_0_matmul_transposed_b;
use leafcutter::kernels::q5_k::Matrix as Q5KMatrix;
use leafcutter::kernels::q5_k_gemm::q5_k_matmul_transposed_b;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = GGUFile::open(&path).unwrap();
    let prefix = "blk.0.";
    
    for t in &file.tensors {
        if !t.name.starts_with(prefix) { continue; }
        let info = file.get_tensor_info(&t.name).unwrap();
        let qtype = leafcutter::model::quant::QuantType::from_u32(info.typ).unwrap();
        let shape_gguf: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
        if shape_gguf.len() != 2 { continue; }
        let raw = file.get_tensor_raw(&t.name).unwrap();
        
        let k = shape_gguf[0];
        let n = shape_gguf[1];
        let m = 2;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01 - 0.5).collect();
        
        let mut expected = vec![0.0f32; m * n];
        
        match qtype {
            leafcutter::model::quant::QuantType::Q8_0 => {
                let shape_data = vec![shape_gguf[1], shape_gguf[0]];
                let q8 = Q8Matrix {
                    rows: shape_data[0], cols: shape_data[1],
                    blocks: leafcutter::kernels::q8_0::blocks_from_bytes(raw),
                };
                let deq_nk = q8.dequantize();
                let mut deq_kn = vec![0.0f32; k * n];
                for i in 0..n { for j in 0..k { deq_kn[j * n + i] = deq_nk[i * k + j]; } }
                for i in 0..m { for j in 0..n { let mut sum = 0.0f32; for l in 0..k { sum += a[i * k + l] * deq_kn[l * n + j]; } expected[i * n + j] = sum; } }
                
                let mut c = vec![0.0f32; m * n];
                q8_0_matmul_transposed_b(&a, &q8, &mut c, m, k, n);
                let max_diff = c.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let dot = c.iter().zip(expected.iter()).map(|(a, b)| a * b).sum::<f32>();
                let norm_c = c.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_e = expected.iter().map(|x| x * x).sum::<f32>().sqrt();
                println!("{}: Q8_0 shape={:?} max_diff={:.6} cos_sim={:.6}", t.name, shape_gguf, max_diff, dot / (norm_c * norm_e));
            }
            leafcutter::model::quant::QuantType::Q5_K => {
                let shape_data = vec![shape_gguf[1], shape_gguf[0]];
                let q5 = Q5KMatrix {
                    rows: shape_data[0], cols: shape_data[1],
                    blocks: leafcutter::kernels::q5_k::blocks_from_bytes(raw),
                };
                let deq_nk = q5.dequantize();
                let mut deq_kn = vec![0.0f32; k * n];
                for i in 0..n { for j in 0..k { deq_kn[j * n + i] = deq_nk[i * k + j]; } }
                for i in 0..m { for j in 0..n { let mut sum = 0.0f32; for l in 0..k { sum += a[i * k + l] * deq_kn[l * n + j]; } expected[i * n + j] = sum; } }
                
                let mut c = vec![0.0f32; m * n];
                q5_k_matmul_transposed_b(&a, &q5, &mut c, m, k, n);
                let max_diff = c.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let dot = c.iter().zip(expected.iter()).map(|(a, b)| a * b).sum::<f32>();
                let norm_c = c.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_e = expected.iter().map(|x| x * x).sum::<f32>().sqrt();
                println!("{}: Q5_K shape={:?} max_diff={:.6} cos_sim={:.6}", t.name, shape_gguf, max_diff, dot / (norm_c * norm_e));
            }
            _ => {}
        }
    }
}
