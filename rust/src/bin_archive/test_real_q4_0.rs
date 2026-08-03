use leafcutter::model::gguf::GGUFile;
use leafcutter::kernels::q4_0::Matrix as Q4Matrix;
use leafcutter::kernels::int8_gemm::{q4_0_matmul_transposed_b, q4_0_matmul_via_dequant};

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = GGUFile::open(&path).unwrap();
    let prefix = "blk.0.";
    
    for t in &file.tensors {
        if !t.name.starts_with(prefix) { continue; }
        if t.typ != 2 { continue; } // Q4_0 only
        
        let info = file.get_tensor_info(&t.name).unwrap();
        let shape_gguf: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
        let shape_data = vec![shape_gguf[1], shape_gguf[0]]; // [outer, inner] = [n, k]
        let raw = file.get_tensor_raw(&t.name).unwrap();
        
        let q4 = Q4Matrix {
            rows: shape_data[0], // n
            cols: shape_data[1], // k
            blocks: leafcutter::kernels::q4_0::blocks_from_bytes(raw),
        };
        
        let k = shape_gguf[0];
        let n = shape_gguf[1];
        let m = 2;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01 - 0.5).collect();
        
        // Reference: dequantize [n, k] to f32, transpose to [k, n], then matmul
        let deq_nk = q4.dequantize(); // [n, k] row-major
        let mut deq_kn = vec![0.0f32; k * n];
        for i in 0..n {
            for j in 0..k {
                deq_kn[j * n + i] = deq_nk[i * k + j];
            }
        }
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * deq_kn[l * n + j];
                }
                expected[i * n + j] = sum;
            }
        }
        
        // Test transposed_b path (what Tensor::matmul actually uses)
        let mut c = vec![0.0f32; m * n];
        q4_0_matmul_transposed_b(&a, &q4, &mut c, m, k, n);
        
        let max_diff: f32 = c.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let dot: f32 = c.iter().zip(expected.iter()).map(|(a, b)| a * b).sum();
        let norm_c: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_e: f32 = expected.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        println!("{}: shape={:?} max_diff={:.6} cos_sim={:.6}", t.name, shape_gguf, max_diff, dot / (norm_c * norm_e));
    }
}
