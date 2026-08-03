use leafcutter::model::gguf::GGUFile;
use leafcutter::kernels::q4_k::Matrix as Q4KMatrix;
use leafcutter::kernels::q4_k_gemm::q4_k_matmul_transposed_b;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = GGUFile::open(&path).unwrap();
    
    let info = file.get_tensor_info("token_embd.weight").unwrap();
    let shape_gguf: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    let shape_data = vec![shape_gguf[1], shape_gguf[0]]; // [outer, inner] = [n, k]
    let raw = file.get_tensor_raw("token_embd.weight").unwrap();
    
    let q4k = Q4KMatrix {
        rows: shape_data[0], // n
        cols: shape_data[1], // k
        blocks: leafcutter::kernels::q4_k::blocks_from_bytes(raw),
    };
    
    let k = shape_gguf[0];
    let n = shape_gguf[1];
    let m = 2;
    let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01 - 0.5).collect();
    
    // Reference: dequantize [n, k] to f32, transpose to [k, n], then matmul
    let deq_nk = q4k.dequantize(); // [n, k] row-major
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
    
    // Test transposed_b path
    let mut c = vec![0.0f32; m * n];
    q4_k_matmul_transposed_b(&a, &q4k, &mut c, m, k, n);
    
    let max_diff: f32 = c.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let dot: f32 = c.iter().zip(expected.iter()).map(|(a, b)| a * b).sum();
    let norm_c: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_e: f32 = expected.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    println!("token_embd.weight: shape={:?} max_diff={:.6} cos_sim={:.6}", shape_gguf, max_diff, dot / (norm_c * norm_e));
}
