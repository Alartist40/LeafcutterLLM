use leafcutter::model::tensor::Tensor;
use leafcutter::kernels::iq4_nl::{Block, Matrix};

fn main() {
    let m = 2;
    let k = 64;
    let n = 128;
    let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let a_tensor = Tensor::from_vec(a.clone(), vec![m, k]);
    
    // Create blocks for a [k, n] matrix in row-major order
    let bpr = n / 32;
    let mut blocks = Vec::with_capacity(k * bpr);
    for row in 0..k {
        for _b in 0..bpr {
            let scale = 0.01f32 * ((row % 8) + 1) as f32;
            let mut qs = [0u8; 16];
            for qi in 0..16 {
                let low = ((qi + row) % 16) as u8;
                let high = ((qi + row + 4) % 16) as u8;
                qs[qi] = (high << 4) | low;
            }
            blocks.push(Block { scale, qs });
        }
    }
    
    // loader.rs creates Matrix with rows=shape_data[0]=n, cols=shape_data[1]=k
    // shape_gguf = [k, n], shape_data = [n, k]
    let b_matrix = Matrix { rows: n, cols: k, blocks };
    let b_tensor = Tensor::from_iq4_nl_only(b_matrix.clone(), vec![k, n]);
    
    // Tensor matmul
    let c_tensor = a_tensor.matmul(&b_tensor);
    
    // Reference: dequantize the matrix and do f32 matmul
    // b_matrix.dequantize() returns [rows, cols] = [n, k] in row-major order
    let b_deq = b_matrix.dequantize();
    
    // We need the weight as [k, n] for reference matmul
    // b_deq[i*n + j] where i is row in [n, k] matrix
    // For weight W[k, n], W[row_k, col_n] = b_deq[col_n * k + row_k]
    let mut c_ref = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                // W[l, j] is at b_deq[j * k + l]
                sum += a[i * k + l] * b_deq[j * k + l];
            }
            c_ref[i * n + j] = sum;
        }
    }
    
    let max_diff: f32 = c_tensor.data.iter().zip(c_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let dot: f32 = c_tensor.data.iter().zip(c_ref.iter()).map(|(a, b)| a * b).sum();
    let norm_c: f32 = c_tensor.data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_e: f32 = c_ref.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("max_diff: {:.6}", max_diff);
    println!("cos_sim: {:.6}", dot / (norm_c * norm_e));
}
