use leafcutter::kernels::iq4_nl::{Block, Matrix};
use leafcutter::kernels::iq4_nl_gemm::{iq4_nl_matmul, iq4_nl_matmul_via_dequant, iq4_nl_matmul_transposed_b};

fn main() {
    let m = 2;
    let k = 4;
    let n = 32;
    let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1).collect();
    
    let bpr = n / 32;
    let mut blocks = Vec::with_capacity(k * bpr);
    for row in 0..k {
        for _b in 0..bpr {
            let scale = 0.01f32 * (row + 1) as f32;
            let mut qs = [0u8; 16];
            for qi in 0..16 {
                let low = ((qi % 8) as u8).min(15);
                let high = (((qi + 4) % 8) as u8).min(15);
                qs[qi] = (high << 4) | low;
            }
            blocks.push(Block { scale, qs });
        }
    }
    let b = Matrix { rows: k, cols: n, blocks };

    // Test non-transposed matmul
    let expected = iq4_nl_matmul_via_dequant(&a, &b, m, k, n);
    let mut c = vec![0.0f32; m * n];
    iq4_nl_matmul(&a, &b, &mut c, m, k, n);
    
    let max_diff: f32 = c.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("iq4_nl_matmul max diff: {:.6}", max_diff);
    
    // Test transposed matmul
    let bt = Matrix { rows: n, cols: k, blocks: b.blocks.clone() };
    let mut ct = vec![0.0f32; m * n];
    iq4_nl_matmul_transposed_b(&a, &bt, &mut ct, m, k, n);
    
    let max_diff_t: f32 = ct.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("iq4_nl_matmul_transposed_b max diff: {:.6}", max_diff_t);
    
    // Cosine similarity
    let dot: f32 = c.iter().zip(expected.iter()).map(|(a, b)| a * b).sum();
    let norm_c: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_e: f32 = expected.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("cos_sim: {:.6}", dot / (norm_c * norm_e));
}
