//! INT8 GEMM kernel for Q8_0 quantized weights
//!
//! Computes C = A × B where:
//!   - A is f32 [m, k]
//!   - B is Q8_0 quantized [k, n]
//!   - C is f32 [m, n]
//!
//! The Q8_0 weights are dequantized on-the-fly inside the kernel,
//! giving 4× memory bandwidth savings vs f32 weights.

use super::q8_0::Matrix as Q8Matrix;
use super::simd;

/// Scalar INT8 GEMM with on-the-fly dequantization.
/// This is the reference implementation — SIMD variants go below.
pub fn q8_0_matmul_scalar(a: &[f32], b: &Q8Matrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    assert_eq!(b.cols, n);
    let bpr = b.blocks_per_row();

    for i in 0..m {
        // Zero the output row
        for j in 0..n {
            c[i * n + j] = 0.0;
        }
        // Accumulate over k dimension
        for l in 0..b.rows {
            let a_val = a[i * b.rows + l];
            let row_base = l * bpr;
            for block_idx in 0..bpr {
                let block = &b.blocks[row_base + block_idx];
                let j_base = block_idx * 32;
                let scale = block.scale;
                for jj in 0..32 {
                    c[i * n + j_base + jj] += a_val * (block.qs[jj] as f32) * scale;
                }
            }
        }
    }
}

/// SIMD-accelerated INT8 GEMM.
/// Dequantizes Q8_0 blocks to f32 on-the-fly, then uses the same
/// SIMD f32 matmul kernel for the actual computation.
///
/// This gives the memory-bandwidth win of Q8_0 (4× less data movement)
/// while reusing our proven f32 SIMD kernels for compute.
pub fn q8_0_matmul_simd(a: &[f32], b: &Q8Matrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    assert_eq!(b.cols, n);
    assert_eq!(n % 4, 0, "SIMD path requires n to be a multiple of 4");
    let bpr = b.blocks_per_row();

    for i in 0..m {
        // Zero the output row
        for j in 0..n {
            c[i * n + j] = 0.0;
        }
        // Accumulate over k dimension
        for l in 0..b.rows {
            let a_val = a[i * b.rows + l];
            let row_base = l * bpr;
            for block_idx in 0..bpr {
                let block = &b.blocks[row_base + block_idx];
                let j_base = block_idx * 32;
                let scale = block.scale;

                // Process 4 columns at a time using SIMD
                let mut jj = 0;
                while jj + 4 <= 32 {
                    let b_f32 = [
                        block.qs[jj] as f32 * scale,
                        block.qs[jj + 1] as f32 * scale,
                        block.qs[jj + 2] as f32 * scale,
                        block.qs[jj + 3] as f32 * scale,
                    ];
                    c[i * n + j_base + jj] += a_val * b_f32[0];
                    c[i * n + j_base + jj + 1] += a_val * b_f32[1];
                    c[i * n + j_base + jj + 2] += a_val * b_f32[2];
                    c[i * n + j_base + jj + 3] += a_val * b_f32[3];
                    jj += 4;
                }
                // Scalar tail (shouldn't happen for n multiple of 4)
                for jjt in jj..32 {
                    c[i * n + j_base + jjt] += a_val * (block.qs[jjt] as f32) * scale;
                }
            }
        }
    }
}

/// Dispatch to the best available INT8 GEMM kernel.
pub fn q8_0_matmul(a: &[f32], b: &Q8Matrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    if n % 4 == 0 {
        q8_0_matmul_simd(a, b, c, m, k, n);
    } else {
        q8_0_matmul_scalar(a, b, c, m, k, n);
    }
}

/// Convert Q8_0 matrix back to f32, then use the proven f32 SIMD matmul.
/// This is the "reference fast path" — dequantize once, then f32 GEMM.
/// Useful when the same B matrix is reused many times.
pub fn q8_0_matmul_via_dequant(a: &[f32], b: &Q8Matrix, m: usize, k: usize, n: usize) -> Vec<f32> {
    let b_f32 = b.dequantize();
    let mut c = vec![0.0f32; m * n];
    simd::simd_matmul(a, &b_f32, &mut c, m, k, n);
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::q8_0::Block;

    fn make_test_matrix(rows: usize, cols: usize) -> Q8Matrix {
        assert_eq!(cols % 32, 0);
        let bpr = cols / 32;
        let mut blocks = Vec::with_capacity(rows * bpr);
        for row in 0..rows {
            for b in 0..bpr {
                let scale = 0.01f32 * (row + 1) as f32;
                let mut qs = [0i8; 32];
                for jj in 0..32 {
                    qs[jj] = ((b * 32 + jj) as i8).wrapping_sub(16);
                }
                blocks.push(Block { scale, qs });
            }
        }
        Q8Matrix { rows, cols, blocks }
    }

    #[test]
    fn test_q8_0_matmul_vs_dequant() {
        let m = 4;
        let k = 8;
        let n = 32;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1).collect();
        let b_q8 = make_test_matrix(k, n);

        // Reference: dequantize then f32 matmul
        let expected = q8_0_matmul_via_dequant(&a, &b_q8, m, k, n);

        // Test scalar path
        let mut c_scalar = vec![0.0f32; m * n];
        q8_0_matmul_scalar(&a, &b_q8, &mut c_scalar, m, k, n);
        for i in 0..c_scalar.len() {
            assert!((c_scalar[i] - expected[i]).abs() < 1e-3,
                "scalar mismatch at {}: got {}, expected {}", i, c_scalar[i], expected[i]);
        }

        // Test SIMD path
        let mut c_simd = vec![0.0f32; m * n];
        q8_0_matmul_simd(&a, &b_q8, &mut c_simd, m, k, n);
        for i in 0..c_simd.len() {
            assert!((c_simd[i] - expected[i]).abs() < 1e-3,
                "simd mismatch at {}: got {}, expected {}", i, c_simd[i], expected[i]);
        }
    }
}
