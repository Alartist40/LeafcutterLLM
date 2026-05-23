//! IQ4_NL GEMM kernel
//!
//! Computes C = A × B where:
//!   - A is f32 [m, k]
//!   - B is IQ4_NL quantized [k, n]
//!   - C is f32 [m, n]
//!
//! IQ4_NL weights are dequantized on-the-fly inside the kernel using
//! a 32-element stack buffer (128 bytes) per block.

use super::iq4_nl::Matrix as IQ4NLMatrix;

// ============================================================================
// Scalar reference implementation
// ============================================================================

pub fn iq4_nl_matmul_scalar(a: &[f32], b: &IQ4NLMatrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    assert_eq!(b.cols, n);
    let bpr = b.blocks_per_row();

    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = 0.0;
        }
        for l in 0..b.rows {
            let a_val = a[i * b.rows + l];
            let row_base = l * bpr;
            for block_idx in 0..bpr {
                let block = &b.blocks[row_base + block_idx];
                let j_base = block_idx * 32;

                let mut deq: [f32; 32] = [0.0; 32];
                block.dequantize(&mut deq);

                for jj in 0..32 {
                    c[i * n + j_base + jj] += a_val * deq[jj];
                }
            }
        }
    }
}

// ============================================================================
// x86_64 AVX2/FMA implementation (256-bit vectors, 8×f32)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn iq4_nl_matmul_avx2_inner(a: &[f32], b: &IQ4NLMatrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    use std::arch::x86_64::*;

    assert_eq!(n % 32, 0, "AVX2 path requires n multiple of 32");
    let bpr = b.blocks_per_row();

    for i in 0..m {
        let c_row = &mut c[i * n..(i + 1) * n];
        {
            let mut j = 0;
            while j + 8 <= n {
                _mm256_storeu_ps(c_row.as_mut_ptr().add(j), _mm256_setzero_ps());
                j += 8;
            }
            for jt in j..n {
                c_row[jt] = 0.0;
            }
        }

        for l in 0..b.rows {
            let a_val = _mm256_set1_ps(*a.get_unchecked(i * b.rows + l));
            let row_base = l * bpr;

            for block_idx in 0..bpr {
                let block = &b.blocks[row_base + block_idx];
                let j_base = block_idx * 32;

                let mut deq: [f32; 32] = [0.0; 32];
                block.dequantize(&mut deq);

                let c_ptr = c_row.as_mut_ptr().add(j_base);
                let d_ptr = deq.as_ptr();

                for vec_idx in 0..4 {
                    let offset = vec_idx * 8;
                    let b_vec = _mm256_loadu_ps(d_ptr.add(offset));
                    let c_vec = _mm256_loadu_ps(c_ptr.add(offset));
                    let prod = _mm256_fmadd_ps(a_val, b_vec, c_vec);
                    _mm256_storeu_ps(c_ptr.add(offset), prod);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn iq4_nl_matmul_avx2(a: &[f32], b: &IQ4NLMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    iq4_nl_matmul_avx2_inner(a, b, c, m, k, n);
}

// ============================================================================
// ARM NEON implementation (128-bit vectors, 4×f32)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn iq4_nl_matmul_neon_inner(a: &[f32], b: &IQ4NLMatrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    use std::arch::aarch64::*;

    assert_eq!(n % 32, 0, "NEON path requires n multiple of 32");
    let bpr = b.blocks_per_row();

    for i in 0..m {
        let c_row = &mut c[i * n..(i + 1) * n];
        {
            let mut j = 0;
            while j + 4 <= n {
                vst1q_f32(c_row.as_mut_ptr().add(j), vdupq_n_f32(0.0));
                j += 4;
            }
            for jt in j..n {
                c_row[jt] = 0.0;
            }
        }

        for l in 0..b.rows {
            let a_val = vdupq_n_f32(*a.get_unchecked(i * b.rows + l));
            let row_base = l * bpr;

            for block_idx in 0..bpr {
                let block = &b.blocks[row_base + block_idx];
                let j_base = block_idx * 32;

                let mut deq: [f32; 32] = [0.0; 32];
                block.dequantize(&mut deq);

                let c_ptr = c_row.as_mut_ptr().add(j_base);
                let d_ptr = deq.as_ptr();

                for vec_idx in 0..8 {
                    let offset = vec_idx * 4;
                    let b_vec = vld1q_f32(d_ptr.add(offset));
                    let c_vec = vld1q_f32(c_ptr.add(offset));
                    let prod = vfmaq_f32(c_vec, a_val, b_vec);
                    vst1q_f32(c_ptr.add(offset), prod);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn iq4_nl_matmul_neon(a: &[f32], b: &IQ4NLMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    iq4_nl_matmul_neon_inner(a, b, c, m, k, n);
}

// ============================================================================
// Dispatch
// ============================================================================

/// Dispatch to the best available IQ4_NL GEMM kernel.
/// Uses row-dequantize hybrid: dequantize one B row to temp buffer, then SIMD FMA.
pub fn iq4_nl_matmul_transposed_b(a: &[f32], b: &IQ4NLMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(b.cols, k, "B cols must match k in transposed mode");
    assert_eq!(b.rows, n, "B rows must match n in transposed mode");
    let bpr = b.blocks_per_row();

    for v in c.iter_mut() { *v = 0.0; }

    // Thread-local reusable buffers to eliminate per-task allocations.
    use std::cell::RefCell;
    thread_local! {
        static TEMP_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
        static COL_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    }

    let compute_col = |j: usize, temp: &mut [f32], col: &mut [f32]| {
        let row_base = j * bpr;
        for block_idx in 0..bpr {
            let block = &b.blocks[row_base + block_idx];
            let base = block_idx * 32;
            block.dequantize(&mut temp[base..base + 32]);
        }
        for i in 0..m {
            col[i] = crate::kernels::simd::simd_dot_product(&a[i * k..(i + 1) * k], temp);
        }
    };

    // For large matrices, parallelize over output columns.
    if n >= 4096 {
        use rayon::prelude::*;
        let col_results: Vec<Vec<f32>> = (0..n)
            .into_par_iter()
            .map(|j| {
                TEMP_BUF.with(|buf| {
                    let mut temp = buf.borrow_mut();
                    temp.resize(k, 0.0);
                    COL_BUF.with(|cbuf| {
                        let mut col = cbuf.borrow_mut();
                        col.resize(m, 0.0);
                        compute_col(j, &mut temp, &mut col);
                        col.clone()
                    })
                })
            })
            .collect();
        for j in 0..n {
            for i in 0..m {
                c[i * n + j] = col_results[j][i];
            }
        }
    } else {
        TEMP_BUF.with(|buf| {
            let mut temp = buf.borrow_mut();
            temp.resize(k, 0.0);
            COL_BUF.with(|cbuf| {
                let mut col = cbuf.borrow_mut();
                col.resize(m, 0.0);
                for j in 0..n {
                    compute_col(j, &mut temp, &mut col);
                    for i in 0..m {
                        c[i * n + j] = col[i];
                    }
                }
            })
        });
    }
}

pub fn iq4_nl_matmul(a: &[f32], b: &IQ4NLMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(b.cols, n);
    assert_eq!(b.rows, k);
    let bpr = b.blocks_per_row();

    for v in c.iter_mut() { *v = 0.0; }

    let mut temp = vec![0.0f32; n];
    for l in 0..k {
        let row_base = l * bpr;
        for block_idx in 0..bpr {
            let block = &b.blocks[row_base + block_idx];
            let j_base = block_idx * 32;
            block.dequantize(&mut temp[j_base..j_base + 32]);
        }
        for i in 0..m {
            let a_val = a[i * k + l];
            let c_row = &mut c[i * n..(i + 1) * n];
            super::quant_gemm_common::row_fma(a_val, &temp, c_row);
        }
    }
}

/// Convert IQ4_NL matrix back to f32, then use the proven f32 SIMD matmul.
/// Reference fast path — dequantize once, then f32 GEMM.
pub fn iq4_nl_matmul_via_dequant(a: &[f32], b: &IQ4NLMatrix, m: usize, k: usize, n: usize) -> Vec<f32> {
    use super::simd;
    let b_f32 = b.dequantize();
    let mut c = vec![0.0f32; m * n];
    simd::simd_matmul(a, &b_f32, &mut c, m, k, n);
    c
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::iq4_nl::{Block, Matrix};

    fn make_test_matrix(rows: usize, cols: usize) -> Matrix {
        assert_eq!(cols % 32, 0);
        let bpr = cols / 32;
        let mut blocks = Vec::with_capacity(rows * bpr);
        for row in 0..rows {
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
        Matrix { rows, cols, blocks }
    }

    #[test]
    fn test_iq4_nl_matmul_vs_dequant() {
        let m = 2;
        let k = 4;
        let n = 32;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1).collect();
        let b_iq4 = make_test_matrix(k, n);

        let expected = iq4_nl_matmul_via_dequant(&a, &b_iq4, m, k, n);

        let mut c_scalar = vec![0.0f32; m * n];
        iq4_nl_matmul_scalar(&a, &b_iq4, &mut c_scalar, m, k, n);
        for i in 0..c_scalar.len() {
            assert!((c_scalar[i] - expected[i]).abs() < 1e-2,
                "scalar mismatch at {}: got {}, expected {}", i, c_scalar[i], expected[i]);
        }

        let mut c_dispatched = vec![0.0f32; m * n];
        iq4_nl_matmul(&a, &b_iq4, &mut c_dispatched, m, k, n);
        for i in 0..c_dispatched.len() {
            assert!((c_dispatched[i] - expected[i]).abs() < 1e-2,
                "dispatched mismatch at {}: got {}, expected {}", i, c_dispatched[i], expected[i]);
        }
    }

    #[test]
    fn test_iq4_nl_matmul_large() {
        let m = 4;
        let k = 8;
        let n = 64;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let b_iq4 = make_test_matrix(k, n);

        let expected = iq4_nl_matmul_via_dequant(&a, &b_iq4, m, k, n);

        let mut c = vec![0.0f32; m * n];
        iq4_nl_matmul(&a, &b_iq4, &mut c, m, k, n);

        for i in 0..c.len() {
            assert!((c[i] - expected[i]).abs() < 1e-2,
                "large test mismatch at {}: got {}, expected {}", i, c[i], expected[i]);
        }
    }
}
