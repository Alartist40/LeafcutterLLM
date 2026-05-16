//! INT8 GEMM kernel for Q8_0 quantized weights
//!
//! Computes C = A × B where:
//!   - A is f32 [m, k]
//!   - B is Q8_0 quantized [k, n]
//!   - C is f32 [m, n]
//!
//! Q8_0 weights are dequantized on-the-fly inside the kernel using
//! small stack buffers (128 bytes per block). This gives 4× memory
//! bandwidth savings vs f32 weights while reusing proven f32 SIMD
//! kernels for the actual arithmetic.

use super::q4_0::Matrix as Q4Matrix;
use super::q8_0::Matrix as Q8Matrix;

// ============================================================================
// Scalar reference implementation
// ============================================================================

pub fn q8_0_matmul_scalar(a: &[f32], b: &Q8Matrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
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
                let scale = block.scale;
                for jj in 0..32 {
                    c[i * n + j_base + jj] += a_val * (block.qs[jj] as f32) * scale;
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
unsafe fn q8_0_matmul_avx2_inner(a: &[f32], b: &Q8Matrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    use std::arch::x86_64::*;

    assert_eq!(n % 32, 0, "AVX2 path requires n multiple of 32");
    let bpr = b.blocks_per_row();

    for i in 0..m {
        let c_row = &mut c[i * n..(i + 1) * n];
        // Zero output row with AVX2
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
                let scale = block.scale;

                // Dequantize 32 int8 values to a stack buffer, then AVX2 fma
                let mut deq: [f32; 32] = [0.0; 32];
                for jj in 0..32 {
                    deq[jj] = block.qs[jj] as f32 * scale;
                }

                let c_ptr = c_row.as_mut_ptr().add(j_base);
                let d_ptr = deq.as_ptr();

                // 4 × 8-wide AVX2 iterations
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
pub unsafe fn q8_0_matmul_avx2(a: &[f32], b: &Q8Matrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    q8_0_matmul_avx2_inner(a, b, c, m, k, n);
}

// ============================================================================
// ARM NEON implementation (128-bit vectors, 4×f32)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn q8_0_matmul_neon_inner(a: &[f32], b: &Q8Matrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    use std::arch::aarch64::*;

    assert_eq!(n % 32, 0, "NEON path requires n multiple of 32");
    let bpr = b.blocks_per_row();

    for i in 0..m {
        let c_row = &mut c[i * n..(i + 1) * n];
        // Zero output row with NEON
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
                let scale = block.scale;

                // Dequantize 32 int8 values to a stack buffer, then NEON fma
                let mut deq: [f32; 32] = [0.0; 32];
                for jj in 0..32 {
                    deq[jj] = block.qs[jj] as f32 * scale;
                }

                let c_ptr = c_row.as_mut_ptr().add(j_base);
                let d_ptr = deq.as_ptr();

                // 8 × 4-wide NEON iterations
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
pub unsafe fn q8_0_matmul_neon(a: &[f32], b: &Q8Matrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    q8_0_matmul_neon_inner(a, b, c, m, k, n);
}

// ============================================================================
// Dispatch
// ============================================================================

/// Dispatch to the best available INT8 GEMM kernel.
pub fn q8_0_matmul(a: &[f32], b: &Q8Matrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if n % 32 == 0 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                q8_0_matmul_avx2(a, b, c, m, k, n);
                return;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n % 32 == 0 && std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                q8_0_matmul_neon(a, b, c, m, k, n);
                return;
            }
        }
    }

    q8_0_matmul_scalar(a, b, c, m, k, n);
}

/// Convert Q8_0 matrix back to f32, then use the proven f32 SIMD matmul.
/// This is the "reference fast path" — dequantize once, then f32 GEMM.
/// Useful when the same B matrix is reused many times.
pub fn q8_0_matmul_via_dequant(a: &[f32], b: &Q8Matrix, m: usize, k: usize, n: usize) -> Vec<f32> {
    use super::simd;
    let b_f32 = b.dequantize();
    let mut c = vec![0.0f32; m * n];
    simd::simd_matmul(a, &b_f32, &mut c, m, k, n);
    c
}

// ============================================================================
// Q4_0 GEMM kernels
// ============================================================================

pub fn q4_0_matmul_scalar(a: &[f32], b: &Q4Matrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
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
                let scale = block.scale;
                for jj in 0..16 {
                    let byte = block.qs[jj];
                    let low = (byte & 0x0F) as f32;
                    let high = ((byte >> 4) & 0x0F) as f32;
                    c[i * n + j_base + jj * 2] += a_val * (low - 8.0) * scale;
                    c[i * n + j_base + jj * 2 + 1] += a_val * (high - 8.0) * scale;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_0_matmul_avx2_inner(a: &[f32], b: &Q4Matrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    use std::arch::x86_64::*;
    assert_eq!(n % 32, 0, "AVX2 path requires n multiple of 32");
    let bpr = b.blocks_per_row();

    for i in 0..m {
        let c_row = &mut c[i * n..(i + 1) * n];
        for j in (0..n).step_by(8) {
            _mm256_storeu_ps(c_row.as_mut_ptr().add(j), _mm256_setzero_ps());
        }

        for l in 0..b.rows {
            let a_val = _mm256_set1_ps(*a.get_unchecked(i * b.rows + l));
            let row_base = l * bpr;

            for block_idx in 0..bpr {
                let block = &b.blocks[row_base + block_idx];
                let j_base = block_idx * 32;
                let scale = block.scale;

                let mut deq: [f32; 32] = [0.0; 32];
                for jj in 0..16 {
                    let byte = block.qs[jj];
                    deq[jj * 2] = ((byte & 0x0F) as f32 - 8.0) * scale;
                    deq[jj * 2 + 1] = (((byte >> 4) & 0x0F) as f32 - 8.0) * scale;
                }

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
pub unsafe fn q4_0_matmul_avx2(a: &[f32], b: &Q4Matrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    q4_0_matmul_avx2_inner(a, b, c, m, k, n);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn q4_0_matmul_neon_inner(a: &[f32], b: &Q4Matrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    use std::arch::aarch64::*;
    assert_eq!(n % 32, 0, "NEON path requires n multiple of 32");
    let bpr = b.blocks_per_row();

    for i in 0..m {
        let c_row = &mut c[i * n..(i + 1) * n];
        for j in (0..n).step_by(4) {
            vst1q_f32(c_row.as_mut_ptr().add(j), vdupq_n_f32(0.0));
        }

        for l in 0..b.rows {
            let a_val = vdupq_n_f32(*a.get_unchecked(i * b.rows + l));
            let row_base = l * bpr;

            for block_idx in 0..bpr {
                let block = &b.blocks[row_base + block_idx];
                let j_base = block_idx * 32;
                let scale = block.scale;

                let mut deq: [f32; 32] = [0.0; 32];
                for jj in 0..16 {
                    let byte = block.qs[jj];
                    deq[jj * 2] = ((byte & 0x0F) as f32 - 8.0) * scale;
                    deq[jj * 2 + 1] = (((byte >> 4) & 0x0F) as f32 - 8.0) * scale;
                }

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
pub unsafe fn q4_0_matmul_neon(a: &[f32], b: &Q4Matrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    q4_0_matmul_neon_inner(a, b, c, m, k, n);
}

pub fn q4_0_matmul(a: &[f32], b: &Q4Matrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if n % 32 == 0 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                q4_0_matmul_avx2(a, b, c, m, k, n);
                return;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if n % 32 == 0 && std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                q4_0_matmul_neon(a, b, c, m, k, n);
                return;
            }
        }
    }

    q4_0_matmul_scalar(a, b, c, m, k, n);
}

pub fn q4_0_matmul_via_dequant(a: &[f32], b: &Q4Matrix, m: usize, k: usize, n: usize) -> Vec<f32> {
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

        // Test dispatched path (will use AVX2 on x86_64, NEON on aarch64)
        let mut c_dispatched = vec![0.0f32; m * n];
        q8_0_matmul(&a, &b_q8, &mut c_dispatched, m, k, n);
        for i in 0..c_dispatched.len() {
            assert!((c_dispatched[i] - expected[i]).abs() < 1e-3,
                "dispatched mismatch at {}: got {}, expected {}", i, c_dispatched[i], expected[i]);
        }
    }

    #[test]
    fn test_q8_0_matmul_large() {
        let m = 8;
        let k = 16;
        let n = 64;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let b_q8 = make_test_matrix(k, n);

        let expected = q8_0_matmul_via_dequant(&a, &b_q8, m, k, n);

        let mut c = vec![0.0f32; m * n];
        q8_0_matmul(&a, &b_q8, &mut c, m, k, n);

        for i in 0..c.len() {
            assert!((c[i] - expected[i]).abs() < 1e-3,
                "large test mismatch at {}: got {}, expected {}", i, c[i], expected[i]);
        }
    }

    // ============================================================================
    // Q4_0 tests
    // ============================================================================

    use super::super::q4_0::Block as Q4Block;

    fn make_q4_test_matrix(rows: usize, cols: usize) -> Q4Matrix {
        assert_eq!(cols % 32, 0);
        let bpr = cols / 32;
        let mut blocks = Vec::with_capacity(rows * bpr);
        for row in 0..rows {
            for b in 0..bpr {
                let scale = 0.01f32 * (row + 1) as f32;
                let mut qs = [0u8; 16];
                for jj in 0..16 {
                    let low = ((8 + (b * 32 + jj * 2) % 8) as u8).min(15);
                    let high = ((8 + (b * 32 + jj * 2 + 1) % 8) as u8).min(15);
                    qs[jj] = (high << 4) | low;
                }
                blocks.push(Q4Block { scale, qs });
            }
        }
        Q4Matrix { rows, cols, blocks }
    }

    #[test]
    fn test_q4_0_matmul_vs_dequant() {
        let m = 4;
        let k = 8;
        let n = 32;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1).collect();
        let b_q4 = make_q4_test_matrix(k, n);

        // Reference: dequantize then f32 matmul
        let expected = q4_0_matmul_via_dequant(&a, &b_q4, m, k, n);

        // Test scalar path
        let mut c_scalar = vec![0.0f32; m * n];
        q4_0_matmul_scalar(&a, &b_q4, &mut c_scalar, m, k, n);
        for i in 0..c_scalar.len() {
            assert!((c_scalar[i] - expected[i]).abs() < 1e-3,
                "Q4 scalar mismatch at {}: got {}, expected {}", i, c_scalar[i], expected[i]);
        }

        // Test dispatched path
        let mut c_dispatched = vec![0.0f32; m * n];
        q4_0_matmul(&a, &b_q4, &mut c_dispatched, m, k, n);
        for i in 0..c_dispatched.len() {
            assert!((c_dispatched[i] - expected[i]).abs() < 1e-3,
                "Q4 dispatched mismatch at {}: got {}, expected {}", i, c_dispatched[i], expected[i]);
        }
    }

    #[test]
    fn test_q4_0_matmul_large() {
        let m = 8;
        let k = 16;
        let n = 64;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let b_q4 = make_q4_test_matrix(k, n);

        let expected = q4_0_matmul_via_dequant(&a, &b_q4, m, k, n);

        let mut c = vec![0.0f32; m * n];
        q4_0_matmul(&a, &b_q4, &mut c, m, k, n);

        for i in 0..c.len() {
            assert!((c[i] - expected[i]).abs() < 1e-3,
                "Q4 large test mismatch at {}: got {}, expected {}", i, c[i], expected[i]);
        }
    }
}
