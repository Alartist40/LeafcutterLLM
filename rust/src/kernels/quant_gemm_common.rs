//! Shared helpers for quantized GEMM kernels
//!
//! Provides a row-dequantize hybrid approach:
//!   1. Dequantize one row of B to a contiguous f32 buffer
//!   2. Vectorized multiply-add: C[i,:] += A[i,l] * B_row[:]
//!
//! This is faster than block-by-block dequantization inside the inner loop
//! because it amortizes dequantization cost and allows better cache prefetching.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_fma_avx2(a_val: f32, b_row: &[f32], c_row: &mut [f32]) {
    use std::arch::x86_64::*;
    let a_vec = _mm256_set1_ps(a_val);
    let n = b_row.len();
    let mut j = 0;
    while j + 8 <= n {
        let b_vec = _mm256_loadu_ps(b_row.as_ptr().add(j));
        let c_vec = _mm256_loadu_ps(c_row.as_mut_ptr().add(j));
        let prod = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
        _mm256_storeu_ps(c_row.as_mut_ptr().add(j), prod);
        j += 8;
    }
    for jt in j..n {
        c_row[jt] += a_val * b_row[jt];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn row_fma_neon(a_val: f32, b_row: &[f32], c_row: &mut [f32]) {
    use std::arch::aarch64::*;
    let a_vec = vdupq_n_f32(a_val);
    let n = b_row.len();
    let mut j = 0;
    while j + 4 <= n {
        let b_vec = vld1q_f32(b_row.as_ptr().add(j));
        let c_vec = vld1q_f32(c_row.as_mut_ptr().add(j));
        let prod = vfmaq_f32(c_vec, a_vec, b_vec);
        vst1q_f32(c_row.as_mut_ptr().add(j), prod);
        j += 4;
    }
    for jt in j..n {
        c_row[jt] += a_val * b_row[jt];
    }
}

/// Scalar fallback for row FMA.
fn row_fma_scalar(a_val: f32, b_row: &[f32], c_row: &mut [f32]) {
    for j in 0..b_row.len() {
        c_row[j] += a_val * b_row[j];
    }
}

/// Generic row FMA that dispatches to the best SIMD implementation.
#[inline]
pub fn row_fma(a_val: f32, b_row: &[f32], c_row: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if b_row.len() % 8 == 0 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { row_fma_avx2(a_val, b_row, c_row); }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if b_row.len() % 4 == 0 && std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { row_fma_neon(a_val, b_row, c_row); }
            return;
        }
    }
    row_fma_scalar(a_val, b_row, c_row);
}
