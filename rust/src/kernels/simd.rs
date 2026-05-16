//! SIMD kernels for f32 tensor operations
//!
//! Architecture-specific implementations:
//! - aarch64: ARM NEON (128-bit vectors, 4×f32)
//! - x86_64:  SSE2    (128-bit vectors, 4×f32)
//! - fallback: scalar loops

// ============================================================================
// ARM NEON implementation
// ============================================================================
#[cfg(target_arch = "aarch64")]
mod arch {
    use std::arch::aarch64::*;

    pub const VLEN: usize = 4; // 128-bit / 32-bit f32

    #[inline(always)]
    pub unsafe fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            let mut j = 0;
            while j + VLEN <= n {
                let mut acc = vdupq_n_f32(0.0);
                for l in 0..k {
                    let a_val = vdupq_n_f32(*a.get_unchecked(i * k + l));
                    let b_vec = vld1q_f32(b.as_ptr().add(l * n + j));
                    acc = vfmaq_f32(acc, a_val, b_vec);
                }
                vst1q_f32(c.as_mut_ptr().add(i * n + j), acc);
                j += VLEN;
            }
            // scalar tail
            for j_rem in j..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j_rem];
                }
                c[i * n + j_rem] = sum;
            }
        }
    }

    #[inline(always)]
    pub unsafe fn vec_add(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + VLEN <= len {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            let sum = vaddq_f32(av, bv);
            vst1q_f32(out.as_mut_ptr().add(i), sum);
            i += VLEN;
        }
        for rem in i..len {
            out[rem] = a[rem] + b[rem];
        }
    }

    #[inline(always)]
    pub unsafe fn vec_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + VLEN <= len {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            let prod = vmulq_f32(av, bv);
            vst1q_f32(out.as_mut_ptr().add(i), prod);
            i += VLEN;
        }
        for rem in i..len {
            out[rem] = a[rem] * b[rem];
        }
    }

    #[inline(always)]
    pub unsafe fn vec_scale(a: &[f32], scale: f32, out: &mut [f32]) {
        let len = a.len();
        let s = vdupq_n_f32(scale);
        let mut i = 0;
        while i + VLEN <= len {
            let av = vld1q_f32(a.as_ptr().add(i));
            let prod = vmulq_f32(av, s);
            vst1q_f32(out.as_mut_ptr().add(i), prod);
            i += VLEN;
        }
        for rem in i..len {
            out[rem] = a[rem] * scale;
        }
    }
}

// ============================================================================
// x86_64 SSE2 implementation
// ============================================================================
#[cfg(target_arch = "x86_64")]
mod arch {
    use std::arch::x86_64::*;

    pub const VLEN: usize = 4; // 128-bit / 32-bit f32

    #[inline(always)]
    pub unsafe fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            let mut j = 0;
            while j + VLEN <= n {
                let mut acc = _mm_setzero_ps();
                for l in 0..k {
                    let a_val = _mm_set1_ps(*a.get_unchecked(i * k + l));
                    let b_vec = _mm_loadu_ps(b.as_ptr().add(l * n + j));
                    acc = _mm_add_ps(_mm_mul_ps(a_val, b_vec), acc);
                }
                _mm_storeu_ps(c.as_mut_ptr().add(i * n + j), acc);
                j += VLEN;
            }
            // scalar tail
            for j_rem in j..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j_rem];
                }
                c[i * n + j_rem] = sum;
            }
        }
    }

    #[inline(always)]
    pub unsafe fn vec_add(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + VLEN <= len {
            let av = _mm_loadu_ps(a.as_ptr().add(i));
            let bv = _mm_loadu_ps(b.as_ptr().add(i));
            let sum = _mm_add_ps(av, bv);
            _mm_storeu_ps(out.as_mut_ptr().add(i), sum);
            i += VLEN;
        }
        for rem in i..len {
            out[rem] = a[rem] + b[rem];
        }
    }

    #[inline(always)]
    pub unsafe fn vec_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + VLEN <= len {
            let av = _mm_loadu_ps(a.as_ptr().add(i));
            let bv = _mm_loadu_ps(b.as_ptr().add(i));
            let prod = _mm_mul_ps(av, bv);
            _mm_storeu_ps(out.as_mut_ptr().add(i), prod);
            i += VLEN;
        }
        for rem in i..len {
            out[rem] = a[rem] * b[rem];
        }
    }

    #[inline(always)]
    pub unsafe fn vec_scale(a: &[f32], scale: f32, out: &mut [f32]) {
        let len = a.len();
        let s = _mm_set1_ps(scale);
        let mut i = 0;
        while i + VLEN <= len {
            let av = _mm_loadu_ps(a.as_ptr().add(i));
            let prod = _mm_mul_ps(av, s);
            _mm_storeu_ps(out.as_mut_ptr().add(i), prod);
            i += VLEN;
        }
        for rem in i..len {
            out[rem] = a[rem] * scale;
        }
    }
}

// ============================================================================
// Scalar fallback (non-SIMD architectures)
// ============================================================================
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
mod arch {
    pub const VLEN: usize = 1;

    #[inline(always)]
    pub unsafe fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    #[inline(always)]
    pub unsafe fn vec_add(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    #[inline(always)]
    pub unsafe fn vec_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    #[inline(always)]
    pub unsafe fn vec_scale(a: &[f32], scale: f32, out: &mut [f32]) {
        for i in 0..a.len() {
            out[i] = a[i] * scale;
        }
    }
}

// ============================================================================
// Safe wrappers
// ============================================================================

/// Matrix multiplication C = A × B
/// A: [m, k], B: [k, n], C: [m, n]
// ============================================================================
// x86_64 AVX2/FMA implementation (256-bit vectors, 8×f32)
// ============================================================================
#[cfg(target_arch = "x86_64")]
pub unsafe fn avx2_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    use std::arch::x86_64::*;
    for i in 0..m {
        let mut j = 0;
        while j + 8 <= n {
            let mut acc = _mm256_setzero_ps();
            for l in 0..k {
                let a_val = _mm256_set1_ps(*a.get_unchecked(i * k + l));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(l * n + j));
                acc = _mm256_fmadd_ps(a_val, b_vec, acc);
            }
            _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), acc);
            j += 8;
        }
        // scalar tail
        for j_rem in j..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j_rem];
            }
            c[i * n + j_rem] = sum;
        }
    }
}

pub fn simd_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                avx2_matmul(a, b, c, m, k, n);
                return;
            }
        }
        arch::matmul(a, b, c, m, k, n);
    }
}

/// Element-wise addition: out = a + b
pub fn simd_vec_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    unsafe {
        arch::vec_add(a, b, out);
    }
}

/// Element-wise multiply: out = a * b
pub fn simd_vec_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    unsafe {
        arch::vec_mul(a, b, out);
    }
}

/// Scale a vector: out = a * scale
pub fn simd_vec_scale(a: &[f32], scale: f32, out: &mut [f32]) {
    unsafe {
        arch::vec_scale(a, scale, out);
    }
}

/// Scale then element-wise multiply: out = a * scale * b
pub fn simd_vec_scale_mul(a: &[f32], scale: f32, b: &[f32], out: &mut [f32]) {
    let len = a.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let s = vdupq_n_f32(scale);
        while i + 4 <= len {
            let av = vld1q_f32(a.as_ptr().add(i));
            let bv = vld1q_f32(b.as_ptr().add(i));
            let scaled = vmulq_f32(av, s);
            let prod = vmulq_f32(scaled, bv);
            vst1q_f32(out.as_mut_ptr().add(i), prod);
            i += 4;
        }
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        let s = _mm_set1_ps(scale);
        while i + 4 <= len {
            let av = _mm_loadu_ps(a.as_ptr().add(i));
            let bv = _mm_loadu_ps(b.as_ptr().add(i));
            let scaled = _mm_mul_ps(av, s);
            let prod = _mm_mul_ps(scaled, bv);
            _mm_storeu_ps(out.as_mut_ptr().add(i), prod);
            i += 4;
        }
    }

    for rem in i..len {
        out[rem] = a[rem] * scale * b[rem];
    }
}

/// SIMD-accelerated sum of squares for a slice
pub fn simd_sum_sq(data: &[f32]) -> f32 {
    let len = data.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    let mut simd_sum = unsafe {
        use std::arch::aarch64::*;
        let mut acc = vdupq_n_f32(0.0);
        while i + 4 <= len {
            let v = vld1q_f32(data.as_ptr().add(i));
            acc = vfmaq_f32(acc, v, v);
            i += 4;
        }
        let arr: [f32; 4] = std::mem::transmute(acc);
        arr.iter().sum::<f32>()
    };

    #[cfg(target_arch = "x86_64")]
    let mut simd_sum = unsafe {
        use std::arch::x86_64::*;
        let mut acc = _mm_setzero_ps();
        while i + 4 <= len {
            let v = _mm_loadu_ps(data.as_ptr().add(i));
            acc = _mm_add_ps(_mm_mul_ps(v, v), acc);
            i += 4;
        }
        let arr: [f32; 4] = std::mem::transmute(acc);
        arr.iter().sum::<f32>()
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let mut simd_sum = 0.0f32;

    for rem in i..len {
        simd_sum += data[rem] * data[rem];
    }
    simd_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    #[test]
    fn test_simd_matmul_small() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0f32; 4];
        simd_matmul(&a, &b, &mut c, 2, 2, 2);
        let expected = reference_matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_simd_matmul_n_not_multiple_of_4() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3×3
        let m = 2;
        let k = 3;
        let n = 3;
        let mut c = vec![0.0f32; m * n];
        simd_matmul(&a, &b, &mut c, m, k, n);
        let expected = reference_matmul(&a, &b, m, k, n);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_simd_matmul_large() {
        let m = 16;
        let k = 32;
        let n = 24;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];
        simd_matmul(&a, &b, &mut c, m, k, n);
        let expected = reference_matmul(&a, &b, m, k, n);
        for i in 0..c.len() {
            assert!((c[i] - expected[i]).abs() < 1e-4,
                "Mismatch at {}: got {}, expected {}", i, c[i], expected[i]);
        }
    }

    #[test]
    fn test_simd_vec_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let mut out = vec![0.0f32; 5];
        simd_vec_add(&a, &b, &mut out);
        assert_eq!(out, vec![6.0, 6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_simd_sum_sq() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = simd_sum_sq(&data);
        let expected: f32 = data.iter().map(|x| x * x).sum();
        assert!((sum - expected).abs() < 1e-5);
    }
}
