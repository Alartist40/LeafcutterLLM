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

/// Recursively split the m dimension for parallel matmul.
/// Each half gets independent slices of A and C; B is read-only and shared.
fn simd_matmul_par(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    const MIN_ROWS_PER_TASK: usize = 4;
    if m <= MIN_ROWS_PER_TASK {
        unsafe { arch::matmul(a, b, c, m, k, n); }
        return;
    }
    let mid = m / 2;
    let (a_left, a_right) = a.split_at(mid * k);
    let (c_left, c_right) = c.split_at_mut(mid * n);
    rayon::join(
        || simd_matmul_par(a_left, b, c_left, mid, k, n),
        || simd_matmul_par(a_right, b, c_right, m - mid, k, n),
    );
}

/// Single-threaded SIMD matmul.
pub fn simd_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            if !crate::deterministic::enabled()
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
            {
                avx2_matmul(a, b, c, m, k, n);
                return;
            }
        }
        arch::matmul(a, b, c, m, k, n);
    }
}

/// Multi-threaded SIMD matmul using rayon.
/// Splits the m dimension across available CPU cores.
pub fn simd_matmul_parallel(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if !crate::deterministic::enabled()
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            simd_matmul_par_avx2(a, b, c, m, k, n);
            return;
        }
    }
    simd_matmul_par(a, b, c, m, k, n);
}

#[cfg(target_arch = "x86_64")]
fn simd_matmul_par_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    const MIN_ROWS_PER_TASK: usize = 4;
    if m <= MIN_ROWS_PER_TASK {
        unsafe { avx2_matmul(a, b, c, m, k, n); }
        return;
    }
    let mid = m / 2;
    let (a_left, a_right) = a.split_at(mid * k);
    let (c_left, c_right) = c.split_at_mut(mid * n);
    rayon::join(
        || simd_matmul_par_avx2(a_left, b, c_left, mid, k, n),
        || simd_matmul_par_avx2(a_right, b, c_right, m - mid, k, n),
    );
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

/// SIMD-accelerated SiLU (x * sigmoid(x)) using a fast base-2 exponential.
/// Out-of-range values fall back to the scalar path tail. Relative error < 1e-4.
pub fn simd_silu(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        // exp(-v) = 2^(t), t = -v * log2(e); polynomial for 2^frac on [0,1)
        const C1: f32 = 0.6931471805599453; // ln2
        const C2: f32 = 0.2402265069591007; // ln2^2/2
        const C3: f32 = 0.0555041086648216; // ln2^3/6
        const C4: f32 = 0.0096181291076284; // ln2^4/24
        const C5: f32 = 0.0013333558146428; // ln2^5/120
        const C6: f32 = 0.0001540353039338; // ln2^6/720
        let c1 = vdupq_n_f32(C1);
        let c2 = vdupq_n_f32(C2);
        let c3 = vdupq_n_f32(C3);
        let c4 = vdupq_n_f32(C4);
        let c5 = vdupq_n_f32(C5);
        let c6 = vdupq_n_f32(C6);
        let le = vdupq_n_f32(-1.4426950408889634); // -log2(e)
        let one = vdupq_n_f32(1.0);
        let neg126 = vdupq_n_f32(-126.0);
        let pos127 = vdupq_n_f32(127.0);
        while i + 4 <= len {
            let v = vld1q_f32(x.as_ptr().add(i));
            let t = vmulq_f32(v, le);
            let fl = vminq_f32(vmaxq_f32(vrndmq_f32(t), neg126), pos127); // clamp under/overflow
            let fr = vsubq_f32(t, fl);
            // 2^fr = 1 + fr*(C1 + fr*(C2 + fr*(C3 + fr*(C4 + fr*(C5 + fr*C6)))))
            let mut p = vfmaq_f32(c5, c6, fr);
            p = vfmaq_f32(c4, p, fr);
            p = vfmaq_f32(c3, p, fr);
            p = vfmaq_f32(c2, p, fr);
            p = vfmaq_f32(c1, p, fr);
            p = vfmaq_f32(one, p, fr);
            // 2^fl via exponent-field reconstruction
            let fl_int = vcvtq_s32_f32(fl);
            let biased_int = vaddq_s32(fl_int, vdupq_n_s32(127));
            let exp2 = vreinterpretq_f32_s32(vshlq_n_s32(biased_int, 23));
            let em = vmulq_f32(exp2, p); // exp(-v)
            let sig = vrecpeq_f32(vaddq_f32(one, em));
            let sig = vmulq_f32(sig, vrecpsq_f32(vaddq_f32(one, em), sig)); // Newton refine
            // silu = v * sigmoid(v); note sigmoid(v) = 1/(1+exp(-v))
            vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(v, sig));
            i += 4;
        }
    }

    for rem in i..len {
        out[rem] = x[rem] * (1.0 / (1.0 + (-x[rem]).exp()));
    }
}

/// SIMD-accelerated dot product (f32). Falls back to scalar on non-x86_64.
/// In deterministic mode (`LEAFCUTTER_DETERMINISTIC=1`) uses a serial,
/// f64-accumulated reference reduction so results are bit-identical
/// across machines and thread counts.
#[inline]
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();

    if crate::deterministic::enabled() {
        return crate::deterministic::dot_product(a, b);
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut i = 0;
        // Process 16 floats at a time (2 AVX2 registers)
        while i + 16 <= len {
            let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
            let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);
            i += 16;
        }
        // Tail: 8 floats
        if i + 8 <= len {
            let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
            i += 8;
        }
        // Reduce 8-lane sums to scalar
        let sum = _mm256_add_ps(sum0, sum1);
        let lo = _mm256_castps256_ps128(sum);
        let hi = _mm256_extractf128_ps(sum, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
        let mut acc = _mm_cvtss_f32(sum32);
        // Scalar tail
        for j in i..len {
            acc += a[j] * b[j];
        }
        acc
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
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
    fn test_simd_silu_matches_scalar() {
        let x: Vec<f32> = (0..200).map(|i| (i as f32 - 100.0) * 0.3).collect();
        let mut out = vec![0.0f32; x.len()];
        simd_silu(&x, &mut out);
        for (i, &v) in x.iter().enumerate() {
            let expected = v * (1.0 / (1.0 + (-v).exp()));
            let rel = if expected.abs() > 1e-6 {
                (out[i] - expected).abs() / expected.abs()
            } else {
                (out[i] - expected).abs()
            };
            assert!(rel < 1e-4, "silu mismatch at {}: got {}, expected {}", i, out[i], expected);
        }
    }

    #[test]
    fn test_simd_sum_sq() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = simd_sum_sq(&data);
        let expected: f32 = data.iter().map(|x| x * x).sum();
        assert!((sum - expected).abs() < 1e-5);
    }

    #[test]
    fn test_parallel_matmul_correctness() {
        let m = 128;
        let k = 256;
        let n = 128;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.001).collect();

        let mut c_single = vec![0.0f32; m * n];
        simd_matmul(&a, &b, &mut c_single, m, k, n);

        let mut c_parallel = vec![0.0f32; m * n];
        simd_matmul_parallel(&a, &b, &mut c_parallel, m, k, n);

        for i in 0..c_single.len() {
            assert!((c_single[i] - c_parallel[i]).abs() < 1e-4,
                "Parallel matmul mismatch at {}: single={}, par={}", i, c_single[i], c_parallel[i]);
        }
    }

    #[test]
    #[ignore = "Benchmark: measures parallel speedup"]
    fn bench_parallel_matmul_speedup() {
        let m = 512;
        let k = 512;
        let n = 512;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.001).collect();

        let iters = 10;

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let mut c = vec![0.0f32; m * n];
            simd_matmul(&a, &b, &mut c, m, k, n);
        }
        let single_ms = start.elapsed().as_millis() as f64 / iters as f64;

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let mut c = vec![0.0f32; m * n];
            simd_matmul_parallel(&a, &b, &mut c, m, k, n);
        }
        let par_ms = start.elapsed().as_millis() as f64 / iters as f64;

        println!("matmul {}x{}x{}: single={:.1}ms, parallel={:.1}ms, speedup={:.2}x",
            m, k, n, single_ms, par_ms, single_ms / par_ms);
    }
}
