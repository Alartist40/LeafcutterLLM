//! Q6_K GEMM kernel
//!
//! Computes C = A × B where:
//!   - A is f32 [m, k]
//!   - B is Q6_K quantized [k, n]
//!   - C is f32 [m, n]
//!
//! Q6_K weights are dequantized on-the-fly inside the kernel using
//! a 256-element stack buffer (1024 bytes) per block.

use super::q6_k::Matrix as Q6KMatrix;

// ============================================================================
// Scalar reference implementation
// ============================================================================

pub fn q6_k_matmul_scalar(a: &[f32], b: &Q6KMatrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
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
                let j_base = block_idx * 256;

                let mut deq: [f32; 256] = [0.0; 256];
                block.dequantize(&mut deq);

                for jj in 0..256 {
                    c[i * n + j_base + jj] += a_val * deq[jj];
                }
            }
        }
    }
}

// ============================================================================
// x86_64 AVX2/FMA implementation (256-bit vectors, 8×f32)
// ============================================================================

/// Fused AVX2 Q6_K GEMV for m == 1.
///
/// For each output column j we accumulate dot(a[0..k], dequantized row j).
/// Instead of dequantizing the whole row into a temp buffer then doing a
/// separate dot product, the dequant and FMA are fused per block so the
/// quantized bytes are consumed directly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q6_k_gemv_col_avx2(a: &[f32], blocks: &[Block], k: usize) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let bpr = k / QK_K;
    let mut acc = _mm256_setzero_ps();

    for block_idx in 0..bpr {
        let block = &blocks[block_idx];
        let base = block_idx * QK_K;

        // Two 128-value superblocks per 256-block.
        for sb in 0..2 {
            let ql_base = sb * 64;
            let qh_base = sb * 32;
            let sc_base = sb * 8;
            let sb_base = base + sb * 128;

            // 32 l-values in chunks of 8 (one ymm lane each).
            for l0 in (0..32).step_by(8) {
                let is = l0 / 16; // scale set switches at l=16
                let a_ptr = a.as_ptr().add(sb_base + l0);

                // ql bytes for this chunk (low-nibble group and +32 group)
                let ql_lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    block.ql[ql_base + l0..ql_base + l0 + 8].as_ptr() as *const _));
                let ql_hi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    block.ql[ql_base + l0 + 32..ql_base + l0 + 40].as_ptr() as *const _));
                // qh bytes for this chunk
                let qh = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                    block.qh[qh_base + l0..qh_base + l0 + 8].as_ptr() as *const _));

                let three = _mm256_set1_epi32(3);
                let fifteen = _mm256_set1_epi32(0x0F);
                let thirty_two = _mm256_set1_epi32(32);

                let a_lo = _mm256_loadu_ps(a_ptr);
                let a_hi = _mm256_loadu_ps(a_ptr.add(32));
                let a_mid_lo = _mm256_loadu_ps(a_ptr.add(64));
                let a_mid_hi = _mm256_loadu_ps(a_ptr.add(96));

                // q1: (ql_lo & 0x0F) | ((qh >> 0 & 3) << 4)
                let q1 = _mm256_or_si256(
                    _mm256_and_si256(ql_lo, fifteen),
                    _mm256_slli_epi32(_mm256_and_si256(qh, three), 4));
                let q1f = _mm256_cvtepi32_ps(_mm256_sub_epi32(q1, thirty_two));
                // q2: (ql_hi & 0x0F) | ((qh >> 2 & 3) << 4)
                let q2 = _mm256_or_si256(
                    _mm256_and_si256(ql_hi, fifteen),
                    _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh, 2), three), 4));
                let q2f = _mm256_cvtepi32_ps(_mm256_sub_epi32(q2, thirty_two));
                // q3: (ql_lo >> 4) | ((qh >> 4 & 3) << 4)
                let q3 = _mm256_or_si256(
                    _mm256_srli_epi32(ql_lo, 4),
                    _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh, 4), three), 4));
                let q3f = _mm256_cvtepi32_ps(_mm256_sub_epi32(q3, thirty_two));
                // q4: (ql_hi >> 4) | ((qh >> 6 & 3) << 4)
                let q4 = _mm256_or_si256(
                    _mm256_srli_epi32(ql_hi, 4),
                    _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(qh, 6), three), 4));
                let q4f = _mm256_cvtepi32_ps(_mm256_sub_epi32(q4, thirty_two));

                // scales[sc_base + is + sub*2] for sub in 0..4
                let s0 = (block.scales[sc_base + is + 0] as i8) as f32;
                let s1 = (block.scales[sc_base + is + 2] as i8) as f32;
                let s2 = (block.scales[sc_base + is + 4] as i8) as f32;
                let s3 = (block.scales[sc_base + is + 6] as i8) as f32;

                let dd = block.d;
                let sc0 = _mm256_set1_ps(dd * s0);
                let sc1 = _mm256_set1_ps(dd * s1);
                let sc2 = _mm256_set1_ps(dd * s2);
                let sc3 = _mm256_set1_ps(dd * s3);

                // a-slices: a_ptr (sub0, offset 0..8), +32 (sub1, 32..40),
                // +64 (sub2, 64..72), +96 (sub3, 96..104)
                let p0 = _mm256_mul_ps(sc0, q1f);
                let p1 = _mm256_mul_ps(sc1, q2f);
                let p2 = _mm256_mul_ps(sc2, q3f);
                let p3 = _mm256_mul_ps(sc3, q4f);

                acc = _mm256_fmadd_ps(a_lo, p0, acc);
                acc = _mm256_fmadd_ps(a_hi, p1, acc);
                acc = _mm256_fmadd_ps(a_mid_lo, p2, acc);
                acc = _mm256_fmadd_ps(a_mid_hi, p3, acc);
            }
        }
    }

    acc
}

/// Scalar fused Q6_K GEMV fallback for m == 1 (non-AVX2 builds).
fn q6_k_gemv_col_scalar(a: &[f32], blocks: &[Block], k: usize) -> f32 {
    let bpr = k / QK_K;
    let mut acc = 0.0f32;
    for block_idx in 0..bpr {
        let block = &blocks[block_idx];
        let base = block_idx * QK_K;
        for sb in 0..2 {
            let ql_base = sb * 64;
            let qh_base = sb * 32;
            let sc_base = sb * 8;
            let sb_base = base + sb * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((block.ql[ql_base + l] & 0x0F) as i8
                    | (((block.qh[qh_base + l] >> 0) & 3) as i8) << 4) - 32;
                let q2 = ((block.ql[ql_base + l + 32] & 0x0F) as i8
                    | (((block.qh[qh_base + l] >> 2) & 3) as i8) << 4) - 32;
                let q3 = ((block.ql[ql_base + l] >> 4) as i8
                    | (((block.qh[qh_base + l] >> 4) & 3) as i8) << 4) - 32;
                let q4 = ((block.ql[ql_base + l + 32] >> 4) as i8
                    | (((block.qh[qh_base + l] >> 6) & 3) as i8) << 4) - 32;

                let s0 = (block.scales[sc_base + is + 0] as i8) as f32;
                let s1 = (block.scales[sc_base + is + 2] as i8) as f32;
                let s2 = (block.scales[sc_base + is + 4] as i8) as f32;
                let s3 = (block.scales[sc_base + is + 6] as i8) as f32;
                let d = block.d;

                let a0 = a[sb_base + l];
                let a1 = a[sb_base + 32 + l];
                let a2 = a[sb_base + 64 + l];
                let a3 = a[sb_base + 96 + l];
                acc += a0 * d * s0 * q1 as f32;
                acc += a1 * d * s1 * q2 as f32;
                acc += a2 * d * s2 * q3 as f32;
                acc += a3 * d * s3 * q4 as f32;
            }
        }
    }
    acc
}

use super::q6_k::Block;
use super::q6_k::QK_K;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q6_k_matmul_avx2_inner(a: &[f32], b: &Q6KMatrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    use std::arch::x86_64::*;

    assert_eq!(n % 256, 0, "AVX2 path requires n multiple of 256");
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
                let j_base = block_idx * 256;

                let mut deq: [f32; 256] = [0.0; 256];
                block.dequantize(&mut deq);

                let c_ptr = c_row.as_mut_ptr().add(j_base);
                let d_ptr = deq.as_ptr();

                for vec_idx in 0..32 {
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
pub unsafe fn q6_k_matmul_avx2(a: &[f32], b: &Q6KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    q6_k_matmul_avx2_inner(a, b, c, m, k, n);
}

// ============================================================================
// ARM NEON implementation (128-bit vectors, 4×f32)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn q6_k_matmul_neon_inner(a: &[f32], b: &Q6KMatrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    use std::arch::aarch64::*;

    assert_eq!(n % 256, 0, "NEON path requires n multiple of 256");
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
                let j_base = block_idx * 256;

                let mut deq: [f32; 256] = [0.0; 256];
                block.dequantize(&mut deq);

                let c_ptr = c_row.as_mut_ptr().add(j_base);
                let d_ptr = deq.as_ptr();

                for vec_idx in 0..64 {
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
pub unsafe fn q6_k_matmul_neon(a: &[f32], b: &Q6KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    q6_k_matmul_neon_inner(a, b, c, m, k, n);
}

// ============================================================================
// Dispatch
// ============================================================================

/// Dispatch to the best available Q6_K GEMM kernel.
/// Uses row-dequantize hybrid: dequantize one B row to temp buffer, then SIMD FMA.
pub fn q6_k_matmul_transposed_b(a: &[f32], b: &Q6KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(b.cols, k, "B cols must match k in transposed mode");
    assert_eq!(b.rows, n, "B rows must match n in transposed mode");
    let bpr = b.blocks_per_row();
    for v in c.iter_mut() { *v = 0.0; }

    // m == 1: fused dequant+dot GEMV (no temp buffer, no per-column Vec churn).
    if m == 1 {
        // Q8_K-activation integer-dot path is the default (faster); the f32
        // fused GEMV below remains as the opt-out (`LEAFCUTTER_Q8_GEMV=0`).
        // Deterministic mode also disables it so results are reproducible.
        let disabled = std::env::var("LEAFCUTTER_Q8_GEMV").map(|v| v == "0").unwrap_or(false);
        if !disabled && !crate::deterministic::enabled() {
            crate::kernels::q8_k_gemm::q6_k_matmul_transposed_b_q8(a, b, c, m, k, n);
            return;
        }
        let a_row = &a[..k];
        #[cfg(target_arch = "x86_64")]
        {
            if !crate::deterministic::enabled() && std::is_x86_feature_detected!("avx2") {
                use rayon::prelude::*;
                // Parallel over columns. Each column accumulates into one
                // output slot, so parallel writes never conflict.
                if n >= 4096 {
                    c.par_iter_mut().enumerate().for_each(|(j, out)| {
                        let row_base = j * bpr;
                        let acc = unsafe { q6_k_gemv_col_avx2(a_row, &b.blocks[row_base..row_base + bpr], k) };
                        let mut buf = [0.0f32; 8];
                        unsafe {
                            use std::arch::x86_64::*;
                            _mm256_storeu_ps(buf.as_mut_ptr(), acc);
                        }
                        *out = buf[0] + buf[1] + buf[2] + buf[3]
                            + buf[4] + buf[5] + buf[6] + buf[7];
                    });
                } else {
                    let mut acc_buf = [0.0f32; 8];
                    for j in 0..n {
                        let row_base = j * bpr;
                        let acc = unsafe { q6_k_gemv_col_avx2(a_row, &b.blocks[row_base..row_base + bpr], k) };
                        unsafe {
                            use std::arch::x86_64::*;
                            _mm256_storeu_ps(acc_buf.as_mut_ptr(), acc);
                        }
                        c[j] = acc_buf[0] + acc_buf[1] + acc_buf[2] + acc_buf[3]
                            + acc_buf[4] + acc_buf[5] + acc_buf[6] + acc_buf[7];
                    }
                }
                return;
            }
        }
        for j in 0..n {
            let row_base = j * bpr;
            c[j] = q6_k_gemv_col_scalar(a_row, &b.blocks[row_base..row_base + bpr], k);
        }
        return;
    }

    // Prefill (m > 1): quantize each activation row to Q8_K and use the
    // integer-dot path (sdot on aarch64), skipping the f32 dequant.
    let q8_disabled = std::env::var("LEAFCUTTER_Q8_GEMV").map(|v| v == "0").unwrap_or(false);
    if !q8_disabled && !crate::deterministic::enabled() {
        crate::kernels::q8_k_gemm::q6_k_matmul_transposed_b_q8(a, b, c, m, k, n);
        return;
    }

    use std::cell::RefCell;
    thread_local! {
        static TEMP_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
        static COL_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    }

    let compute_col = |j: usize, temp: &mut [f32], col: &mut [f32]| {
        let row_base = j * bpr;
        for block_idx in 0..bpr {
            let block = &b.blocks[row_base + block_idx];
            let base = block_idx * 256;
            block.dequantize(&mut temp[base..base + 256]);
        }
        for i in 0..m {
            col[i] = crate::kernels::simd::simd_dot_product(&a[i * k..(i + 1) * k], temp);
        }
    };

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

pub fn q6_k_matmul(a: &[f32], b: &Q6KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(b.cols, n);
    assert_eq!(b.rows, k);
    let bpr = b.blocks_per_row();

    for v in c.iter_mut() { *v = 0.0; }

    let mut temp = vec![0.0f32; n];
    for l in 0..k {
        let row_base = l * bpr;
        for block_idx in 0..bpr {
            let block = &b.blocks[row_base + block_idx];
            let j_base = block_idx * 256;
            block.dequantize(&mut temp[j_base..j_base + 256]);
        }
        for i in 0..m {
            let a_val = a[i * k + l];
            let c_row = &mut c[i * n..(i + 1) * n];
            super::quant_gemm_common::row_fma(a_val, &temp, c_row);
        }
    }
}

/// Convert Q6_K matrix back to f32, then use the proven f32 SIMD matmul.
/// Reference fast path — dequantize once, then f32 GEMM.
pub fn q6_k_matmul_via_dequant(a: &[f32], b: &Q6KMatrix, m: usize, k: usize, n: usize) -> Vec<f32> {
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
    use super::super::q6_k::{Block, QK_K};

    fn make_test_matrix(rows: usize, cols: usize) -> Q6KMatrix {
        assert_eq!(cols % QK_K, 0);
        let bpr = cols / QK_K;
        let mut blocks = Vec::with_capacity(rows * bpr);
        for row in 0..rows {
            for _b in 0..bpr {
                let d = 0.01f32 * (row + 1) as f32;
                let ql = [0u8; 128];
                let qh = [0u8; 64];
                let mut scales = [1u8; 16];
                // With ql=0, qh=0, q = -32 for all values
                // value = d * 1 * (-32) = -32 * d
                // Make some variation
                for s in 0..16 {
                    scales[s] = (s as u8).wrapping_add(1);
                }
                blocks.push(Block { d, ql, qh, scales });
            }
        }
        Q6KMatrix { rows, cols, blocks }
    }

    #[test]
    fn test_q6_k_matmul_vs_dequant() {
        let m = 2;
        let k = 4;
        let n = 256;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1).collect();
        let b_q6 = make_test_matrix(k, n);

        let expected = q6_k_matmul_via_dequant(&a, &b_q6, m, k, n);

        let mut c_scalar = vec![0.0f32; m * n];
        q6_k_matmul_scalar(&a, &b_q6, &mut c_scalar, m, k, n);
        for i in 0..c_scalar.len() {
            assert!((c_scalar[i] - expected[i]).abs() < 1e-2,
                "scalar mismatch at {}: got {}, expected {}", i, c_scalar[i], expected[i]);
        }

        let mut c_dispatched = vec![0.0f32; m * n];
        q6_k_matmul(&a, &b_q6, &mut c_dispatched, m, k, n);
        for i in 0..c_dispatched.len() {
            assert!((c_dispatched[i] - expected[i]).abs() < 1e-2,
                "dispatched mismatch at {}: got {}, expected {}", i, c_dispatched[i], expected[i]);
        }
    }

    #[test]
    fn test_q6_k_matmul_large() {
        let m = 4;
        let k = 8;
        let n = 512;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let b_q6 = make_test_matrix(k, n);

        let expected = q6_k_matmul_via_dequant(&a, &b_q6, m, k, n);

        let mut c = vec![0.0f32; m * n];
        q6_k_matmul(&a, &b_q6, &mut c, m, k, n);

        for i in 0..c.len() {
            assert!((c[i] - expected[i]).abs() < 1e-2,
                "large test mismatch at {}: got {}, expected {}", i, c[i], expected[i]);
        }
    }

    #[test]
    fn test_q6_k_gemv_fused_matches_transposed_b() {
        // The m == 1 dispatch defaults to the Q8_K-activation integer-dot
        // path. Verify the f32 fused GEMV matches the exact f32 scalar
        // reference, and that the Q8_K default dispatch stays within the
        // inherent Q8_K activation-quantization error.
        // Uses non-trivial ql/qh patterns and signed scales so the nibble
        // extraction logic is actually exercised.
        let m = 1;
        let k = 1024;
        let n = 2048;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32).cos() * 1.5).collect();

        let mut blocks = Vec::with_capacity(n * (k / QK_K));
        for row in 0..n {
            for _b in 0..(k / QK_K) {
                let d = 0.02f32 * (row as f32 + 1.0) / (n as f32);
                let mut ql = [0u8; 128];
                let mut qh = [0u8; 64];
                for i in 0..128 {
                    ql[i] = ((i * 7 + row) % 256) as u8;
                }
                for i in 0..64 {
                    qh[i] = ((i * 13 + row * 3) % 256) as u8;
                }
                let mut scales = [0u8; 16];
                for s in 0..16 {
                    scales[s] = (((s * 5 + row) as i32) % 61 - 30) as u8;
                }
                blocks.push(Block { d, ql, qh, scales });
            }
        }
        let b_q6 = Q6KMatrix { rows: n, cols: k, blocks };

        // (a) f32 fused AVX2 col kernel == exact f32 scalar reference.
        let mut c_scalar = vec![0.0f32; m * n];
        for j in 0..n {
            let row_base = j * (k / QK_K);
            c_scalar[j] = q6_k_gemv_col_scalar(&a, &b_q6.blocks[row_base..row_base + (k / QK_K)], k);
        }
        #[cfg(target_arch = "x86_64")]
        {
            let mut c_fused = vec![0.0f32; m * n];
            for j in 0..n {
                let row_base = j * (k / QK_K);
                let acc = unsafe { q6_k_gemv_col_avx2(&a, &b_q6.blocks[row_base..row_base + (k / QK_K)], k) };
                let mut buf = [0.0f32; 8];
                unsafe {
                    use std::arch::x86_64::*;
                    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
                }
                c_fused[j] = buf[0] + buf[1] + buf[2] + buf[3]
                    + buf[4] + buf[5] + buf[6] + buf[7];
            }
            for i in 0..n {
                assert!((c_scalar[i] - c_fused[i]).abs() < 1e-2,
                    "fused mismatch at col {}: scalar={}, fused={}", i, c_scalar[i], c_fused[i]);
            }
        }

        // (b) Q8_K default dispatch must match the Q8_K scalar reference.
        let mut a_blocks = vec![crate::kernels::q8_k::Q8KBlock::default(); k / QK_K];
        crate::kernels::q8_k::quantize_row_q8_k(&a, &mut a_blocks);
        let mut c_q8_scalar = vec![0.0f32; m * n];
        for j in 0..n {
            let row_base = j * (k / QK_K);
            c_q8_scalar[j] = crate::kernels::q8_k::q6_k_dot_q8_k_scalar(
                &b_q6.blocks[row_base..row_base + (k / QK_K)],
                &a_blocks,
            );
        }

        let mut c_dispatch = vec![0.0f32; m * n];
        q6_k_matmul_transposed_b(&a, &b_q6, &mut c_dispatch, m, k, n);
        for i in 0..n {
            assert!(c_dispatch[i].is_finite(), "sanity: dispatch must be finite at col {}", i);
            let tol = 1e-3 + 1e-3 * c_q8_scalar[i].abs();
            assert!((c_q8_scalar[i] - c_dispatch[i]).abs() < tol,
                "q8 dispatch mismatch at col {}: scalar={}, dispatch={}", i, c_q8_scalar[i], c_dispatch[i]);
        }
    }
}
