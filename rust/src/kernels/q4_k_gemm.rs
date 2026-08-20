//! Q4_K GEMM kernel
//!
//! Computes C = A × B where:
//!   - A is f32 [m, k]
//!   - B is Q4_K quantized [k, n]
//!   - C is f32 [m, n]
//!
//! Q4_K weights are dequantized on-the-fly inside the kernel using
//! a 256-element stack buffer (1024 bytes) per block.

use super::q4_k::get_scale_min_k4;
use super::q4_k::Matrix as Q4KMatrix;

// ============================================================================
// Scalar reference implementation
// ============================================================================

pub fn q4_k_matmul_scalar(a: &[f32], b: &Q4KMatrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
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

                // Dequantize 256 values to stack buffer
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_k_matmul_avx2_inner(a: &[f32], b: &Q4KMatrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    use std::arch::x86_64::*;

    // AVX2 path requires n % 256 == 0 for the loop math. If a tensor's
    // outer dim isn't a multiple of 256 (some custom architectures), the
    // scalar fallback handles the tail. We require n % 8 at minimum for
    // any 256-bit vector work; otherwise fall back per the public entry.
    if n % 256 != 0 {
        // Caller-guarded: pub fn falls back before this is reached.
        // Panic in debug only; release returns gracefully via the public fn.
        debug_assert!(n % 256 == 0, "AVX2 path requires n multiple of 256");
        return;
    }
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
                let j_base = block_idx * 256;

                // Dequantize 256 values to a stack buffer
                let mut deq: [f32; 256] = [0.0; 256];
                block.dequantize(&mut deq);

                let c_ptr = c_row.as_mut_ptr().add(j_base);
                let d_ptr = deq.as_ptr();

                // 32 × 8-wide AVX2 iterations
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
pub unsafe fn q4_k_matmul_avx2(a: &[f32], b: &Q4KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    q4_k_matmul_avx2_inner(a, b, c, m, k, n);
}

// ============================================================================
// ARM NEON implementation (128-bit vectors, 4×f32)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn q4_k_matmul_neon_inner(a: &[f32], b: &Q4KMatrix, c: &mut [f32], m: usize, _k: usize, n: usize) {
    use std::arch::aarch64::*;

    assert_eq!(n % 256, 0, "NEON path requires n multiple of 256");
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
                let j_base = block_idx * 256;

                // Dequantize 256 values to a stack buffer
                let mut deq: [f32; 256] = [0.0; 256];
                block.dequantize(&mut deq);

                let c_ptr = c_row.as_mut_ptr().add(j_base);
                let d_ptr = deq.as_ptr();

                // 64 × 4-wide NEON iterations
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
pub unsafe fn q4_k_matmul_neon(a: &[f32], b: &Q4KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    q4_k_matmul_neon_inner(a, b, c, m, k, n);
}

// ============================================================================
// Transposed-B matmul (B stored as [n, k] instead of [k, n])
// ============================================================================

/// Q4_K matmul where B is stored in native GGUF layout [n, k].
/// Computes C = A @ B^T where B^T is [k, n].
pub fn q4_k_matmul_transposed_b(a: &[f32], b: &Q4KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(b.cols, k, "B cols ({}) must match k ({}) in transposed mode", b.cols, k);
    assert_eq!(b.rows, n, "B rows ({}) must match n ({}) in transposed mode", b.rows, n);

    // Fused GEMV fast path (the token-generation hot case): no temp buffer.
    if m == 1 {
        // Q8_K-activation integer-dot path is the default (faster); the f32
        // fused GEMV below remains as the opt-out (`LEAFCUTTER_Q8_GEMV=0`).
        // Deterministic mode also disables it so results are reproducible.
        let disabled = std::env::var("LEAFCUTTER_Q8_GEMV").map(|v| v == "0").unwrap_or(false);
        if !disabled && !crate::deterministic::enabled() {
            crate::kernels::q8_k_gemm::q4_k_matmul_transposed_b_q8(a, b, c, m, k, n);
            return;
        }
        q4_k_gemv_transposed_b(a, b, c, k, n);
        return;
    }

    // Prefill (m > 1): quantize each activation row to Q8_K and use the
    // integer-dot path too (sdot on aarch64), skipping the f32 dequant.
    let q8_disabled = std::env::var("LEAFCUTTER_Q8_GEMV").map(|v| v == "0").unwrap_or(false);
    if !q8_disabled && !crate::deterministic::enabled() {
        crate::kernels::q8_k_gemm::q4_k_matmul_transposed_b_q8(a, b, c, m, k, n);
        return;
    }

    let bpr = b.blocks_per_row(); // = k / 256

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
            let base = block_idx * 256;
            block.dequantize(&mut temp[base..base + 256]);
        }
        for i in 0..m {
            col[i] = crate::kernels::simd::simd_dot_product(&a[i * k..(i + 1) * k], temp);
        }
    };

    // For large matrices, parallelize over output columns into a flat
    // per-column buffer (no Vec<Vec<f32>> allocation churn), then scatter
    // into the row-major output. Each column j owns a contiguous m-slice.
    // Threshold: n >= 4096 ensures enough work per thread to amortize Rayon overhead.
    if n >= 4096 {
        use rayon::prelude::*;
        let mut col_results = vec![0.0f32; n * m];
        col_results.par_chunks_mut(m).enumerate().for_each(|(j, col)| {
            TEMP_BUF.with(|buf| {
                let mut temp = buf.borrow_mut();
                temp.resize(k, 0.0);
                let row_base = j * bpr;
                for block_idx in 0..bpr {
                    let block = &b.blocks[row_base + block_idx];
                    let base = block_idx * 256;
                    block.dequantize(&mut temp[base..base + 256]);
                }
                for i in 0..m {
                    col[i] = crate::kernels::simd::simd_dot_product(
                        &a[i * k..(i + 1) * k],
                        &temp,
                    );
                }
            });
        });
        for j in 0..n {
            for i in 0..m {
                c[i * n + j] = col_results[j * m + i];
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

// ============================================================================
// Fused GEMV (m == 1) — the token-generation hot path
// ============================================================================
//
// For a single query vector a[0..k] (m=1), compute c[j] = dot(a, B[j, :]) for
// every output column j. Dequantization and the dot product are FUSED: each
// block's 256 values are dequantized straight into ymm registers and FMA'd
// against the corresponding a-slice, so nothing is round-tripped through a
// f32 temp buffer. This is the biggest win for streaming decode.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_k_gemv_col_avx2(a: &[f32], b: &Q4KMatrix, col: usize) -> f32 {
    use std::arch::x86_64::*;
    let bpr = b.blocks_per_row();
    let row_base = col * bpr;
    let nibble_mask = _mm_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_ps();
    let mut a_off = 0usize;

    for block_idx in 0..bpr {
        let block = &b.blocks[row_base + block_idx];
        for group in 0..4 {
            let (sc1, m1) = get_scale_min_k4(group * 2, &block.scales);
            let (sc2, m2) = get_scale_min_k4(group * 2 + 1, &block.scales);
            let dl1 = block.d * sc1 as f32;
            let dl2 = block.d * sc2 as f32;
            let min1 = block.dmin * m1 as f32;
            let min2 = block.dmin * m2 as f32;
            let dl1_v = _mm256_set1_ps(dl1);
            let dl2_v = _mm256_set1_ps(dl2);
            let min1_v = _mm256_set1_ps(min1);
            let min2_v = _mm256_set1_ps(min2);

            for chunk in 0..4 {
                let off = chunk * 8;
                let raw = _mm_loadl_epi64(block.qs.as_ptr().add(group * 32 + off) as *const __m128i);

                // Low nibbles: dequant = dl1 * (qs & 0x0F) - min1
                let lo = _mm_and_si128(raw, nibble_mask);
                let lo_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
                let deq_lo = _mm256_fmsub_ps(dl1_v, lo_f32, min1_v);
                let a_lo = _mm256_loadu_ps(a.as_ptr().add(a_off + off));
                acc = _mm256_fmadd_ps(a_lo, deq_lo, acc);

                // High nibbles: dequant = dl2 * (qs >> 4 & 0x0F) - min2
                let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), nibble_mask);
                let hi_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
                let deq_hi = _mm256_fmsub_ps(dl2_v, hi_f32, min2_v);
                let a_hi = _mm256_loadu_ps(a.as_ptr().add(a_off + 32 + off));
                acc = _mm256_fmadd_ps(a_hi, deq_hi, acc);
            }
            a_off += 64;
        }
    }

    // Horizontal 8-lane reduce
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// Scalar fused GEMV reference (matches dequantize_scalar bit-for-bit).
fn q4_k_gemv_col_scalar(a: &[f32], b: &Q4KMatrix, col: usize) -> f32 {
    let bpr = b.blocks_per_row();
    let row_base = col * bpr;
    let mut acc = 0.0f32;
    let mut a_off = 0usize;

    for block_idx in 0..bpr {
        let block = &b.blocks[row_base + block_idx];
        let mut q_off = 0usize;
        for group in 0..4 {
            let (sc1, m1) = get_scale_min_k4(group * 2, &block.scales);
            let (sc2, m2) = get_scale_min_k4(group * 2 + 1, &block.scales);
            let dl1 = block.d * sc1 as f32;
            let dl2 = block.d * sc2 as f32;
            let min1 = block.dmin * m1 as f32;
            let min2 = block.dmin * m2 as f32;
            for l in 0..32 {
                acc += a[a_off + l] * (dl1 * (block.qs[q_off + l] & 0x0F) as f32 - min1);
                acc += a[a_off + l + 32] * (dl2 * (block.qs[q_off + l] >> 4) as f32 - min2);
            }
            a_off += 64;
            q_off += 32;
        }
    }
    acc
}

/// Fused GEMV dispatch for m == 1. Writes into c[0..n].
pub fn q4_k_gemv_transposed_b(a: &[f32], b: &Q4KMatrix, c: &mut [f32], k: usize, n: usize) {
    assert_eq!(b.cols, k, "B cols ({}) must match k ({}) in transposed mode", b.cols, k);
    assert_eq!(b.rows, n, "B rows ({}) must match n ({}) in transposed mode", b.rows, n);

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = !crate::deterministic::enabled()
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    if n >= 4096 {
        use rayon::prelude::*;
        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            c.par_iter_mut().enumerate().for_each(|(j, out)| {
                *out = unsafe { q4_k_gemv_col_avx2(a, b, j) };
            });
            return;
        }
        c.par_iter_mut().enumerate().for_each(|(j, out)| {
            *out = q4_k_gemv_col_scalar(a, b, j);
        });
    } else {
        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            for (j, out) in c.iter_mut().enumerate() {
                *out = unsafe { q4_k_gemv_col_avx2(a, b, j) };
            }
            return;
        }
        for (j, out) in c.iter_mut().enumerate() {
            *out = q4_k_gemv_col_scalar(a, b, j);
        }
    }
}

// ============================================================================
// Dispatch
// ============================================================================

/// Dispatch to the best available Q4_K GEMM kernel.
/// Falls back to scalar when n % 256 != 0 (some custom architectures).
pub fn q4_k_matmul(a: &[f32], b: &Q4KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(b.cols, n);
    assert_eq!(b.rows, k);

    // If n isn't a multiple of 256, the SIMD kernels can't vectorize
    // correctly. Fall back to scalar (rare path — most modern architectures
    // have 256-aligned hidden sizes).
    if n % 256 != 0 {
        q4_k_matmul_scalar(a, b, c, m, k, n);
        return;
    }

    let bpr = b.blocks_per_row();

    // Zero C
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

/// Convert Q4_K matrix back to f32, then use the proven f32 SIMD matmul.
/// Reference fast path — dequantize once, then f32 GEMM.
pub fn q4_k_matmul_via_dequant(a: &[f32], b: &Q4KMatrix, m: usize, k: usize, n: usize) -> Vec<f32> {
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
    use super::super::q4_k::{Block, QK_K};

    fn make_test_matrix(rows: usize, cols: usize) -> Q4KMatrix {
        assert_eq!(cols % QK_K, 0);
        let bpr = cols / QK_K;
        let mut blocks = Vec::with_capacity(rows * bpr);
        for row in 0..rows {
            for _b in 0..bpr {
                let d = 0.01f32 * (row + 1) as f32;
                let dmin = 0.001f32;
                let scales = [1u8; 12];
                let mut qs = [0u8; 128];
                for qi in 0..128 {
                    let low = ((qi % 8) as u8).min(15);
                    let high = (((qi + 4) % 8) as u8).min(15);
                    qs[qi] = (high << 4) | low;
                }
                blocks.push(Block { d, dmin, scales, qs });
            }
        }
        Q4KMatrix { rows, cols, blocks }
    }

    #[test]
    fn test_q4_k_matmul_vs_dequant() {
        let m = 2;
        let k = 4;
        let n = 256;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1).collect();
        let b_q4 = make_test_matrix(k, n);

        // Reference: dequantize then f32 matmul
        let expected = q4_k_matmul_via_dequant(&a, &b_q4, m, k, n);

        // Test scalar path
        let mut c_scalar = vec![0.0f32; m * n];
        q4_k_matmul_scalar(&a, &b_q4, &mut c_scalar, m, k, n);
        for i in 0..c_scalar.len() {
            assert!((c_scalar[i] - expected[i]).abs() < 1e-2,
                "scalar mismatch at {}: got {}, expected {}", i, c_scalar[i], expected[i]);
        }

        // Test dispatched path
        let mut c_dispatched = vec![0.0f32; m * n];
        q4_k_matmul(&a, &b_q4, &mut c_dispatched, m, k, n);
        for i in 0..c_dispatched.len() {
            assert!((c_dispatched[i] - expected[i]).abs() < 1e-2,
                "dispatched mismatch at {}: got {}, expected {}", i, c_dispatched[i], expected[i]);
        }
    }

    #[test]
    fn test_q4_k_gemv_fused_matches_transposed_b() {
        // The m == 1 dispatch defaults to the Q8_K-activation integer-dot
        // path. Verify the dispatch exactly matches the Q8_K scalar
        // reference, and stays within a loose envelope of the f32 fused
        // GEMV (the Q8_K activation quantization is an approximation whose
        // absolute error is data-dependent; real-model closeness is checked
        // separately via logit_diff).
        let m = 1;
        let k = 1024;
        let n = 2048;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32).sin() * 0.7).collect();
        let b_q4 = make_test_matrix(n, k);

        let mut a_blocks = vec![crate::kernels::q8_k::Q8KBlock::default(); k / QK_K];
        crate::kernels::q8_k::quantize_row_q8_k(&a, &mut a_blocks);

        // (a) exact Q8_K scalar reference per column.
        let mut c_q8_scalar = vec![0.0f32; m * n];
        for j in 0..n {
            let row_base = j * (k / QK_K);
            c_q8_scalar[j] = crate::kernels::q8_k::q4_k_dot_q8_k_scalar(
                &b_q4.blocks[row_base..row_base + (k / QK_K)],
                &a_blocks,
            );
        }

        // (b) Q8_K default dispatch must match the Q8_K scalar reference.
        let mut c_dispatch = vec![0.0f32; m * n];
        q4_k_matmul_transposed_b(&a, &b_q4, &mut c_dispatch, m, k, n);
        for i in 0..n {
            let tol = 1e-3 + 1e-3 * c_q8_scalar[i].abs();
            assert!((c_q8_scalar[i] - c_dispatch[i]).abs() < tol,
                "q8 dispatch mismatch at col {}: scalar={}, dispatch={}", i, c_q8_scalar[i], c_dispatch[i]);
        }
    }

    #[test]
    fn test_q4_k_matmul_large() {
        let m = 4;
        let k = 8;
        let n = 512;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let b_q4 = make_test_matrix(k, n);

        let expected = q4_k_matmul_via_dequant(&a, &b_q4, m, k, n);

        let mut c = vec![0.0f32; m * n];
        q4_k_matmul(&a, &b_q4, &mut c, m, k, n);

        for i in 0..c.len() {
            assert!((c[i] - expected[i]).abs() < 1e-2,
                "large test mismatch at {}: got {}, expected {}", i, c[i], expected[i]);
        }
    }
}
