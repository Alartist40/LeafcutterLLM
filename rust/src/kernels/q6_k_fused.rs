//! Optimized Q6_K matmul — fused dequant + FMA, designed for the
//! prefill hot path on the Ornith/Qwen 9B Q4_K_M model.
//!
//! Key improvement over `q6_k_matmul_transposed_b` in q6_k_gemm.rs:
//!   - The original code does N×M dot products where each dot recomputes
//!     the block-dequant independently for each i in 0..m.  This is
//!     O(m × k × bpr) work for the dequant step alone.
//!   - This kernel dequantises each block ONCE per column-slab, then
//!     performs m separate FMA inner loops over the cached dequant.
//!     That collapses m×bpr dequants → 1 dequant per block.
//!
//! Output: `c[i, j] = sum_l a[i, l] * B[j, l]` where B is stored as
//! Q6_K blocks in row-major order on (rows=k, cols=n).
//!
//! Implementation notes:
//!   - Slab width = 256 (one Q6_K super-block wide). Small enough to
//!     fit several in L1, big enough to amortize dequant overhead.
//!   - Inner M-FMA row-major loop (scalar + rustc auto-vectorization
//!     in release). Real AVX2/FMA inner loop follows in a separate
//!     patch once the algorithm is validated.

use super::q6_k::Matrix as Q6KMatrix;
use std::cell::RefCell;

const QK_K: usize = 256; // Q6_K super-block width

thread_local! {
    /// One shared scratch buffer reused across Q6_K matmuls. Sized for
    /// the largest input k we've seen (bpr * 256 f32 = bpr * 1KB).
    /// Grown via `require_capacity`, never shrunk.
    static SCRATCH: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

/// AVX2+FMA inner loop: c_row[i..i+8] += a_val * scratch[i..i+8]
/// for i in 0..QK_K, step 8.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn fma_row_avx2(a_val: f32, scratch: &[f32; QK_K], c_row: &mut [f32]) {
    use std::arch::x86_64::*;
    assert_eq!(c_row.len(), QK_K);
    let a_vec = _mm256_set1_ps(a_val);
    let mut k = 0usize;
    while k + 8 <= QK_K {
        let s_vec = _mm256_loadu_ps(scratch.as_ptr().add(k));
        let c_vec = _mm256_loadu_ps(c_row.as_mut_ptr().add(k));
        let r_vec = _mm256_fmadd_ps(a_vec, s_vec, c_vec);
        _mm256_storeu_ps(c_row.as_mut_ptr().add(k), r_vec);
        k += 8;
    }
    // Tail (must be empty since QK_K % 8 == 0).
    while k < QK_K {
        c_row[k] += a_val * scratch[k];
        k += 1;
    }
}

/// Fused dequant+matmul for Q6_K with transposed-B layout.
///
/// Computes `c[m × n] = a[m × k] @ b[k × n]^T` where `b` is stored
/// quantized as Q6_K blocks in row-major order on a [n × k] matrix
/// (`b.blocks[j * bpr + bi]` = block for row j, slab bi).
///
/// Algorithm: for each output column j (== B row index), dequantize
/// all bpr blocks of that row into the scratch buffer (one pass),
/// then run m FMA inner loops over the cached dequant.  This avoids
/// re-dequantising the same bpr blocks per i in 0..m.
pub fn q6_k_matmul_fused(a: &[f32], b: &Q6KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(b.cols, k, "B cols must match k");
    assert_eq!(b.rows, n, "B rows must match n");
    let bpr = b.blocks_per_row(); // = k / 256
    debug_assert_eq!(n % QK_K, 0, "Q6_K fused assumes n % 256 == 0");

    // Zero output.
    for v in c.iter_mut() {
        *v = 0.0;
    }

    // AVX2+FMA when available.
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    require_capacity(bpr);

    for j in 0..n {
        // Dequant all bpr blocks of row j into the shared scratch.
        SCRATCH.with(|buf| {
            let mut scratch = buf.borrow_mut();
            for bi in 0..bpr {
                b.blocks[j * bpr + bi]
                    .dequantize(&mut scratch[bi * QK_K..(bi + 1) * QK_K]);
            }
            // Now scratch = b[j, :] (full row, length k).
            // Do m FMA inner loops.
            #[allow(clippy::needless_range_loop)]
            for i in 0..m {
                let a_row = &a[i * k..(i + 1) * k];
                let mut acc = 0.0f32;
                #[allow(unused_assignments)]
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if use_avx2 {
                        use std::arch::x86_64::*;
                        let mut v_sum = _mm256_setzero_ps();
                        let mut kk = 0usize;
                        let a_ptr = a_row.as_ptr();
                        let s_ptr = scratch.as_ptr();
                        while kk + 8 <= k {
                            let a_vec = _mm256_loadu_ps(a_ptr.add(kk));
                            let s_vec = _mm256_loadu_ps(s_ptr.add(kk));
                            v_sum = _mm256_fmadd_ps(a_vec, s_vec, v_sum);
                            kk += 8;
                        }
                        let lo = _mm256_castps256_ps128(v_sum);
                        let hi = _mm256_extractf128_ps(v_sum, 1);
                        let sum4 = _mm_add_ps(lo, hi);
                        let shuf = _mm_movehdup_ps(sum4);
                        let sums2 = _mm_add_ps(sum4, shuf);
                        let shuf2 = _mm_movehl_ps(sums2, sums2);
                        let total = _mm_add_ss(sums2, shuf2);
                        acc = _mm_cvtss_f32(total);
                        while kk < k {
                            acc += a_row[kk] * scratch[kk];
                            kk += 1;
                        }
                        c[i * n + j] = acc;
                        continue;
                    }
                }
                // Scalar fallback.
                for kk in 0..k {
                    acc += a_row[kk] * scratch[kk];
                }
                c[i * n + j] = acc;
            }
        });
    }
}

/// Resize the thread-local SCRATCH buffer to fit `bpr` blocks (= `bpr * 256` floats).
fn require_capacity(bpr: usize) {
    let need = bpr * QK_K;
    SCRATCH.with(|buf| {
        let mut s = buf.borrow_mut();
        if s.len() < need {
            s.resize(need, 0.0);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::q6_k::{Block as Q6KBlock, Matrix};

    fn make_test_q6k(n_blocks: usize, seed_lo: u32, seed_hi: u32) -> Q6KBlock {
        // Construct a Q6_K block with deterministic non-trivial bytes.
        // d = 0.5, scales alternated 1/-1/2/-2, qh varied, ql varied.
        let mut data = [0u8; 210];
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        data[208] = d_bytes[0];
        data[209] = d_bytes[1];
        for i in 192..208 {
            data[i] = (i as i8 ^ 7) as u8;
        }
        // Bake in some pseudo-random bytes for ql & qh
        for byte in &mut data[0..192].iter_mut() {
            *byte = ((seed_lo ^ seed_hi) & 0xFF) as u8;
        }
        let _ = n_blocks;
        Q6KBlock::from_bytes(&data)
    }

    #[test]
    fn fused_matches_scalar_within_tolerance() {
        // The fused matmul takes b in transposed-B layout:
        //   b.rows = n, b.cols = k, B has shape [n, k] (one Q6_K block per row of length k/256).
        // Output: c[i, j] = Σ_l a[i, l] * b[j, l]
        //
        // Test: m=2, k=256, n=512.  b has rows=n=512, cols=k=256.
        // Each row of b has 1 block (cols/256 = 1).
        let m = 2usize;
        let k = 256usize;
        let n = 512usize;
        // b is [rows=n, cols=k].  blocks_per_row = cols / 256 = 1.
        let blocks_per_row = k / QK_K; // = 1
        assert_eq!(blocks_per_row, 1);
        let mut blocks = Vec::with_capacity(n * blocks_per_row);
        for r in 0..n {
            for b_idx in 0..blocks_per_row {
                blocks.push(make_test_q6k(1, r as u32, b_idx as u32));
            }
        }
        let b = Matrix {
            rows: n,
            cols: k,
            blocks,
        };
        // a: deterministic small floats
        let mut a = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                a[i * k + j] = (i + 1) as f32 * 0.01 + (j as f32).sin() * 0.5;
            }
        }
        // ground truth: exact f32 dequant + dot (independent of the Q8_K
        // activation approximation used by the default m>1 dispatch).
        let mut truth = vec![0.0f32; m * n];
        for r in 0..n {
            let mut deq = vec![0.0f32; k];
            for b_idx in 0..blocks_per_row {
                b.blocks[r * blocks_per_row + b_idx].dequantize(&mut deq[b_idx * 256..(b_idx + 1) * 256]);
            }
            for i in 0..m {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += a[i * k + l] * deq[l];
                }
                truth[i * n + r] = acc;
            }
        }

        let mut got = vec![0.0f32; m * n];
        q6_k_matmul_fused(&a, &b, &mut got, m, k, n);

        let mut max_err = 0.0f32;
        for (t, g) in truth.iter().zip(got.iter()) {
            max_err = max_err.max((t - g).abs());
        }
        let tol = (k as f32) * 1e-4;
        assert!(
            max_err < tol,
            "fused vs scalar mismatch: max_err={} > tol={}",
            max_err,
            tol
        );
    }
}
