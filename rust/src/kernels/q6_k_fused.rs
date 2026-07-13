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

const QK_K: usize = 256; // Q6_K super-block width

/// Fused dequant+matmul for Q6_K with transposed-B layout.
///
/// Computes `c[m × n] = a[m × k] @ b[k × n]^T` where `b` is stored
/// quantized as Q6_K blocks (in native GGUF layout: rows=n, cols=k).
pub fn q6_k_matmul_fused(a: &[f32], b: &Q6KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(b.cols, k, "B cols must match k");
    assert_eq!(b.rows, n, "B rows must match n");
    let bpr = b.blocks_per_row();
    debug_assert!(n % QK_K == 0, "Q6_K dequant assumes n % 256 == 0");

    // Zero output. (B leads to a += pattern; precondition c = 0.)
    for v in c.iter_mut() {
        *v = 0.0;
    }

    // Walk columns in slabs of 256 (= one super-block wide). Each
    // iteration dequants the slab fully and walks all m rows for it.
    let mut scratch: [f32; QK_K] = [0.0; QK_K];

    let mut j_slab_start = 0;
    while j_slab_start < n {
        let block_in_row = j_slab_start / QK_K;

        // For row l in 0..k: b.blocks[l*bpr + block_in_row] represents
        // one Q6_K block whose dequantised f32 values land at
        // c[.., j_slab_start..j_slab_start+256].
        for l in 0..k {
            b.blocks[l * bpr + block_in_row].dequantize(&mut scratch);
            // m-axis FMA inner loop. We re-use the just-dequantized
            // scratch `m` times instead of `m` times re-dequanting —
            // this is the whole point of the optimisation.
            #[allow(clippy::needless_range_loop)]
            for i in 0..m {
                let a_val = a[i * k + l];
                let c_row = &mut c[i * n + j_slab_start..i * n + j_slab_start + QK_K];
                for kk in 0..QK_K {
                    c_row[kk] += a_val * scratch[kk];
                }
            }
        }
        j_slab_start += QK_K;
    }
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
        // Small case: m=2, k=256, n=512. Should produce same output as
        // the existing scalar path within f32 epsilon.
        let m = 2usize;
        let k = 256usize;
        let n = 512usize;
        let bpr = n / QK_K;
        let mut blocks = Vec::with_capacity(n * bpr);
        for r in 0..n {
            for b_idx in 0..bpr {
                blocks.push(make_test_q6k(1, r as u32, b_idx as u32));
            }
        }
        let b = Matrix { rows: n, cols: k, blocks };
        // a: deterministic small floats
        let mut a = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                a[i * k + j] = (i + 1) as f32 * 0.01 + (j as f32).sin() * 0.5;
            }
        }
        // ground truth via dequant into f32 + simd_matmul
        let b_f32 = b.dequantize();
        let mut truth = vec![0.0f32; m * n];
        crate::kernels::simd::simd_matmul(&a, &b_f32, &mut truth, m, k, n);

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
