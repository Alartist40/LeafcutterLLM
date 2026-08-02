//! Q8_K activation quantization and integer dot products.
//!
//! GEMV hot path (m == 1): instead of dequantizing every Q4_K/Q6_K weight
//! block to f32 and doing f32 FMA, we quantize the activation vector `a` to
//! Q8_K once per matmul, then compute each output dot product in the integer
//! domain (`_mm256_maddubs_epi16`, 16 MACs per instruction). This mirrors
//! llama.cpp's `ggml_vec_dot_q4_K_q8_K` / `ggml_vec_dot_q6_K_q8_K`.
//!
//! Block format matches llama.cpp's `block_q8_K`:
//!   - d     : f32 per-superblock scale
//!   - qs    : 256 i8 quantized values
//!   - bsums : 16 i16 sums of qs in groups of 16

pub const QK_K: usize = 256;

#[derive(Debug, Clone)]
pub struct Q8KBlock {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

impl Default for Q8KBlock {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        }
    }
}

/// Quantize a row of f32 activations into Q8_K blocks (llama.cpp
/// `quantize_row_q8_K_ref`). `x.len()` must be a multiple of 256.
pub fn quantize_row_q8_k(x: &[f32], blocks: &mut [Q8KBlock]) {
    assert_eq!(x.len() % QK_K, 0);
    assert_eq!(x.len() / QK_K, blocks.len());
    for (i, block) in blocks.iter_mut().enumerate() {
        let row = &x[i * QK_K..(i + 1) * QK_K];

        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in row {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax == 0.0 {
            block.d = 0.0;
            block.qs = [0; QK_K];
            block.bsums = [0; QK_K / 16];
            continue;
        }
        let iscale = -127.0f32 / max;
        for j in 0..QK_K {
            let v = (iscale * row[j]).round() as i32;
            block.qs[j] = v.min(127) as i8;
        }
        for j in 0..(QK_K / 16) {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += block.qs[j * 16 + ii] as i32;
            }
            block.bsums[j] = sum as i16;
        }
        block.d = 1.0 / iscale;
    }
}

/// Dequantize a Q8_K row back to f32 (reference / test only).
pub fn dequantize_row_q8_k(blocks: &[Q8KBlock], out: &mut [f32]) {
    for (i, block) in blocks.iter().enumerate() {
        for j in 0..QK_K {
            out[i * QK_K + j] = block.d * block.qs[j] as f32;
        }
    }
}

// ============================================================================
// Q4_K x Q8_K dot product
// ============================================================================
//
// Per output column j, dot(a, B[j,:]) where B is Q4_K and a is Q8_K.
// llama.cpp formula (ggml_vec_dot_q4_K_q8_K_generic):
//   for each block i:
//     sumi = sum over 32-chunks of scale[chunk] * dot(q8, nibbles)
//     minc = sum over 16-groups of bsums[g] * min[g/2]
//     result += d * sumi - dmin * minc
//   where d = w.d * y.d, dmin = w.dmin * y.d

/// Reference scalar Q4_K x Q8_K dot for one output column.
/// `w_blocks` = the k/256 weight blocks of that column; `a_blocks` = the
/// k/256 activation blocks of `a`. Mathematically equivalent to the f32
/// dequant + dot reference (within Q8_K activation rounding).
pub fn q4_k_dot_q8_k_scalar(
    w_blocks: &[crate::kernels::q4_k::Block],
    a_blocks: &[Q8KBlock],
) -> f32 {
    use crate::kernels::q4_k::get_scale_min_k4;
    debug_assert_eq!(w_blocks.len(), a_blocks.len());
    let mut sumf = 0.0f32;
    for i in 0..w_blocks.len() {
        let w = &w_blocks[i];
        let y = &a_blocks[i];

        // Reconstruct aux8: nibble-dequant w.qs (128 bytes -> 256 i8, 0..15).
        let mut aux8 = [0i8; QK_K];
        let mut q_off = 0usize;
        for g in 0..(QK_K / 64) {
            for l in 0..32 {
                aux8[g * 64 + l] = (w.qs[q_off + l] & 0x0F) as i8;
                aux8[g * 64 + 32 + l] = (w.qs[q_off + l] >> 4) as i8;
            }
            q_off += 32;
        }

        // minc = sum over 16-groups of bsums[g] * min[g/2]
        let mut minc = 0i32;
        for g in 0..(QK_K / 16) {
            let (_, m) = get_scale_min_k4(g / 2, &w.scales);
            minc += y.bsums[g] as i32 * m as i32;
        }

        // sumi = sum over 32-chunks of scale[chunk] * dot(q8, aux8)
        let mut sumi = 0i32;
        for chunk in 0..(QK_K / 32) {
            let (sc, _) = get_scale_min_k4(chunk, &w.scales);
            let mut dot = 0i32;
            let base = chunk * 32;
            for l in 0..32 {
                dot += aux8[base + l] as i32 * y.qs[base + l] as i32;
            }
            sumi += sc as i32 * dot;
        }

        let d = w.d * y.d;
        let dmin = w.dmin * y.d;
        sumf += d * sumi as f32 - dmin * minc as f32;
    }
    sumf
}

// ============================================================================
// Q6_K x Q8_K dot product
// ============================================================================
//
// Per block (Q6_K 210 bytes):
//   value = d * scales[sc] * (q - 32) where q is 6-bit, scales is i8
//   d = w.d * y.d
//   result += d * sum over groups of scales * dot(q-32, q8)
//
// llama.cpp computes the integer part with maddubs and subtracts a fixed
// "q8sclsub" = (sum of q8) * scales << 5 correction, since (q-32) is split
// as q and a constant 32 applied to the scale.

/// Reference scalar Q6_K x Q8_K dot for one output column.
pub fn q6_k_dot_q8_k_scalar(
    w_blocks: &[crate::kernels::q6_k::Block],
    a_blocks: &[Q8KBlock],
) -> f32 {
    use crate::kernels::q6_k::Block;
    use crate::kernels::q6_k::QK_K as Q6QK;
    debug_assert_eq!(w_blocks.len(), a_blocks.len());
    let mut sumf = 0.0f32;
    for i in 0..w_blocks.len() {
        let w = &w_blocks[i];
        let y = &a_blocks[i];

        // Reconstruct aux8: 6-bit dequant of ql/qh into i8, then -32.
        let mut aux8 = [0i8; Q6QK];
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        let mut idx = 0usize;
        for _ in 0..(Q6QK / 128) {
            for l in 0..32 {
                let q1 = ((w.ql[ql_off + l] & 0x0F) as i8
                    | (((w.qh[qh_off + l] >> 0) & 3) as i8) << 4) - 32;
                let q2 = ((w.ql[ql_off + l + 32] & 0x0F) as i8
                    | (((w.qh[qh_off + l] >> 2) & 3) as i8) << 4) - 32;
                let q3 = ((w.ql[ql_off + l] >> 4) as i8
                    | (((w.qh[qh_off + l] >> 4) & 3) as i8) << 4) - 32;
                let q4 = ((w.ql[ql_off + l + 32] >> 4) as i8
                    | (((w.qh[qh_off + l] >> 6) & 3) as i8) << 4) - 32;

                aux8[idx + l] = q1;
                aux8[idx + l + 32] = q2;
                aux8[idx + l + 64] = q3;
                aux8[idx + l + 96] = q4;
            }
            idx += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }

        // Sum over 128-value halves, each with 8 i8 scales covering 16 values.
        // scale applies to a group of 16 values (scales[sc] with sc in 0..16).
        let mut sumi = 0i32;
        for half in 0..2 {
            let base = half * 128;
            for sc in 0..8 {
                let s = w.scales[half * 8 + sc] as i8 as i32; // signed i8
                let mut dot = 0i32;
                let start = base + sc * 16;
                for l in 0..16 {
                    dot += aux8[start + l] as i32 * y.qs[start + l] as i32;
                }
                sumi += s * dot;
            }
        }

        let d = w.d * y.d;
        sumf += d * sumi as f32;
    }
    sumf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_k_quantize_roundtrip() {
        let mut x = vec![0.0f32; 512];
        for i in 0..512 {
            x[i] = ((i as f32) * 0.01).sin() * 3.0;
        }
        let mut blocks = vec![Q8KBlock::default(); 2];
        quantize_row_q8_k(&x, &mut blocks);
        let mut back = vec![0.0f32; 512];
        dequantize_row_q8_k(&blocks, &mut back);
        // Per-element relative error is meaningless near zero; instead check
        // that dot products against a probe vector are preserved.
        let probe: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.13).cos() * 1.0).collect();
        let exact: f32 = x.iter().zip(&probe).map(|(a, b)| a * b).sum();
        let approx: f32 = back.iter().zip(&probe).map(|(a, b)| a * b).sum();
        let rel = (exact - approx).abs() / exact.abs().max(1e-6);
        assert!(rel < 0.02, "q8_K dot rel error {} (exact={}, approx={})", rel, exact, approx);
        // Large-magnitude elements must be close in absolute terms.
        let scale = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        for i in 0..512 {
            let abs_err = (back[i] - x[i]).abs();
            assert!(abs_err < 0.05 * scale, "q8_K abs error {} at {}", abs_err, i);
        }
    }

    #[test]
    fn test_q8_k_zero_input() {
        let x = vec![0.0f32; 256];
        let mut blocks = vec![Q8KBlock::default(); 1];
        quantize_row_q8_k(&x, &mut blocks);
        assert_eq!(blocks[0].d, 0.0);
    }

    #[test]
    fn test_q4_k_dot_q8_k_matches_f32() {
        // Verify the integer-dot math equals the f32 dequant+dot reference.
        // Isolates integer-dot correctness from Q8_K approximation by using
        // the Q8_K-dequantized activations on BOTH sides.
        use crate::kernels::q4_k::{Block as Q4Block, QK_K as Q4QK};
        let k = 1024;
        let n = 8;
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.37).sin() * 2.0).collect();

        let mut a_blocks = vec![Q8KBlock::default(); k / QK_K];
        quantize_row_q8_k(&a, &mut a_blocks);
        let mut deq_a = vec![0.0f32; k];
        dequantize_row_q8_k(&a_blocks, &mut deq_a);

        for col in 0..n {
            let mut w_blocks = Vec::with_capacity(k / Q4QK);
            for b in 0..(k / Q4QK) {
                let d = 0.03f32 * (col as f32 + 1.0);
                let dmin = 0.01f32 * (col as f32 + 1.0);
                let mut scales = [0u8; 12];
                for s in 0..12 {
                    scales[s] = (((s * 3 + col) as i32) % 64) as u8;
                }
                let mut qs = [0u8; 128];
                for s in 0..128 {
                    qs[s] = ((s * 5 + col * 7 + b * 11) % 256) as u8;
                }
                w_blocks.push(Q4Block { d, dmin, scales, qs });
            }

            // f32 reference: dequant w to f32, dot with Q8_K-dequantized a
            let mut f32_out = 0.0f32;
            for (bi, wb) in w_blocks.iter().enumerate() {
                let mut deq = [0.0f32; Q4QK];
                wb.dequantize_scalar(&mut deq);
                for l in 0..Q4QK {
                    f32_out += deq_a[bi * Q4QK + l] * deq[l];
                }
            }

            let q8_out = q4_k_dot_q8_k_scalar(&w_blocks, &a_blocks);
            let rel = (q8_out - f32_out).abs() / f32_out.abs().max(1e-6);
            assert!(rel < 1e-3,
                "col {} q4 integer-dot mismatch: q8={}, f32={}, rel={:.6}", col, q8_out, f32_out, rel);
        }
    }

    #[test]
    fn test_q6_k_dot_q8_k_matches_f32() {
        use crate::kernels::q6_k::Block as Q6Block;
        use crate::kernels::q6_k::QK_K as Q6QK;
        let k = 1024;
        let n = 8;
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.29).cos() * 2.5).collect();

        let mut a_blocks = vec![Q8KBlock::default(); k / QK_K];
        quantize_row_q8_k(&a, &mut a_blocks);
        let mut deq_a = vec![0.0f32; k];
        dequantize_row_q8_k(&a_blocks, &mut deq_a);

        for col in 0..n {
            let mut w_blocks = Vec::with_capacity(k / Q6QK);
            for b in 0..(k / Q6QK) {
                let d = 0.02f32 * (col as f32 + 1.0);
                let mut ql = [0u8; 128];
                let mut qh = [0u8; 64];
                for s in 0..128 {
                    ql[s] = ((s * 3 + col + b * 13) % 256) as u8;
                }
                for s in 0..64 {
                    qh[s] = ((s * 7 + col * 5) % 256) as u8;
                }
                let mut scales = [0u8; 16];
                for s in 0..16 {
                    scales[s] = (((s * 5 + col) as i32) % 61 - 30) as u8;
                }
                w_blocks.push(Q6Block { d, ql, qh, scales });
            }

            let mut f32_out = 0.0f32;
            for (bi, wb) in w_blocks.iter().enumerate() {
                let mut deq = [0.0f32; Q6QK];
                wb.dequantize_scalar(&mut deq);
                for l in 0..Q6QK {
                    f32_out += deq_a[bi * Q6QK + l] * deq[l];
                }
            }

            let q8_out = q6_k_dot_q8_k_scalar(&w_blocks, &a_blocks);
            let rel = (q8_out - f32_out).abs() / f32_out.abs().max(1e-6);
            assert!(rel < 1e-3,
                "col {} q6 integer-dot mismatch: q8={}, f32={}, rel={:.6}", col, q8_out, f32_out, rel);
        }
    }
}
