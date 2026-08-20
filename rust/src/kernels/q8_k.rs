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

/// ARM NEON Q4_K x Q8_K dot for one output column.
///
/// Bit-identical to `q4_k_dot_q8_k_scalar` (same integer math, NEON
/// vectorized): reconstructs the nibble-dequantized `aux8`, computes
/// `sumi = Σ_chunk scale[chunk] * dot(q8, aux8)` and
/// `minc = Σ_group bsums[g] * min[g/2]`, returns `d*sumi - dmin*minc`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn q4_k_dot_q8_k_neon(
    w_blocks: &[crate::kernels::q4_k::Block],
    a_blocks: &[Q8KBlock],
) -> f32 {
    use crate::kernels::q4_k::get_scale_min_k4;
    use crate::kernels::q4_k::QK_K as Q4QK;
    use std::arch::aarch64::*;
    debug_assert_eq!(w_blocks.len(), a_blocks.len());
    let mut sumf = 0.0f32;
    for i in 0..w_blocks.len() {
        let w = &w_blocks[i];
        let y = &a_blocks[i];

        // Reconstruct aux8: nibble-dequant 128 u8 -> 256 i8 (0..15).
        let mut aux8 = [0i8; Q4QK];
        unsafe {
            for g in 0..(Q4QK / 64) {
                let lo: uint8x16_t = vld1q_u8(w.qs[g * 32..].as_ptr());
                let hi: uint8x16_t = vld1q_u8(w.qs[g * 32 + 16..].as_ptr());
                let lo_mask = vdupq_n_u8(0x0F);
                let low_l = vandq_u8(lo, lo_mask);
                let high_l = vshrq_n_u8(lo, 4);
                let low_h = vandq_u8(hi, lo_mask);
                let high_h = vshrq_n_u8(hi, 4);
                vst1q_s8(aux8[g * 64..].as_mut_ptr(), vreinterpretq_s8_u8(low_l));
                vst1q_s8(aux8[g * 64 + 16..].as_mut_ptr(), vreinterpretq_s8_u8(low_h));
                vst1q_s8(aux8[g * 64 + 32..].as_mut_ptr(), vreinterpretq_s8_u8(high_l));
                vst1q_s8(aux8[g * 64 + 48..].as_mut_ptr(), vreinterpretq_s8_u8(high_h));
            }
        }

        // minc = sum over 16-groups of bsums[g] * min[g/2]
        let mut minc = 0i32;
        for g in 0..(Q4QK / 16) {
            let (_, m) = get_scale_min_k4(g / 2, &w.scales);
            minc += y.bsums[g] as i32 * m as i32;
        }

        // sumi = sum over 8 chunks of scale[chunk] * dot(q8, aux8)
        let mut sumi = 0i32;
        for chunk in 0..(Q4QK / 32) {
            let (sc, _) = get_scale_min_k4(chunk, &w.scales);
            let dot = unsafe {
                let a16_0 = vmovl_s8(vld1_s8(aux8[chunk * 32..].as_ptr()));
                let a16_1 = vmovl_s8(vld1_s8(aux8[chunk * 32 + 8..].as_ptr()));
                let a16_2 = vmovl_s8(vld1_s8(aux8[chunk * 32 + 16..].as_ptr()));
                let a16_3 = vmovl_s8(vld1_s8(aux8[chunk * 32 + 24..].as_ptr()));
                let y16_0 = vmovl_s8(vld1_s8(y.qs[chunk * 32..].as_ptr()));
                let y16_1 = vmovl_s8(vld1_s8(y.qs[chunk * 32 + 8..].as_ptr()));
                let y16_2 = vmovl_s8(vld1_s8(y.qs[chunk * 32 + 16..].as_ptr()));
                let y16_3 = vmovl_s8(vld1_s8(y.qs[chunk * 32 + 24..].as_ptr()));
                let p0 = vaddq_s32(
                    vpaddlq_s16(vmulq_s16(a16_0, y16_0)),
                    vpaddlq_s16(vmulq_s16(a16_1, y16_1)),
                );
                let p1 = vaddq_s32(
                    vpaddlq_s16(vmulq_s16(a16_2, y16_2)),
                    vpaddlq_s16(vmulq_s16(a16_3, y16_3)),
                );
                vaddvq_s32(vaddq_s32(p0, p1))
            };
            sumi += sc as i32 * dot;
        }

        let d = w.d * y.d;
        let dmin = w.dmin * y.d;
        sumf += d * sumi as f32 - dmin * minc as f32;
    }
    sumf
}

/// ARM NEON Q4_K x Q8_K dot for **two adjacent output columns**.
///
/// Same integer math as `q4_k_dot_q8_k_scalar` (so bit-identical results),
/// but interleaves the two columns' accumulate chains and shares the single
/// activation block load per `Q8_K` block.  This roughly doubles arithmetic
/// intensity for the compute-bound Q4_K decode GEMVs.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn q4_k_dot_q8_k_2col_neon(
    w1: &[crate::kernels::q4_k::Block],
    w2: &[crate::kernels::q4_k::Block],
    a_blocks: &[Q8KBlock],
) -> (f32, f32) {
    use crate::kernels::q4_k::get_scale_min_k4;
    use crate::kernels::q4_k::QK_K as Q4QK;
    use std::arch::aarch64::*;
    debug_assert_eq!(w1.len(), a_blocks.len());
    debug_assert_eq!(w2.len(), a_blocks.len());
    let mut sumf1 = 0.0f32;
    let mut sumf2 = 0.0f32;
    for i in 0..w1.len() {
        let w = &w1[i];
        let wb = &w2[i];
        let y = &a_blocks[i];

        // Reconstruct aux8 for both columns' nibbles (0..15).
        let mut aux1 = [0i8; Q4QK];
        let mut aux2 = [0i8; Q4QK];
        unsafe {
            for g in 0..(Q4QK / 64) {
                let lo: uint8x16_t = vld1q_u8(w.qs[g * 32..].as_ptr());
                let hi: uint8x16_t = vld1q_u8(w.qs[g * 32 + 16..].as_ptr());
                let lo2: uint8x16_t = vld1q_u8(wb.qs[g * 32..].as_ptr());
                let hi2: uint8x16_t = vld1q_u8(wb.qs[g * 32 + 16..].as_ptr());
                let lo_mask = vdupq_n_u8(0x0F);
                let low_l = vandq_u8(lo, lo_mask);
                let high_l = vshrq_n_u8(lo, 4);
                let low_h = vandq_u8(hi, lo_mask);
                let high_h = vshrq_n_u8(hi, 4);
                let low_l2 = vandq_u8(lo2, lo_mask);
                let high_l2 = vshrq_n_u8(lo2, 4);
                let low_h2 = vandq_u8(hi2, lo_mask);
                let high_h2 = vshrq_n_u8(hi2, 4);
                vst1q_s8(aux1[g * 64..].as_mut_ptr(), vreinterpretq_s8_u8(low_l));
                vst1q_s8(aux1[g * 64 + 16..].as_mut_ptr(), vreinterpretq_s8_u8(low_h));
                vst1q_s8(aux1[g * 64 + 32..].as_mut_ptr(), vreinterpretq_s8_u8(high_l));
                vst1q_s8(aux1[g * 64 + 48..].as_mut_ptr(), vreinterpretq_s8_u8(high_h));
                vst1q_s8(aux2[g * 64..].as_mut_ptr(), vreinterpretq_s8_u8(low_l2));
                vst1q_s8(aux2[g * 64 + 16..].as_mut_ptr(), vreinterpretq_s8_u8(low_h2));
                vst1q_s8(aux2[g * 64 + 32..].as_mut_ptr(), vreinterpretq_s8_u8(high_l2));
                vst1q_s8(aux2[g * 64 + 48..].as_mut_ptr(), vreinterpretq_s8_u8(high_h2));
            }
        }

        // minc = sum over 16-groups of bsums[g] * min[g/2] (both columns).
        let mut minc1 = 0i32;
        let mut minc2 = 0i32;
        for g in 0..(Q4QK / 16) {
            let (_, m) = get_scale_min_k4(g / 2, &w.scales);
            let (_, m2) = get_scale_min_k4(g / 2, &wb.scales);
            let b = y.bsums[g] as i32;
            minc1 += b * m as i32;
            minc2 += b * m2 as i32;
        }

        // sumi over 8 chunks, two interleaved dot chains sharing y loads.
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for chunk in 0..(Q4QK / 32) {
            let (sc, _) = get_scale_min_k4(chunk, &w.scales);
            let (sc2, _) = get_scale_min_k4(chunk, &wb.scales);
            let (dot1, dot2) = unsafe {
                let y16_0 = vmovl_s8(vld1_s8(y.qs[chunk * 32..].as_ptr()));
                let y16_1 = vmovl_s8(vld1_s8(y.qs[chunk * 32 + 8..].as_ptr()));
                let y16_2 = vmovl_s8(vld1_s8(y.qs[chunk * 32 + 16..].as_ptr()));
                let y16_3 = vmovl_s8(vld1_s8(y.qs[chunk * 32 + 24..].as_ptr()));
                let a1_0 = vmovl_s8(vld1_s8(aux1[chunk * 32..].as_ptr()));
                let a1_1 = vmovl_s8(vld1_s8(aux1[chunk * 32 + 8..].as_ptr()));
                let a1_2 = vmovl_s8(vld1_s8(aux1[chunk * 32 + 16..].as_ptr()));
                let a1_3 = vmovl_s8(vld1_s8(aux1[chunk * 32 + 24..].as_ptr()));
                let a2_0 = vmovl_s8(vld1_s8(aux2[chunk * 32..].as_ptr()));
                let a2_1 = vmovl_s8(vld1_s8(aux2[chunk * 32 + 8..].as_ptr()));
                let a2_2 = vmovl_s8(vld1_s8(aux2[chunk * 32 + 16..].as_ptr()));
                let a2_3 = vmovl_s8(vld1_s8(aux2[chunk * 32 + 24..].as_ptr()));
                let p1 = vaddq_s32(
                    vpaddlq_s16(vmulq_s16(a1_0, y16_0)),
                    vpaddlq_s16(vmulq_s16(a1_1, y16_1)),
                );
                let p2 = vaddq_s32(
                    vpaddlq_s16(vmulq_s16(a1_2, y16_2)),
                    vpaddlq_s16(vmulq_s16(a1_3, y16_3)),
                );
                let p3 = vaddq_s32(
                    vpaddlq_s16(vmulq_s16(a2_0, y16_0)),
                    vpaddlq_s16(vmulq_s16(a2_1, y16_1)),
                );
                let p4 = vaddq_s32(
                    vpaddlq_s16(vmulq_s16(a2_2, y16_2)),
                    vpaddlq_s16(vmulq_s16(a2_3, y16_3)),
                );
                (vaddvq_s32(vaddq_s32(p1, p2)), vaddvq_s32(vaddq_s32(p3, p4)))
            };
            sumi1 += sc as i32 * dot1;
            sumi2 += sc2 as i32 * dot2;
        }

        let d = w.d * y.d;
        let dmin = w.dmin * y.d;
        sumf1 += d * sumi1 as f32 - dmin * minc1 as f32;
        let d2 = wb.d * y.d;
        let dmin2 = wb.dmin * y.d;
        sumf2 += d2 * sumi2 as f32 - dmin2 * minc2 as f32;
    }
    (sumf1, sumf2)
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

/// ARM NEON Q6_K x Q8_K dot for one output column.
///
/// Bit-identical to `q6_k_dot_q8_k_scalar`: reconstructs the 6-bit `aux8`
/// (from `ql`/`qh`, minus 32), computes
/// `sumi = Σ_{half,sc} scale[half*8+sc] * dot(aux8[16], y.qs[16])`, returns
/// `d * sumi`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn q6_k_dot_q8_k_neon(
    w_blocks: &[crate::kernels::q6_k::Block],
    a_blocks: &[Q8KBlock],
) -> f32 {
    use crate::kernels::q6_k::Block;
    use crate::kernels::q6_k::QK_K as Q6QK;
    use std::arch::aarch64::*;
    debug_assert_eq!(w_blocks.len(), a_blocks.len());
    let mut sumf = 0.0f32;
    for i in 0..w_blocks.len() {
        let w = &w_blocks[i];
        let y = &a_blocks[i];

        // Reconstruct aux8: 6-bit dequant of ql/qh into i8, then -32.
        // Per 128-value half: ql 64 bytes, qh 32 bytes; for l in 0..32:
        //   q1(l) = (ql[l]&0xF)  | ((qh[l]>>0)&3)<<4   -> aux8[l]
        //   q2(l) = (ql[32+l]&0xF)| ((qh[l]>>2)&3)<<4   -> aux8[32+l]
        //   q3(l) = (ql[l]>>4)   | ((qh[l]>>4)&3)<<4    -> aux8[64+l]
        //   q4(l) = (ql[32+l]>>4)| ((qh[l]>>6)&3)<<4    -> aux8[96+l]
        let mut aux8 = [0i8; Q6QK];
        macro_rules! hib {
            ($v:expr, 0) => {
                vshlq_n_u8(vandq_u8($v, vdupq_n_u8(0x03)), 4)
            };
            ($v:expr, $s:literal) => {
                vshlq_n_u8(
                    vandq_u8(vshrq_n_u8($v, $s), vdupq_n_u8(0x03)),
                    4,
                )
            };
        }
        unsafe {
            for half in 0..2 {
                let ql = &w.ql[half * 64..half * 64 + 64];
                let qh = &w.qh[half * 32..half * 32 + 32];
                let ql0 = vld1q_u8(ql.as_ptr());
                let ql1 = vld1q_u8(ql.as_ptr().add(16));
                let ql2 = vld1q_u8(ql.as_ptr().add(32));
                let ql3 = vld1q_u8(ql.as_ptr().add(48));
                let qh0 = vld1q_u8(qh.as_ptr());
                let qh1 = vld1q_u8(qh.as_ptr().add(16));

                let mask0f = vdupq_n_u8(0x0F);
                let mask03 = vdupq_n_u8(0x03);
                let b0 = vdupq_n_u8(32);
                let base = half * 128;

                let store = |dst: &mut [i8], v: uint8x16_t| {
                    vst1q_s8(dst.as_mut_ptr(), vreinterpretq_s8_u8(vsubq_u8(v, b0)));
                };
                store(
                    &mut aux8[base..base + 16],
                    vorrq_u8(vandq_u8(ql0, mask0f), hib!(qh0, 0)),
                );
                store(
                    &mut aux8[base + 16..base + 32],
                    vorrq_u8(vandq_u8(ql1, mask0f), hib!(qh1, 0)),
                );
                store(
                    &mut aux8[base + 32..base + 48],
                    vorrq_u8(vandq_u8(ql2, mask0f), hib!(qh0, 2)),
                );
                store(
                    &mut aux8[base + 48..base + 64],
                    vorrq_u8(vandq_u8(ql3, mask0f), hib!(qh1, 2)),
                );
                store(
                    &mut aux8[base + 64..base + 80],
                    vorrq_u8(vshrq_n_u8(ql0, 4), hib!(qh0, 4)),
                );
                store(
                    &mut aux8[base + 80..base + 96],
                    vorrq_u8(vshrq_n_u8(ql1, 4), hib!(qh1, 4)),
                );
                store(
                    &mut aux8[base + 96..base + 112],
                    vorrq_u8(vshrq_n_u8(ql2, 4), hib!(qh0, 6)),
                );
                store(
                    &mut aux8[base + 112..base + 128],
                    vorrq_u8(vshrq_n_u8(ql3, 4), hib!(qh1, 6)),
                );
            }
        }

        // sumi = sum over 2 halves x 8 groups of scale * dot(aux8[16], y.qs[16])
        let mut sumi = 0i32;
        for half in 0..2 {
            let base = half * 128;
            for sc in 0..8 {
                let s = w.scales[half * 8 + sc] as i8 as i32;
                let dot = unsafe {
                    let a16_0 = vmovl_s8(vld1_s8(aux8[base + sc * 16..].as_ptr()));
                    let a16_1 = vmovl_s8(vld1_s8(aux8[base + sc * 16 + 8..].as_ptr()));
                    let y16_0 = vmovl_s8(vld1_s8(y.qs[base + sc * 16..].as_ptr()));
                    let y16_1 = vmovl_s8(vld1_s8(y.qs[base + sc * 16 + 8..].as_ptr()));
                    let p = vaddq_s32(
                        vpaddlq_s16(vmulq_s16(a16_0, y16_0)),
                        vpaddlq_s16(vmulq_s16(a16_1, y16_1)),
                    );
                    vaddvq_s32(p)
                };
                sumi += s * dot;
            }
        }

        let d = w.d * y.d;
        sumf += d * sumi as f32;
    }
    sumf
}

// ============================================================================
// dotprod (sdot) Q4_K / Q6_K x Q8_K dot products
// ============================================================================
//
// The CIX Sky1 (OPi 6 Plus) has asimddp (SDOT) and i8mm (US/MMLA).  Using the
// `sdot vd.4s, vn.16b, vm.16b` instruction replaces the 16-bit widening
// multiply chains (vmovl_s8 + vmulq_s16 + vpaddlq_s16) with one instruction
// that accumulates 16 int8xint8 products into int32 across 4 lanes.
//
// Q4_K nibbles are u8 (0..15); reinterpreting them as s8 is numerically
// identical because all values are non-negative.  Q6_K aux8 is s8 (-32..31).
// Both use plain `sdot` (s8 x s8) and stay bit-identical to the scalar
// integer reference.

/// sdot-accelerated Q4_K x Q8_K dot for one output column.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub fn q4_k_dot_q8_k_sdot(
    w_blocks: &[crate::kernels::q4_k::Block],
    a_blocks: &[Q8KBlock],
) -> f32 {
    use crate::kernels::q4_k::QK_K as Q4QK;
    debug_assert_eq!(w_blocks.len(), a_blocks.len());
    let mut sumf = 0.0f32;
    for i in 0..w_blocks.len() {
        let w = &w_blocks[i];
        let y = &a_blocks[i];

        // Reconstruct aux8 (u8 nibbles 0..15) with plain NEON.
        let mut aux8 = [0u8; Q4QK];
        build_q4_aux8(w, &mut aux8);

        sumf += unsafe { q4_block_dot_sdot(w, &aux8, y) };
    }
    sumf
}

/// Reconstruct Q4_K `aux8` (u8 nibbles 0..15) from a weight block.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn build_q4_aux8(w: &crate::kernels::q4_k::Block, aux8: &mut [u8]) {
    use crate::kernels::q4_k::QK_K as Q4QK;
    use std::arch::aarch64::*;
    unsafe {
        for g in 0..(Q4QK / 64) {
            let lo: uint8x16_t = vld1q_u8(w.qs[g * 32..].as_ptr());
            let hi: uint8x16_t = vld1q_u8(w.qs[g * 32 + 16..].as_ptr());
            let lo_mask = vdupq_n_u8(0x0F);
            vst1q_u8(aux8[g * 64..].as_mut_ptr(), vandq_u8(lo, lo_mask));
            vst1q_u8(aux8[g * 64 + 16..].as_mut_ptr(), vandq_u8(hi, lo_mask));
            vst1q_u8(aux8[g * 64 + 32..].as_mut_ptr(), vshrq_n_u8(lo, 4));
            vst1q_u8(aux8[g * 64 + 48..].as_mut_ptr(), vshrq_n_u8(hi, 4));
        }
    }
}

/// sdot dot of one Q4_K weight block against one Q8_K activation block, with
/// the weight's `aux8` nibbles already reconstructed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q4_block_dot_sdot(
    w: &crate::kernels::q4_k::Block,
    aux8: &[u8],
    y: &Q8KBlock,
) -> f32 {
    use crate::kernels::q4_k::get_scale_min_k4;
    use crate::kernels::q4_k::QK_K as Q4QK;
    use std::arch::aarch64::*;

    // minc = sum over 16-groups of bsums[g] * min[g/2]
    let mut minc = 0i32;
    for g in 0..(Q4QK / 16) {
        let (_, m) = get_scale_min_k4(g / 2, &w.scales);
        minc += y.bsums[g] as i32 * m as i32;
    }

    // sumi = sum over 8 chunks of scale[chunk] * dot(q8, aux8), sdot dot.
    let mut sumi = 0i32;
    for chunk in 0..(Q4QK / 32) {
        let (sc, _) = get_scale_min_k4(chunk, &w.scales);
        let base = chunk * 32;
        let dot = {
            let a1: uint8x16_t = vld1q_u8(aux8[base..].as_ptr());
            let a2: uint8x16_t = vld1q_u8(aux8[base + 16..].as_ptr());
            let b1: int8x16_t = vld1q_s8(y.qs[base..].as_ptr());
            let b2: int8x16_t = vld1q_s8(y.qs[base + 16..].as_ptr());
            let mut acc: int32x4_t = vdupq_n_s32(0);
            std::arch::asm!(
                "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
                inout(vreg) acc,
                in(vreg) vreinterpretq_s8_u8(a1),
                in(vreg) b1,
                options(pure, nomem, nostack)
            );
            let mut acc2: int32x4_t = vdupq_n_s32(0);
            std::arch::asm!(
                "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
                inout(vreg) acc2,
                in(vreg) vreinterpretq_s8_u8(a2),
                in(vreg) b2,
                options(pure, nomem, nostack)
            );
            vaddvq_s32(vaddq_s32(acc, acc2))
        };
        sumi += sc as i32 * dot;
    }

    let d = w.d * y.d;
    let dmin = w.dmin * y.d;
    d * sumi as f32 - dmin * minc as f32
}

/// sdot-accelerated Q6_K x Q8_K dot for one output column.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub fn q6_k_dot_q8_k_sdot(
    w_blocks: &[crate::kernels::q6_k::Block],
    a_blocks: &[Q8KBlock],
) -> f32 {
    use crate::kernels::q6_k::QK_K as Q6QK;
    debug_assert_eq!(w_blocks.len(), a_blocks.len());
    let mut sumf = 0.0f32;
    for i in 0..w_blocks.len() {
        let w = &w_blocks[i];
        let y = &a_blocks[i];

        // Reconstruct aux8: 6-bit dequant of ql/qh into i8, then -32.
        let mut aux8 = [0i8; Q6QK];
        build_q6_aux8(w, &mut aux8);

        sumf += unsafe { q6_block_dot_sdot(w, &aux8, y) };
    }
    sumf
}

/// Reconstruct Q6_K `aux8` (i8 -32..31) from a weight block.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub fn build_q6_aux8(w: &crate::kernels::q6_k::Block, aux8: &mut [i8]) {
    use crate::kernels::q6_k::QK_K as Q6QK;
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
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
    }
}

/// sdot dot of one Q6_K weight block against one Q8_K activation block, with
/// the weight's `aux8` already reconstructed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn q6_block_dot_sdot(
    w: &crate::kernels::q6_k::Block,
    aux8: &[i8],
    y: &Q8KBlock,
) -> f32 {
    use crate::kernels::q6_k::QK_K as Q6QK;
    use std::arch::aarch64::*;

    // sumi = sum over 2 halves x 8 groups of scale * dot(aux8[16], y.qs[16])
    let mut sumi = 0i32;
    for half in 0..2 {
        let base = half * 128;
        for sc in 0..8 {
            let s = w.scales[half * 8 + sc] as i8 as i32;
            let dot = {
                let a16: int8x16_t = vld1q_s8(aux8[base + sc * 16..].as_ptr());
                let y16: int8x16_t = vld1q_s8(y.qs[base + sc * 16..].as_ptr());
                let mut acc: int32x4_t = vdupq_n_s32(0);
                std::arch::asm!(
                    "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
                    inout(vreg) acc,
                    in(vreg) a16,
                    in(vreg) y16,
                    options(pure, nomem, nostack)
                );
                vaddvq_s32(acc)
            };
            sumi += s * dot;
        }
    }

    let d = w.d * y.d;
    d * sumi as f32
}

/// sdot-accelerated Q4_K x Q8_K dot for **two adjacent output columns**.
///
/// Same integer math as `q4_k_dot_q8_k_scalar`, but interleaves the two
/// columns' `sdot` chains so the memory pipe stays fed across asm blocks.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub fn q4_k_dot_q8_k_2col_sdot(
    w1: &[crate::kernels::q4_k::Block],
    w2: &[crate::kernels::q4_k::Block],
    a_blocks: &[Q8KBlock],
) -> (f32, f32) {
    use crate::kernels::q4_k::get_scale_min_k4;
    use crate::kernels::q4_k::QK_K as Q4QK;
    use std::arch::aarch64::*;
    debug_assert_eq!(w1.len(), a_blocks.len());
    debug_assert_eq!(w2.len(), a_blocks.len());
    let mut sumf1 = 0.0f32;
    let mut sumf2 = 0.0f32;
    for i in 0..w1.len() {
        let w = &w1[i];
        let wb = &w2[i];
        let y = &a_blocks[i];

        let mut aux1 = [0u8; Q4QK];
        let mut aux2 = [0u8; Q4QK];
        unsafe {
            for g in 0..(Q4QK / 64) {
                let lo: uint8x16_t = vld1q_u8(w.qs[g * 32..].as_ptr());
                let hi: uint8x16_t = vld1q_u8(w.qs[g * 32 + 16..].as_ptr());
                let lo2: uint8x16_t = vld1q_u8(wb.qs[g * 32..].as_ptr());
                let hi2: uint8x16_t = vld1q_u8(wb.qs[g * 32 + 16..].as_ptr());
                let lo_mask = vdupq_n_u8(0x0F);
                vst1q_u8(aux1[g * 64..].as_mut_ptr(), vandq_u8(lo, lo_mask));
                vst1q_u8(aux1[g * 64 + 16..].as_mut_ptr(), vandq_u8(hi, lo_mask));
                vst1q_u8(aux1[g * 64 + 32..].as_mut_ptr(), vshrq_n_u8(lo, 4));
                vst1q_u8(aux1[g * 64 + 48..].as_mut_ptr(), vshrq_n_u8(hi, 4));
                vst1q_u8(aux2[g * 64..].as_mut_ptr(), vandq_u8(lo2, lo_mask));
                vst1q_u8(aux2[g * 64 + 16..].as_mut_ptr(), vandq_u8(hi2, lo_mask));
                vst1q_u8(aux2[g * 64 + 32..].as_mut_ptr(), vshrq_n_u8(lo2, 4));
                vst1q_u8(aux2[g * 64 + 48..].as_mut_ptr(), vshrq_n_u8(hi2, 4));
            }
        }

        let mut minc1 = 0i32;
        let mut minc2 = 0i32;
        for g in 0..(Q4QK / 16) {
            let (_, m) = get_scale_min_k4(g / 2, &w.scales);
            let (_, m2) = get_scale_min_k4(g / 2, &wb.scales);
            let b = y.bsums[g] as i32;
            minc1 += b * m as i32;
            minc2 += b * m2 as i32;
        }

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for chunk in 0..(Q4QK / 32) {
            let (sc, _) = get_scale_min_k4(chunk, &w.scales);
            let (sc2, _) = get_scale_min_k4(chunk, &wb.scales);
            let (dot1, dot2) = unsafe {
                let base = chunk * 32;
                let a1a: uint8x16_t = vld1q_u8(aux1[base..].as_ptr());
                let a1b: uint8x16_t = vld1q_u8(aux1[base + 16..].as_ptr());
                let a2a: uint8x16_t = vld1q_u8(aux2[base..].as_ptr());
                let a2b: uint8x16_t = vld1q_u8(aux2[base + 16..].as_ptr());
                let b1: int8x16_t = vld1q_s8(y.qs[base..].as_ptr());
                let b2: int8x16_t = vld1q_s8(y.qs[base + 16..].as_ptr());
                let mut acc1: int32x4_t = vdupq_n_s32(0);
                let mut acc2: int32x4_t = vdupq_n_s32(0);
                let mut acc3: int32x4_t = vdupq_n_s32(0);
                let mut acc4: int32x4_t = vdupq_n_s32(0);
                std::arch::asm!(
                    "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
                    inout(vreg) acc1, in(vreg) vreinterpretq_s8_u8(a1a), in(vreg) b1,
                    options(pure, nomem, nostack)
                );
                std::arch::asm!(
                    "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
                    inout(vreg) acc2, in(vreg) vreinterpretq_s8_u8(a1b), in(vreg) b2,
                    options(pure, nomem, nostack)
                );
                std::arch::asm!(
                    "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
                    inout(vreg) acc3, in(vreg) vreinterpretq_s8_u8(a2a), in(vreg) b1,
                    options(pure, nomem, nostack)
                );
                std::arch::asm!(
                    "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
                    inout(vreg) acc4, in(vreg) vreinterpretq_s8_u8(a2b), in(vreg) b2,
                    options(pure, nomem, nostack)
                );
                let d1 = vaddvq_s32(vaddq_s32(acc1, acc2));
                let d2 = vaddvq_s32(vaddq_s32(acc3, acc4));
                (d1, d2)
            };
            sumi1 += sc as i32 * dot1;
            sumi2 += sc2 as i32 * dot2;
        }

        let d = w.d * y.d;
        let dmin = w.dmin * y.d;
        sumf1 += d * sumi1 as f32 - dmin * minc1 as f32;
        let d2 = wb.d * y.d;
        let dmin2 = wb.dmin * y.d;
        sumf2 += d2 * sumi2 as f32 - dmin2 * minc2 as f32;
    }
    (sumf1, sumf2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_q4_k_dot_q8_sdot_matches_scalar() {
        use crate::kernels::q4_k::Block as Q4Block;
        let k = 1024;
        let n = 64;
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.37).cos() * 1.5).collect();
        let mut a_blocks = vec![Q8KBlock::default(); k / QK_K];
        quantize_row_q8_k(&a, &mut a_blocks);

        for col in 0..n {
            let mut w_blocks = Vec::with_capacity(k / 256);
            for b in 0..(k / 256) {
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
            let expected = q4_k_dot_q8_k_scalar(&w_blocks, &a_blocks);
            let got = unsafe { q4_k_dot_q8_k_sdot(&w_blocks, &a_blocks) };
            let diff = (got - expected).abs();
            assert!(diff < 1e-4,
                "q4 sdot col {}: got={}, expected={}, diff={:.6}", col, got, expected, diff);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_q6_k_dot_q8_sdot_matches_scalar() {
        use crate::kernels::q6_k::Block as Q6Block;
        let k = 1024;
        let n = 64;
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.29).sin() * 2.5).collect();
        let mut a_blocks = vec![Q8KBlock::default(); k / QK_K];
        quantize_row_q8_k(&a, &mut a_blocks);

        for col in 0..n {
            let mut w_blocks = Vec::with_capacity(k / 256);
            for b in 0..(k / 256) {
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
            let expected = q6_k_dot_q8_k_scalar(&w_blocks, &a_blocks);
            let got = unsafe { q6_k_dot_q8_k_sdot(&w_blocks, &a_blocks) };
            let diff = (got - expected).abs();
            assert!(diff < 1e-4,
                "q6 sdot col {}: got={}, expected={}, diff={:.6}", col, got, expected, diff);
        }
    }

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
