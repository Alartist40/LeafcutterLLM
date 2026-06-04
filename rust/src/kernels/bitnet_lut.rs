//! BitNet Lookup Table (LUT) kernels
//!
//! BitNet b1.58 uses ternary weights {-1, 0, +1} stored in 2-bit signed format.
//! Instead of dequantizing to f32 then doing matmul, we use Lookup Tables
//! to compute partial sums directly from the packed 2-bit values.
//!
//! The key insight: a byte contains 4 packed 2-bit weights. There are only
//! 256 possible byte patterns. We precompute a LUT[256][4] where each entry
//! contains the 4 decoded f32 weights. During matmul, we index into this
//! table byte-by-byte, eliminating the bit-shifting overhead.
//!
//! SIMD variants:
//!   - Scalar: process 1 output column at a time
//!   - NEON:   4-wide f32 vectors, process 4 output columns in parallel
//!   - AVX2:   8-wide f32 vectors, process 8 output columns in parallel

pub const BITNET_BLOCK_SIZE: usize = 128;
pub const BITNET_BLOCK_BYTES: usize = 34;

/// Precomputed lookup table: lut[byte][w] = decoded f32 weight for the w-th
/// 2-bit value in `byte` (little-endian: w=0 is bits 0-1, w=1 is bits 2-3, ...).
static LUT: [[f32; 4]; 256] = {
    let mut table = [[0.0f32; 4]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut w = 0usize;
        while w < 4 {
            let packed = ((byte >> (w * 2)) & 0x03) as u8;
            table[byte][w] = match packed {
                0b00 => -1.0f32,
                0b01 =>  0.0f32,
                0b10 =>  1.0f32,
                _    =>  0.0f32,
            };
            w += 1;
        }
        byte += 1;
    }
    table
};

/// Dequantize I2_S blocks to f32 (reference scalar implementation).
pub fn dequantize_i2_s(data: &[u8], out: &mut [f32]) {
    let num_blocks = out.len() / BITNET_BLOCK_SIZE;
    for i in 0..num_blocks {
        let block = &data[i * BITNET_BLOCK_BYTES..(i + 1) * BITNET_BLOCK_BYTES];
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let weights = &block[2..34];

        for j in 0..BITNET_BLOCK_SIZE {
            let byte_idx = j / 4;
            let w = j % 4;
            let _packed = (weights[byte_idx] >> (w * 2)) & 0x03;
            out[i * BITNET_BLOCK_SIZE + j] = LUT[weights[byte_idx] as usize][w] * scale;
        }
    }
}

// ============================================================================
// BitNet LUT GEMM — scalar reference
// ============================================================================

/// BitNet matrix multiplication using the LUT approach.
///
/// Computes `C = A @ B` where:
///   - `A` is [M, K] in f32 (row-major)
///   - `B` is [K, N] in I2_S packed format (column-major blocks)
///   - `C` is [M, N] in f32 (row-major)
///
/// `b_packed` layout: for each column j, K weights are stored as consecutive
/// I2_S blocks (each block = 2-byte f16 scale + 32 bytes packed weights).
pub fn bitnet_matmul_lut(a: &[f32], b_packed: &[u8], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(c.len(), m * n);

    let k_blocks = (k + BITNET_BLOCK_SIZE - 1) / BITNET_BLOCK_SIZE;

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for blk in 0..k_blocks {
                let block_offset = (j * k_blocks + blk) * BITNET_BLOCK_BYTES;
                let block = &b_packed[block_offset..block_offset + BITNET_BLOCK_BYTES];
                let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let weights = &block[2..34];

                let k_start = blk * BITNET_BLOCK_SIZE;
                let k_end = (k_start + BITNET_BLOCK_SIZE).min(k);
                let mut byte_idx = 0;
                let mut k_pos = k_start;

                while k_pos + 4 <= k_end {
                    let byte = weights[byte_idx] as usize;
                    let a0 = a[i * k + k_pos];
                    let a1 = a[i * k + k_pos + 1];
                    let a2 = a[i * k + k_pos + 2];
                    let a3 = a[i * k + k_pos + 3];
                    acc += scale * (LUT[byte][0] * a0 + LUT[byte][1] * a1
                                  + LUT[byte][2] * a2 + LUT[byte][3] * a3);
                    byte_idx += 1;
                    k_pos += 4;
                }
                // Tail handling for non-multiple-of-4 K
                for rem in k_pos..k_end {
                    let w = rem % 4;
                    let byte = weights[byte_idx] as usize;
                    acc += scale * LUT[byte][w] * a[i * k + rem];
                }
            }
            c[i * n + j] = acc;
        }
    }
}

// ============================================================================
// BitNet LUT GEMM — NEON (aarch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub fn bitnet_matmul_lut_neon(a: &[f32], b_packed: &[u8], c: &mut [f32], m: usize, k: usize, n: usize) {
    use std::arch::aarch64::*;

    let k_blocks = (k + BITNET_BLOCK_SIZE - 1) / BITNET_BLOCK_SIZE;

    for i in 0..m {
        let mut j = 0;
        while j + 4 <= n {
            unsafe {
                // 4 output accumulators in parallel
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut acc2 = vdupq_n_f32(0.0);
                let mut acc3 = vdupq_n_f32(0.0);

                for blk in 0..k_blocks {
                    let _block_offset_base = blk * BITNET_BLOCK_BYTES;
                    let k_start = blk * BITNET_BLOCK_SIZE;
                    let k_end = (k_start + BITNET_BLOCK_SIZE).min(k);
                    let valid_in_block = k_end - k_start;
                    let bytes_to_process = valid_in_block / 4;

                    for col in 0..4 {
                        let block_offset = ((j + col) * k_blocks + blk) * BITNET_BLOCK_BYTES;
                        let block = &b_packed[block_offset..block_offset + BITNET_BLOCK_BYTES];
                        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                        let weights = &block[2..34];
                        let s = vdupq_n_f32(scale);

                        let acc_ptr = match col {
                            0 => &mut acc0,
                            1 => &mut acc1,
                            2 => &mut acc2,
                            _ => &mut acc3,
                        };

                        for byte_idx in 0..bytes_to_process {
                            let k_pos = k_start + byte_idx * 4;
                            let a_vec = vld1q_f32(a.as_ptr().add(i * k + k_pos));
                            let byte = weights[byte_idx] as usize;
                            // Load 4 decoded weights from LUT
                            let w = vld1q_f32(LUT[byte].as_ptr());
                            *acc_ptr = vfmaq_f32(*acc_ptr, vmulq_f32(w, s), a_vec);
                        }
                    }
                }

                // Sum the 4 lanes and store
                c[i * n + j]     = vaddvq_f32(acc0);
                c[i * n + j + 1] = vaddvq_f32(acc1);
                c[i * n + j + 2] = vaddvq_f32(acc2);
                c[i * n + j + 3] = vaddvq_f32(acc3);
            }
            j += 4;
        }

        // Scalar tail for remaining columns
        for j_rem in j..n {
            let mut acc = 0.0f32;
            for blk in 0..k_blocks {
                let block_offset = (j_rem * k_blocks + blk) * BITNET_BLOCK_BYTES;
                let block = &b_packed[block_offset..block_offset + BITNET_BLOCK_BYTES];
                let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let weights = &block[2..34];
                let k_start = blk * BITNET_BLOCK_SIZE;
                let k_end = (k_start + BITNET_BLOCK_SIZE).min(k);
                let mut byte_idx = 0;
                let mut k_pos = k_start;
                while k_pos + 4 <= k_end {
                    let byte = weights[byte_idx] as usize;
                    let a0 = a[i * k + k_pos];
                    let a1 = a[i * k + k_pos + 1];
                    let a2 = a[i * k + k_pos + 2];
                    let a3 = a[i * k + k_pos + 3];
                    acc += scale * (LUT[byte][0] * a0 + LUT[byte][1] * a1
                                  + LUT[byte][2] * a2 + LUT[byte][3] * a3);
                    byte_idx += 1;
                    k_pos += 4;
                }
                for rem in k_pos..k_end {
                    let w = rem % 4;
                    let byte = weights[byte_idx] as usize;
                    acc += scale * LUT[byte][w] * a[i * k + rem];
                }
            }
            c[i * n + j_rem] = acc;
        }
    }
}

// ============================================================================
// BitNet LUT GEMM — AVX2 (x86_64)
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub fn bitnet_matmul_lut_avx2(a: &[f32], b_packed: &[u8], c: &mut [f32], m: usize, k: usize, n: usize) {
    use std::arch::x86_64::*;

    let k_blocks = (k + BITNET_BLOCK_SIZE - 1) / BITNET_BLOCK_SIZE;

    for i in 0..m {
        let mut j = 0;
        while j + 8 <= n {
            // 8 output accumulators (2× AVX2 4-wide vectors, or 1× 8-wide)
            // For simplicity, process 4 at a time using 128-bit operations
            // A full 256-bit AVX2 version would use _mm256_fmadd_ps
            unsafe {
                let mut acc0 = _mm_setzero_ps();
                let mut acc1 = _mm_setzero_ps();
                let mut acc2 = _mm_setzero_ps();
                let mut acc3 = _mm_setzero_ps();
                let mut acc4 = _mm_setzero_ps();
                let mut acc5 = _mm_setzero_ps();
                let mut acc6 = _mm_setzero_ps();
                let mut acc7 = _mm_setzero_ps();

                for blk in 0..k_blocks {
                    let k_start = blk * BITNET_BLOCK_SIZE;
                    let k_end = (k_start + BITNET_BLOCK_SIZE).min(k);
                    let valid_in_block = k_end - k_start;
                    let bytes_to_process = valid_in_block / 4;

                    for byte_idx in 0..bytes_to_process {
                        let k_pos = k_start + byte_idx * 4;
                        let a_vec = _mm_loadu_ps(a.as_ptr().add(i * k + k_pos));

                        for col in 0..8 {
                            let block_offset = ((j + col) * k_blocks + blk) * BITNET_BLOCK_BYTES;
                            let block = &b_packed[block_offset..block_offset + BITNET_BLOCK_BYTES];
                            let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                            let weights = &block[2..34];
                            let byte = weights[byte_idx] as usize;
                            let w = _mm_loadu_ps(LUT[byte].as_ptr());
                            let ws = _mm_mul_ps(w, _mm_set1_ps(scale));
                            let fma = _mm_mul_ps(ws, a_vec);

                            match col {
                                0 => acc0 = _mm_add_ps(fma, acc0),
                                1 => acc1 = _mm_add_ps(fma, acc1),
                                2 => acc2 = _mm_add_ps(fma, acc2),
                                3 => acc3 = _mm_add_ps(fma, acc3),
                                4 => acc4 = _mm_add_ps(fma, acc4),
                                5 => acc5 = _mm_add_ps(fma, acc5),
                                6 => acc6 = _mm_add_ps(fma, acc6),
                                _ => acc7 = _mm_add_ps(fma, acc7),
                            }
                        }
                    }
                }

                // Horizontal sum each accumulator
                c[i * n + j]     = hsum_ps(acc0);
                c[i * n + j + 1] = hsum_ps(acc1);
                c[i * n + j + 2] = hsum_ps(acc2);
                c[i * n + j + 3] = hsum_ps(acc3);
                c[i * n + j + 4] = hsum_ps(acc4);
                c[i * n + j + 5] = hsum_ps(acc5);
                c[i * n + j + 6] = hsum_ps(acc6);
                c[i * n + j + 7] = hsum_ps(acc7);
            }

            j += 8;
        }

        // Scalar tail for remaining columns
        for j_rem in j..n {
            let mut acc = 0.0f32;
            for blk in 0..k_blocks {
                let block_offset = (j_rem * k_blocks + blk) * BITNET_BLOCK_BYTES;
                let block = &b_packed[block_offset..block_offset + BITNET_BLOCK_BYTES];
                let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                let weights = &block[2..34];
                let k_start = blk * BITNET_BLOCK_SIZE;
                let k_end = (k_start + BITNET_BLOCK_SIZE).min(k);
                let mut byte_idx = 0;
                let mut k_pos = k_start;
                while k_pos + 4 <= k_end {
                    let byte = weights[byte_idx] as usize;
                    let a0 = a[i * k + k_pos];
                    let a1 = a[i * k + k_pos + 1];
                    let a2 = a[i * k + k_pos + 2];
                    let a3 = a[i * k + k_pos + 3];
                    acc += scale * (LUT[byte][0] * a0 + LUT[byte][1] * a1
                                  + LUT[byte][2] * a2 + LUT[byte][3] * a3);
                    byte_idx += 1;
                    k_pos += 4;
                }
                for rem in k_pos..k_end {
                    let w = rem % 4;
                    let byte = weights[byte_idx] as usize;
                    acc += scale * LUT[byte][w] * a[i * k + rem];
                }
            }
            c[i * n + j_rem] = acc;
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn hsum_ps(v: std::arch::x86_64::__m128) -> f32 {
    let shuf = std::arch::x86_64::_mm_movehl_ps(v, v);
    let sums = std::arch::x86_64::_mm_add_ps(v, shuf);
    let shuf2 = std::arch::x86_64::_mm_shuffle_ps(sums, sums, 0x01);
    std::arch::x86_64::_mm_cvtss_f32(std::arch::x86_64::_mm_add_ss(sums, shuf2))
}

// ============================================================================
// Unified dispatch
// ============================================================================

pub fn bitnet_matmul_lut_dispatch(a: &[f32], b_packed: &[u8], c: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(target_arch = "aarch64")]
    { return bitnet_matmul_lut_neon(a, b_packed, c, m, k, n); }
    #[cfg(target_arch = "x86_64")]
    { return bitnet_matmul_lut_avx2(a, b_packed, c, m, k, n); }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    { return bitnet_matmul_lut(a, b_packed, c, m, k, n); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i2_s_block_layout() {
        assert_eq!(BITNET_BLOCK_SIZE, 128);
        assert_eq!(BITNET_BLOCK_BYTES, 34);
    }

    #[test]
    fn test_lut_values() {
        // LUT is indexed by byte, decoded little-endian (w=0 = bits 0-1)
        assert_eq!(LUT[0b00_00_00_00], [-1.0, -1.0, -1.0, -1.0]); // all 00
        assert_eq!(LUT[0b01_01_01_01], [ 0.0,  0.0,  0.0,  0.0]); // all 01
        assert_eq!(LUT[0b10_10_10_10], [ 1.0,  1.0,  1.0,  1.0]); // all 10
        // 0x69 = 0b0110_1001: bits 0-1=01→0, bits 2-3=10→1, bits 4-5=10→1, bits 6-7=01→0
        assert_eq!(LUT[0x69], [0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_i2_s_dequant_zero() {
        let mut data = vec![0u8; 34];
        data[0] = 0x00; data[1] = 0x00;
        for i in 2..34 { data[i] = 0x55; }
        let mut out = vec![0.0f32; 128];
        dequantize_i2_s(&data, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_i2_s_dequant_all_ones() {
        let mut data = vec![0u8; 34];
        data[0] = 0x00; data[1] = 0x3C;
        for i in 2..34 { data[i] = 0xAA; }
        let mut out = vec![0.0f32; 128];
        dequantize_i2_s(&data, &mut out);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 0.001));
    }

    fn make_packed_column(k: usize, scale: f32) -> Vec<u8> {
        let k_blocks = (k + BITNET_BLOCK_SIZE - 1) / BITNET_BLOCK_SIZE;
        let mut packed = vec![0u8; k_blocks * BITNET_BLOCK_BYTES];
        let scale_f16 = half::f16::from_f32(scale).to_le_bytes();
        for blk in 0..k_blocks {
            let off = blk * BITNET_BLOCK_BYTES;
            packed[off] = scale_f16[0];
            packed[off + 1] = scale_f16[1];
            // Fill with 0xAA = all +1 weights
            for b in 2..34 { packed[off + b] = 0xAA; }
        }
        packed
    }

    #[test]
    fn test_bitnet_matmul_lut_all_ones() {
        // A = [1.0, 1.0, 1.0, 1.0], B = all +1 with scale=2.0
        // C = [4 * 2.0] = [8.0]
        let a = vec![1.0f32; 128];
        let b = make_packed_column(128, 2.0);
        let mut c = vec![0.0f32; 1];
        bitnet_matmul_lut(&a, &b, &mut c, 1, 128, 1);
        assert!((c[0] - 256.0).abs() < 0.1, "Expected ~256.0, got {}", c[0]);
        // 128 weights * 1.0 * 2.0 = 256.0
    }

    #[test]
    fn test_bitnet_matmul_lut_mixed_weights() {
        // A = all ones, B = pattern 0x69 = [0, -1, 1, 0] per byte
        // Sum per byte = 0 + (-1) + 1 + 0 = 0
        // 32 bytes * 0 = 0 total
        let k = 128;
        let a = vec![1.0f32; k];
        let k_blocks = 1;
        let mut b = vec![0u8; k_blocks * BITNET_BLOCK_BYTES];
        let scale = half::f16::from_f32(1.0).to_le_bytes();
        b[0] = scale[0]; b[1] = scale[1];
        // 0x69 = [0, 1, 1, 0] → sum per byte = 2. 32 bytes * 2 = 64
        for i in 2..34 { b[i] = 0x69; }
        let mut c = vec![0.0f32; 1];
        bitnet_matmul_lut(&a, &b, &mut c, 1, k, 1);
        assert!((c[0] - 64.0).abs() < 0.1, "Expected ~64.0, got {}", c[0]);
    }

    #[test]
    fn test_bitnet_matmul_lut_vs_dequant() {
        let m = 4; let k = 128; let n = 3;
        let a: Vec<f32> = (0..m*k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        // Build packed B (column-major blocks)
        let mut b_packed = Vec::new();
        for j in 0..n {
            let mut col_packed = make_packed_column(k, 1.5);
            // Make each column slightly different
            col_packed[10] = (j * 17) as u8;
            b_packed.extend_from_slice(&col_packed);
        }

        let mut c_lut = vec![0.0f32; m * n];
        bitnet_matmul_lut(&a, &b_packed, &mut c_lut, m, k, n);

        // Dequantize B to f32, transpose to row-major, then use standard matmul
        let mut b_colmajor = vec![0.0f32; k * n];
        for j in 0..n {
            let col_start = j * ((k + BITNET_BLOCK_SIZE - 1) / BITNET_BLOCK_SIZE) * BITNET_BLOCK_BYTES;
            dequantize_i2_s(&b_packed[col_start..], &mut b_colmajor[j * k..(j + 1) * k]);
        }
        // Transpose from column-major to row-major
        let mut b_rowmajor = vec![0.0f32; k * n];
        for l in 0..k {
            for j in 0..n {
                b_rowmajor[l * n + j] = b_colmajor[j * k + l];
            }
        }
        let mut c_ref = vec![0.0f32; m * n];
        crate::kernels::simd::simd_matmul(&a, &b_rowmajor, &mut c_ref, m, k, n);

        for i in 0..c_lut.len() {
            assert!((c_lut[i] - c_ref[i]).abs() < 0.01,
                "Mismatch at {}: LUT={}, ref={}", i, c_lut[i], c_ref[i]);
        }
    }

    #[test]
    fn test_bitnet_dispatch_matches_scalar() {
        let m = 2; let k = 128; let n = 2;
        let a: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.1).collect();
        let mut b_packed = Vec::new();
        for _j in 0..n {
            b_packed.extend_from_slice(&make_packed_column(k, 0.8));
        }
        let mut c_scalar = vec![0.0f32; m * n];
        let mut c_dispatch = vec![0.0f32; m * n];
        bitnet_matmul_lut(&a, &b_packed, &mut c_scalar, m, k, n);
        bitnet_matmul_lut_dispatch(&a, &b_packed, &mut c_dispatch, m, k, n);
        for i in 0..c_scalar.len() {
            assert!((c_scalar[i] - c_dispatch[i]).abs() < 0.001,
                "Dispatch mismatch at {}: scalar={}, dispatch={}", i, c_scalar[i], c_dispatch[i]);
        }
    }
}
