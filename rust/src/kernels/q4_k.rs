//! Q4_K block format and matrix storage
//!
//! Q4_K is a K-quant format with 256 values per block:
//!   - d (scale)     : f16 at bytes 0-1
//!   - dmin (min scale): f16 at bytes 2-3
//!   - scales        : 12 bytes at 4-15 (6-bit packed, 8 scale pairs)
//!   - qs (quantized): 128 bytes at 16-143 (4-bit nibbles)
//!   - Total: 144 bytes per block
//!
//! Dequantization within a block:
//!   - Block has 4 groups of 64 values
//!   - Each group uses 2 scale pairs (sc1,m1) and (sc2,m2)
//!   - Values 0..31  in group: qs[l] & 0x0F  with dl1 = d*sc1, min1 = dmin*m1
//!   - Values 32..63 in group: qs[l] >> 4    with dl2 = d*sc2, min2 = dmin*m2

use half::f16;

pub const QK_K: usize = 256;
pub const BLOCK_BYTES: usize = 144;

/// Extract 6-bit scale and min from the packed 12-byte scales field.
/// `j` is the scale index (0..7).
#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    assert!(j < 8);
    if j < 4 {
        (q[j] & 0x3F, q[j + 4] & 0x3F)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// One Q4_K quantization block.
#[derive(Debug, Clone)]
pub struct Block {
    pub d: f32,       // scale
    pub dmin: f32,    // min scale
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

impl Block {
    pub const BYTES: usize = BLOCK_BYTES;
    pub const K: usize = QK_K;

    /// Parse a Q4_K block from raw GGUF bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        assert_eq!(data.len(), Self::BYTES);
        let d = f16::from_le_bytes([data[0], data[1]]).to_f32();
        let dmin = f16::from_le_bytes([data[2], data[3]]).to_f32();
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&data[4..16]);
        let mut qs = [0u8; 128];
        qs.copy_from_slice(&data[16..144]);
        Self { d, dmin, scales, qs }
    }

    /// Dequantize this block to a 256-element f32 output slice.
    ///
    /// Dispatches to AVX2+FMA on x86_64 when available, otherwise falls
    /// back to the scalar reference implementation.
    pub fn dequantize(&self, out: &mut [f32]) {
        assert_eq!(out.len(), Self::K);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { self.dequantize_avx2(out) };
                return;
            }
        }

        self.dequantize_scalar(out);
    }

    /// Scalar reference dequantize — identical results to the original
    /// implementation. Kept as fallback for non-AVX2 platforms and as
    /// ground truth for tests.
    #[inline]
    pub fn dequantize_scalar(&self, out: &mut [f32]) {
        assert_eq!(out.len(), Self::K);
        let mut q_off = 0;
        let mut is = 0;
        let mut idx = 0;

        for _j in 0..(Self::K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, &self.scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &self.scales);
            let dl1 = self.d * sc1 as f32;
            let dl2 = self.d * sc2 as f32;
            let min1 = self.dmin * m1 as f32;
            let min2 = self.dmin * m2 as f32;

            for l in 0..32 {
                out[idx + l] = dl1 * (self.qs[q_off + l] & 0x0F) as f32 - min1;
                out[idx + l + 32] = dl2 * (self.qs[q_off + l] >> 4) as f32 - min2;
            }
            idx += 64;
            q_off += 32;
            is += 2;
        }
    }

    /// AVX2+FMA dequantize: processes 8 nibble-packed bytes at a time.
    ///
    /// Per group of 64: 4 chunks of 8 bytes. Each chunk:
    ///   - Load 8 u8 from qs[q_off+chunk*8]
    ///   - Low nibbles (qs & 0x0F) → 8 f32 → dl1 * lo - min1 → store to out[idx+chunk*8]
    ///   - High nibbles (qs >> 4 & 0x0F) → 8 f32 → dl2 * hi - min2 → store to out[idx+32+chunk*8]
    ///
    /// Uses `_mm_cvtepu8_epi32` to zero-extend 8 u8 → 8 i32, then
    /// `_mm256_cvtepi32_ps` to convert to f32. FMA does the scale+offset.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn dequantize_avx2(&self, out: &mut [f32]) {
        use std::arch::x86_64::*;

        let nibble_mask = _mm_set1_epi8(0x0F);
        let mut q_off = 0;
        let mut idx = 0;

        for group in 0..4 {
            let (sc1, m1) = get_scale_min_k4(group * 2, &self.scales);
            let (sc2, m2) = get_scale_min_k4(group * 2 + 1, &self.scales);
            let dl1 = self.d * sc1 as f32;
            let dl2 = self.d * sc2 as f32;
            let min1 = self.dmin * m1 as f32;
            let min2 = self.dmin * m2 as f32;

            let dl1_vec = _mm256_set1_ps(dl1);
            let dl2_vec = _mm256_set1_ps(dl2);
            let min1_vec = _mm256_set1_ps(min1);
            let min2_vec = _mm256_set1_ps(min2);

            for chunk in 0..4 {
                let off = chunk * 8;
                // Load 8 bytes from qs
                let raw = _mm_loadl_epi64(self.qs.as_ptr().add(q_off + off) as *const __m128i);

                // Low nibbles: raw & 0x0F, then zero-extend 8 u8 → 8 i32
                let lo = _mm_and_si128(raw, nibble_mask);
                let lo_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
                // out[idx+off] = dl1 * lo - min1
                let lo_res = _mm256_fmsub_ps(dl1_vec, lo_f32, min1_vec);
                _mm256_storeu_ps(out.as_mut_ptr().add(idx + off), lo_res);

                // High nibbles: (raw >> 4) & 0x0F, then zero-extend 8 u8 → 8 i32
                let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), nibble_mask);
                let hi_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
                // out[idx+32+off] = dl2 * hi - min2
                let hi_res = _mm256_fmsub_ps(dl2_vec, hi_f32, min2_vec);
                _mm256_storeu_ps(out.as_mut_ptr().add(idx + 32 + off), hi_res);
            }

            idx += 64;
            q_off += 32;
        }
    }
}

/// Convert raw GGUF Q4_K bytes to a flat vector of Blocks.
pub fn blocks_from_bytes(data: &[u8]) -> Vec<Block> {
    assert_eq!(data.len() % Block::BYTES, 0);
    let num_blocks = data.len() / Block::BYTES;
    (0..num_blocks)
        .map(|i| Block::from_bytes(&data[i * Block::BYTES..(i + 1) * Block::BYTES]))
        .collect()
}

/// A 2D weight matrix stored as Q4_K blocks.
/// Shape: [rows, cols] where cols must be a multiple of 256.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    /// Blocks stored row-major: blocks[row * blocks_per_row + block_in_row]
    pub blocks: Vec<Block>,
}

impl Matrix {
    pub fn blocks_per_row(&self) -> usize {
        self.cols / Block::K
    }

    /// Dequantize the entire matrix to f32 [rows, cols].
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        let bpr = self.blocks_per_row();
        for row in 0..self.rows {
            for b in 0..bpr {
                let block = &self.blocks[row * bpr + b];
                let base = row * self.cols + b * Block::K;
                block.dequantize(&mut out[base..base + Block::K]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_roundtrip() {
        // Create a synthetic Q4_K block
        let mut data = vec![0u8; 144];
        // d = 0.5, dmin = 0.1
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        data[0] = d_bytes[0];
        data[1] = d_bytes[1];
        let dmin_bytes = half::f16::from_f32(0.1).to_le_bytes();
        data[2] = dmin_bytes[0];
        data[3] = dmin_bytes[1];
        // scales: all 1 (simple)
        for i in 4..16 {
            data[i] = 0x01;
        }
        // qs: alternating nibbles 0..15
        for i in 0..128 {
            data[16 + i] = ((i % 16) << 4 | (i % 16)) as u8;
        }

        let block = Block::from_bytes(&data);
        assert!((block.d - 0.5).abs() < 1e-3);
        assert!((block.dmin - 0.1).abs() < 1e-3);

        let mut out = [0.0f32; 256];
        block.dequantize(&mut out);

        // First value: dl1 = 0.5 * 1 = 0.5, min1 = 0.1 * 1 = 0.1
        // nibble = 0, so value = 0.5 * 0 - 0.1 = -0.1
        assert!((out[0] - (-0.1)).abs() < 1e-3, "out[0] = {}", out[0]);

        // Value at index 31 (last of first half-group): nibble = 15
        // value = 0.5 * 15 - 0.1 = 7.5 - 0.1 = 7.4
        assert!((out[31] - 7.4).abs() < 1e-2, "out[31] = {}", out[31]);
    }

    /// Verify AVX2 dequantize produces identical results to scalar dequantize
    /// across a variety of blocks with different scales, mins, and nibble values.
    #[test]
    fn test_avx2_matches_scalar() {
        // Build 100 blocks with random-ish but deterministic parameters
        let mut rng_seed: u64 = 0x1234_5678_DEAD_BEEF;
        let mut next_seed = || {
            // xorshift64
            rng_seed ^= rng_seed << 13;
            rng_seed ^= rng_seed >> 7;
            rng_seed ^= rng_seed << 17;
            rng_seed
        };

        for _ in 0..100 {
            let mut data = vec![0u8; 144];
            // Random d and dmin in reasonable range
            let d = (next_seed() % 2000) as f32 / 1000.0;       // 0..2
            let dmin = (next_seed() % 1000) as f32 / 1000.0;    // 0..1
            let d_bytes = half::f16::from_f32(d).to_le_bytes();
            let dmin_bytes = half::f16::from_f32(dmin).to_le_bytes();
            data[0..2].copy_from_slice(&d_bytes);
            data[2..4].copy_from_slice(&dmin_bytes);
            // Random scales (6-bit values)
            for i in 4..16 {
                data[i] = (next_seed() % 64) as u8;
            }
            // Random quantized values
            for i in 16..144 {
                data[i] = (next_seed() % 256) as u8;
            }

            let block = Block::from_bytes(&data);

            let mut out_scalar = [0.0f32; 256];
            let mut out_avx2 = [0.0f32; 256];
            block.dequantize_scalar(&mut out_scalar);
            block.dequantize(&mut out_avx2);  // will dispatch to AVX2

            for i in 0..256 {
                let diff = (out_scalar[i] - out_avx2[i]).abs();
                assert!(diff < 1e-5,
                    "Mismatch at index {} for block (d={}, dmin={}): scalar={}, avx2={}, diff={}",
                    i, d, dmin, out_scalar[i], out_avx2[i], diff);
            }
        }
    }
}
