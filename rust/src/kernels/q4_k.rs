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
    pub fn dequantize(&self, out: &mut [f32]) {
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
}
