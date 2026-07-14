//! Q6_K block format and matrix storage
//!
//! Q6_K is a K-quant format with 256 values per block:
//!   - ql     : 128 bytes at 0-127 (low 4 bits of 6-bit values)
//!   - qh     : 64 bytes at 128-191 (high 2 bits of 6-bit values)
//!   - scales : 16 bytes at 192-207 (signed int8 scale multipliers)
//!   - d      : f16 at bytes 208-209
//!   - Total: 210 bytes per block
//!
//! Dequantization: value = d * scales[sc] * (q - 32)

use half::f16;

pub const QK_K: usize = 256;
pub const BLOCK_BYTES: usize = 210;

/// One Q6_K quantization block.
#[derive(Debug, Clone)]
pub struct Block {
    pub ql: [u8; 128],
    pub qh: [u8; 64],
    pub scales: [u8; 16],
    pub d: f32,
}

impl Block {
    pub const BYTES: usize = BLOCK_BYTES;
    pub const K: usize = QK_K;

    /// Parse a Q6_K block from raw GGUF bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        assert_eq!(data.len(), Self::BYTES);
        let mut ql = [0u8; 128];
        ql.copy_from_slice(&data[0..128]);
        let mut qh = [0u8; 64];
        qh.copy_from_slice(&data[128..192]);
        let mut scales = [0u8; 16];
        scales.copy_from_slice(&data[192..208]);
        let d = f16::from_le_bytes([data[208], data[209]]).to_f32();
        Self { ql, qh, scales, d }
    }

    /// Dequantize this block to a 256-element f32 output slice.
    pub fn dequantize(&self, out: &mut [f32]) {
        assert_eq!(out.len(), Self::K);
        self.dequantize_scalar(out);
    }

    /// Scalar reference dequantize — kept as fallback and ground truth.
    #[inline]
    pub fn dequantize_scalar(&self, out: &mut [f32]) {
        assert_eq!(out.len(), Self::K);
        let mut ql_off = 0;
        let mut qh_off = 0;
        let mut sc_off = 0;
        let mut idx = 0;

        for _ in 0..(Self::K / 128) {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((self.ql[ql_off + l] & 0x0F) as i8
                    | (((self.qh[qh_off + l] >> 0) & 3) as i8) << 4) - 32;
                let q2 = ((self.ql[ql_off + l + 32] & 0x0F) as i8
                    | (((self.qh[qh_off + l] >> 2) & 3) as i8) << 4) - 32;
                let q3 = ((self.ql[ql_off + l] >> 4) as i8
                    | (((self.qh[qh_off + l] >> 4) & 3) as i8) << 4) - 32;
                let q4 = ((self.ql[ql_off + l + 32] >> 4) as i8
                    | (((self.qh[qh_off + l] >> 6) & 3) as i8) << 4) - 32;

                out[idx + l + 0] = self.d * self.scales[sc_off + is + 0] as i8 as f32 * q1 as f32;
                out[idx + l + 32] = self.d * self.scales[sc_off + is + 2] as i8 as f32 * q2 as f32;
                out[idx + l + 64] = self.d * self.scales[sc_off + is + 4] as i8 as f32 * q3 as f32;
                out[idx + l + 96] = self.d * self.scales[sc_off + is + 6] as i8 as f32 * q4 as f32;
            }
            idx += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

/// Convert raw GGUF Q6_K bytes to a flat vector of Blocks.
pub fn blocks_from_bytes(data: &[u8]) -> Vec<Block> {
    assert_eq!(data.len() % Block::BYTES, 0);
    let num_blocks = data.len() / Block::BYTES;
    (0..num_blocks)
        .map(|i| Block::from_bytes(&data[i * Block::BYTES..(i + 1) * Block::BYTES]))
        .collect()
}

/// A 2D weight matrix stored as Q6_K blocks.
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
        let mut data = vec![0u8; 210];
        // d = 0.5
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        data[208] = d_bytes[0];
        data[209] = d_bytes[1];
        // scales: all 1
        for i in 192..208 {
            data[i] = 1;
        }
        // ql: all low nibbles = 0, high nibbles = 0
        // qh: all 0
        // q = (0 | 0) - 32 = -32
        // value = 0.5 * 1 * (-32) = -16

        let block = Block::from_bytes(&data);
        assert!((block.d - 0.5).abs() < 1e-3);

        let mut out = [0.0f32; 256];
        block.dequantize(&mut out);
        assert!((out[0] - (-16.0)).abs() < 1e-2, "out[0] = {}", out[0]);
        assert!((out[255] - (-16.0)).abs() < 1e-2, "out[255] = {}", out[255]);
    }
}
