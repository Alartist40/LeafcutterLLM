//! IQ4_NL block format and matrix storage
//!
//! IQ4_NL is an "improved" 4-bit quantization using a non-linear lookup table:
//!   - scale : f16 at bytes 0-1
//!   - qs    : 16 bytes at 2-17 (32 nibbles packed)
//!   - Total: 18 bytes per block (4.5 bpw)
//!
//! Dequantization: value = scale * IQ4NL_TABLE[nibble]

use half::f16;

/// Non-linear lookup table for IQ4_NL.
/// Source: llama.cpp ggml-common.h
const IQ4NL_TABLE: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
       1.0,   13.0,  25.0,  38.0,  53.0,  69.0,  89.0, 113.0,
];

/// One IQ4_NL quantization block.
#[derive(Debug, Clone)]
pub struct Block {
    pub scale: f32,
    pub qs: [u8; 16],
}

impl Block {
    pub const BYTES: usize = 18;
    pub const K: usize = 32;

    /// Parse an IQ4_NL block from raw GGUF bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        assert_eq!(data.len(), Self::BYTES);
        let scale = f16::from_le_bytes([data[0], data[1]]).to_f32();
        let mut qs = [0u8; 16];
        qs.copy_from_slice(&data[2..18]);
        Self { scale, qs }
    }

    /// Dequantize this block to a 32-element f32 output slice.
    pub fn dequantize(&self, out: &mut [f32]) {
        assert_eq!(out.len(), Self::K);
        for j in 0..16 {
            let byte = self.qs[j];
            let nibble0 = (byte & 0x0F) as usize;
            let nibble1 = (byte >> 4) as usize;
            out[j * 2]     = self.scale * IQ4NL_TABLE[nibble0];
            out[j * 2 + 1] = self.scale * IQ4NL_TABLE[nibble1];
        }
    }
}

/// Convert raw GGUF IQ4_NL bytes to a flat vector of Blocks.
pub fn blocks_from_bytes(data: &[u8]) -> Vec<Block> {
    assert_eq!(data.len() % Block::BYTES, 0);
    let num_blocks = data.len() / Block::BYTES;
    (0..num_blocks)
        .map(|i| Block::from_bytes(&data[i * Block::BYTES..(i + 1) * Block::BYTES]))
        .collect()
}

/// A 2D weight matrix stored as IQ4_NL blocks.
/// Shape: [rows, cols] where cols must be a multiple of 32.
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
        let mut data = vec![0u8; 18];
        let scale_bytes = half::f16::from_f32(0.5).to_le_bytes();
        data[0] = scale_bytes[0];
        data[1] = scale_bytes[1];
        // qs: low nibble = 15, high nibble = 0 for all bytes
        for i in 0..16 {
            data[2 + i] = 0x0F;
        }

        let block = Block::from_bytes(&data);
        assert!((block.scale - 0.5).abs() < 1e-3);

        let mut out = [0.0f32; 32];
        block.dequantize(&mut out);

        // nibble0 (low) = 15 → table[15] = 113.0, value = 0.5 * 113.0 = 56.5
        assert!((out[0] - 56.5).abs() < 1e-3, "out[0] = {}", out[0]);
        // nibble1 (high) = 0 → table[0] = -127.0, value = 0.5 * -127.0 = -63.5
        assert!((out[1] - (-63.5)).abs() < 1e-3, "out[1] = {}", out[1]);
    }
}
