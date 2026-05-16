//! Q8_0 block format and operations
//!
//! Q8_0 is a simple per-block 8-bit quantization:
//!   - 32 int8 values + f16 scale = 34 bytes per block
//!   - Dequant: value = int8 * scale
//!
//! This module provides the block representation and conversion
//! from raw GGUF Q8_0 bytes.

use half::f16;

/// One Q8_0 quantization block: 32 int8 weights + scale.
#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub scale: f32,
    pub qs: [i8; 32],
}

impl Block {
    pub const BYTES: usize = 34; // 2 (f16 scale) + 32 (int8 weights)
    pub const K: usize = 32;     // elements per block

    /// Parse a Q8_0 block from raw GGUF bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        assert_eq!(data.len(), Self::BYTES);
        let scale = f16::from_le_bytes([data[0], data[1]]).to_f32();
        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = data[2 + i] as i8;
        }
        Self { scale, qs }
    }

    /// Dequantize this block to f32 output slice.
    pub fn dequantize(&self, out: &mut [f32]) {
        assert_eq!(out.len(), Self::K);
        for i in 0..Self::K {
            out[i] = self.qs[i] as f32 * self.scale;
        }
    }

    /// Dequantize a single element by index (0..31).
    #[inline(always)]
    pub fn dequant_idx(&self, idx: usize) -> f32 {
        self.qs[idx] as f32 * self.scale
    }
}

/// Convert raw GGUF Q8_0 bytes to a flat vector of Blocks.
pub fn blocks_from_bytes(data: &[u8]) -> Vec<Block> {
    assert_eq!(data.len() % Block::BYTES, 0);
    let num_blocks = data.len() / Block::BYTES;
    (0..num_blocks)
        .map(|i| Block::from_bytes(&data[i * Block::BYTES..(i + 1) * Block::BYTES]))
        .collect()
}

/// A 2D weight matrix stored as Q8_0 blocks.
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

/// Quantize a slice of f32 values to Q8_0 blocks.
/// Returns raw bytes in GGUF Q8_0 format.
pub fn quantize_f32_to_q8_0(data: &[f32]) -> Vec<u8> {
    assert_eq!(data.len() % Block::K, 0, "data length must be multiple of 32");
    let num_blocks = data.len() / Block::K;
    let mut out = vec![0u8; num_blocks * Block::BYTES];

    for b in 0..num_blocks {
        let block_start = b * Block::K;
        let block_slice = &data[block_start..block_start + Block::K];

        // Find max absolute value for scale
        let max_abs = block_slice.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };

        // Write scale as f16
        let scale_f16 = half::f16::from_f32(scale);
        let scale_bytes = scale_f16.to_le_bytes();
        let out_base = b * Block::BYTES;
        out[out_base] = scale_bytes[0];
        out[out_base + 1] = scale_bytes[1];

        // Quantize each value
        for i in 0..Block::K {
            let q = if scale > 0.0 {
                (block_slice[i] / scale).round().clamp(-127.0, 127.0) as i8
            } else {
                0i8
            };
            out[out_base + 2 + i] = q as u8;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_roundtrip() {
        let mut data = vec![0u8; 34];
        // scale = 0.5 in f16
        let scale_bytes = half::f16::from_f32(0.5).to_le_bytes();
        data[0] = scale_bytes[0];
        data[1] = scale_bytes[1];
        // qs values: -128 to 127
        for i in 0..32 {
            data[2 + i] = (i as i8).to_le_bytes()[0];
        }

        let block = Block::from_bytes(&data);
        assert!((block.scale - 0.5).abs() < 1e-3);
        assert_eq!(block.qs[0], 0);
        assert_eq!(block.qs[31], 31);

        let mut out = [0.0f32; 32];
        block.dequantize(&mut out);
        assert!((out[0] - 0.0).abs() < 1e-3);
        assert!((out[31] - 15.5).abs() < 1e-3); // 31 * 0.5
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let q8_bytes = quantize_f32_to_q8_0(&data);
        assert_eq!(q8_bytes.len(), 2 * Block::BYTES);

        // Dequantize back
        let mut out = vec![0.0f32; 64];
        crate::kernels::dequantize_q8_0(&q8_bytes, &mut out);

        // Check roundtrip error is small
        for i in 0..64 {
            let err = (out[i] - data[i]).abs();
            assert!(err < 0.1, "large error at {}: got {}, expected {}, err={}", i, out[i], data[i], err);
        }
    }
}
