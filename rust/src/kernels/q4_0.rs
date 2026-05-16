//! Q4_0 block format and operations
//!
//! Q4_0 is 4-bit quantization with per-block scale:
//!   - 32 weights packed into 16 bytes (4 bits each)
//!   - f16 scale = 2 bytes
//!   - Total: 18 bytes for 32 weights = 4.5 bits/weight
//!
//! Dequant: value = (q4_value - 8) * scale
//!   (q4 values are 0..15, subtract 8 to get signed -8..7)

use half::f16;

/// One Q4_0 quantization block: 32 4-bit weights + scale.
#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub scale: f32,
    /// 32 nibbles (4-bit values), packed into 16 bytes.
    /// qs[i] stores values 0..15. Dequant: (qs[i] as f32 - 8.0) * scale
    pub qs: [u8; 16],
}

impl Block {
    pub const BYTES: usize = 18; // 2 (f16 scale) + 16 (32×4-bit)
    pub const K: usize = 32;     // elements per block

    /// Parse a Q4_0 block from raw GGUF bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        assert_eq!(data.len(), Self::BYTES);
        let scale = f16::from_le_bytes([data[0], data[1]]).to_f32();
        let mut qs = [0u8; 16];
        qs.copy_from_slice(&data[2..18]);
        Self { scale, qs }
    }

    /// Dequantize this block to f32 output slice.
    pub fn dequantize(&self, out: &mut [f32]) {
        assert_eq!(out.len(), Self::K);
        let scale = self.scale;
        for i in 0..16 {
            let byte = self.qs[i];
            let low = (byte & 0x0F) as f32;
            let high = ((byte >> 4) & 0x0F) as f32;
            out[i * 2] = (low - 8.0) * scale;
            out[i * 2 + 1] = (high - 8.0) * scale;
        }
    }

    /// Dequantize a single element by index (0..31).
    #[inline(always)]
    pub fn dequant_idx(&self, idx: usize) -> f32 {
        let byte = self.qs[idx / 2];
        let nibble = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        (nibble as f32 - 8.0) * self.scale
    }
}

/// Convert raw GGUF Q4_0 bytes to a flat vector of Blocks.
pub fn blocks_from_bytes(data: &[u8]) -> Vec<Block> {
    assert_eq!(data.len() % Block::BYTES, 0);
    let num_blocks = data.len() / Block::BYTES;
    (0..num_blocks)
        .map(|i| Block::from_bytes(&data[i * Block::BYTES..(i + 1) * Block::BYTES]))
        .collect()
}

/// A 2D weight matrix stored as Q4_0 blocks.
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

/// Quantize a slice of f32 values to Q4_0 blocks.
/// Returns raw bytes in GGUF Q4_0 format.
pub fn quantize_f32_to_q4_0(data: &[f32]) -> Vec<u8> {
    assert_eq!(data.len() % Block::K, 0, "data length must be multiple of 32");
    let num_blocks = data.len() / Block::K;
    let mut out = vec![0u8; num_blocks * Block::BYTES];

    for b in 0..num_blocks {
        let block_start = b * Block::K;
        let block_slice = &data[block_start..block_start + Block::K];

        // Find max absolute value for scale
        let max_abs = block_slice.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
        // Q4_0 range is -8..7, so scale = max_abs / 8.0
        let scale = if max_abs > 0.0 { max_abs / 8.0 } else { 0.0 };

        // Write scale as f16
        let scale_f16 = half::f16::from_f32(scale);
        let scale_bytes = scale_f16.to_le_bytes();
        let out_base = b * Block::BYTES;
        out[out_base] = scale_bytes[0];
        out[out_base + 1] = scale_bytes[1];

        // Quantize each value to 4-bit unsigned (0..15)
        // value = (q - 8) * scale  =>  q = round(value / scale) + 8
        for i in 0..16 {
            let q0 = if scale > 0.0 {
                (block_slice[i * 2] / scale + 8.0).round().clamp(0.0, 15.0) as u8
            } else {
                8u8 // zero -> middle of range
            };
            let q1 = if scale > 0.0 {
                (block_slice[i * 2 + 1] / scale + 8.0).round().clamp(0.0, 15.0) as u8
            } else {
                8u8
            };
            out[out_base + 2 + i] = (q1 << 4) | q0;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_roundtrip() {
        let mut data = vec![0u8; 18];
        // scale = 0.5 in f16
        let scale_bytes = half::f16::from_f32(0.5).to_le_bytes();
        data[0] = scale_bytes[0];
        data[1] = scale_bytes[1];
        // qs values: 8, 9, 10, ... (meaning 0, 0.5, 1.0, ...)
        for i in 0..16 {
            let low = (8 + i * 2) as u8;
            let high = (8 + i * 2 + 1).min(15) as u8;
            data[2 + i] = (high << 4) | low;
        }

        let block = Block::from_bytes(&data);
        assert!((block.scale - 0.5).abs() < 1e-3);

        let mut out = [0.0f32; 32];
        block.dequantize(&mut out);
        assert!((out[0] - 0.0).abs() < 1e-3);      // (8-8)*0.5 = 0
        assert!((out[1] - 0.5).abs() < 1e-3);      // (9-8)*0.5 = 0.5
        assert!((out[2] - 1.0).abs() < 1e-3);      // (10-8)*0.5 = 1.0
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let q4_bytes = quantize_f32_to_q4_0(&data);
        assert_eq!(q4_bytes.len(), 2 * Block::BYTES);

        // Dequantize back
        let mut out = vec![0.0f32; 64];
        let blocks = blocks_from_bytes(&q4_bytes);
        for (b, block) in blocks.iter().enumerate() {
            block.dequantize(&mut out[b * 32..(b + 1) * 32]);
        }

        // Check roundtrip error is small (Q4_0 is 4-bit; ~0.5 abs error is acceptable)
        for i in 0..64 {
            let err = (out[i] - data[i]).abs();
            assert!(err < 0.5, "large error at {}: got {}, expected {}, err={}", i, out[i], data[i], err);
        }
    }
}
