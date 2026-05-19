//! Q5_K block format and matrix storage
//!
//! Q5_K is a K-quant format with 256 values per block:
//!   - d (scale)     : f16 at bytes 0-1
//!   - dmin (min scale): f16 at bytes 2-3
//!   - scales        : 12 bytes at 4-15 (6-bit packed, 8 scale pairs)
//!   - qh (high bits): 32 bytes at 16-47 (1 high bit per value)
//!   - ql (low bits) : 128 bytes at 48-175 (4-bit nibbles)
//!   - Total: 176 bytes per block
//!
//! Dequantization within a block:
//!   - Block has 4 groups of 64 values
//!   - Each group uses 2 scale pairs (sc1,m1) and (sc2,m2)
//!   - Values 0..31  in group: ql[l] & 0x0F + (qh[l] & u1 ? 16 : 0) with dl1 = d*sc1, min1 = dmin*m1
//!   - Values 32..63 in group: ql[l] >> 4    + (qh[l] & u2 ? 16 : 0) with dl2 = d*sc2, min2 = dmin*m2

use half::f16;
use super::q4_k::get_scale_min_k4;

pub const QK_K: usize = 256;
pub const BLOCK_BYTES: usize = 176;

/// One Q5_K quantization block.
#[derive(Debug, Clone)]
pub struct Block {
    pub d: f32,
    pub dmin: f32,
    pub scales: [u8; 12],
    pub qh: [u8; 32],
    pub ql: [u8; 128],
}

impl Block {
    pub const BYTES: usize = BLOCK_BYTES;
    pub const K: usize = QK_K;

    pub fn from_bytes(data: &[u8]) -> Self {
        assert_eq!(data.len(), Self::BYTES);
        let d = f16::from_le_bytes([data[0], data[1]]).to_f32();
        let dmin = f16::from_le_bytes([data[2], data[3]]).to_f32();
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&data[4..16]);
        let mut qh = [0u8; 32];
        qh.copy_from_slice(&data[16..48]);
        let mut ql = [0u8; 128];
        ql.copy_from_slice(&data[48..176]);
        Self { d, dmin, scales, qh, ql }
    }

    pub fn dequantize(&self, out: &mut [f32]) {
        assert_eq!(out.len(), Self::K);
        let mut ql_off = 0;
        let mut is = 0;
        let (mut u1, mut u2) = (1u8, 2u8);
        let mut idx = 0;

        for _j in 0..(Self::K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, &self.scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &self.scales);
            let dl1 = self.d * sc1 as f32;
            let dl2 = self.d * sc2 as f32;
            let min1 = self.dmin * m1 as f32;
            let min2 = self.dmin * m2 as f32;

            for l in 0..32 {
                let mut q = (self.ql[ql_off + l] & 0x0F) as u8;
                if self.qh[l] & u1 != 0 { q += 16; }
                out[idx + l] = dl1 * q as f32 - min1;

                let mut q = (self.ql[ql_off + l] >> 4) as u8;
                if self.qh[l] & u2 != 0 { q += 16; }
                out[idx + l + 32] = dl2 * q as f32 - min2;
            }
            idx += 64;
            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

pub fn blocks_from_bytes(data: &[u8]) -> Vec<Block> {
    assert_eq!(data.len() % Block::BYTES, 0);
    let num_blocks = data.len() / Block::BYTES;
    (0..num_blocks)
        .map(|i| Block::from_bytes(&data[i * Block::BYTES..(i + 1) * Block::BYTES]))
        .collect()
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub blocks: Vec<Block>,
}

impl Matrix {
    pub fn blocks_per_row(&self) -> usize {
        self.cols / Block::K
    }

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
        let mut data = vec![0u8; 176];
        let d_bytes = half::f16::from_f32(0.5).to_le_bytes();
        data[0] = d_bytes[0];
        data[1] = d_bytes[1];
        let dmin_bytes = half::f16::from_f32(0.1).to_le_bytes();
        data[2] = dmin_bytes[0];
        data[3] = dmin_bytes[1];
        for i in 4..16 {
            data[i] = 0x01;
        }
        // qh: all 0 (no high bits set)
        // ql: alternating nibbles
        for i in 0..128 {
            data[48 + i] = ((i % 16) << 4 | (i % 16)) as u8;
        }

        let block = Block::from_bytes(&data);
        assert!((block.d - 0.5).abs() < 1e-3);
        assert!((block.dmin - 0.1).abs() < 1e-3);

        let mut out = [0.0f32; 256];
        block.dequantize(&mut out);

        // First value: dl1 = 0.5 * 1 = 0.5, min1 = 0.1 * 1 = 0.1
        // q = 0, so value = 0.5 * 0 - 0.1 = -0.1
        assert!((out[0] - (-0.1)).abs() < 1e-3, "out[0] = {}", out[0]);
    }
}
