//! Quantized kernel dequantization + SIMD compute kernels
//!
//! Converts GGUF quantized blocks to f32 tensors for computation.
//! simd.rs provides architecture-specific SIMD matmul and element-wise ops.

pub mod int8_gemm;
pub mod q8_0;
pub mod simd;

use half::f16;

pub const QK_K: usize = 256;

/// Dequantize Q4_0 blocks to f32
pub fn dequantize_q4_0(data: &[u8], out: &mut [f32]) {
    let block_size = 32;
    let group_size = 18;
    let num_blocks = out.len() / block_size;

    for i in 0..num_blocks {
        let start = i * group_size;
        let block = &data[start..start + group_size];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

        for j in 0..16 {
            let qs = block[2 + j];
            let q0 = (qs & 0x0F) as f32 - 8.0;
            let q1 = (qs >> 4) as f32 - 8.0;
            out[i * block_size + j] = q0 * scale;
            out[i * block_size + j + 16] = q1 * scale;
        }
    }
}

/// Dequantize Q8_0 blocks to f32
pub fn dequantize_q8_0(data: &[u8], out: &mut [f32]) {
    let block_size = 32;
    let group_size = 34;
    let num_blocks = out.len() / block_size;

    for i in 0..num_blocks {
        let start = i * group_size;
        let block = &data[start..start + group_size];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

        for j in 0..32 {
            out[i * block_size + j] = (block[2 + j] as i8) as f32 * scale;
        }
    }
}

/// Unpack 6-bit scale/min for K-quant sub-block j
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 0x3F, q[j + 4] & 0x3F)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantize Q4_K blocks to f32
pub fn dequantize_q4_k(data: &[u8], out: &mut [f32]) {
    let bytes_per_block = 144;
    let num_blocks = out.len() / QK_K;

    for i in 0..num_blocks {
        let block = &data[i * bytes_per_block..(i + 1) * bytes_per_block];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut q_off = 0;
        let mut is = 0;
        let mut idx = i * QK_K;

        for _j in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let dl1 = d * sc1 as f32;
            let dl2 = d * sc2 as f32;
            let min1 = dmin * m1 as f32;
            let min2 = dmin * m2 as f32;

            for l in 0..32 {
                out[idx + l] = dl1 * (qs[q_off + l] & 0x0F) as f32 - min1;
                out[idx + l + 32] = dl2 * (qs[q_off + l] >> 4) as f32 - min2;
            }
            idx += 64;
            q_off += 32;
            is += 2;
        }
    }
}

/// Dequantize Q5_K blocks to f32
pub fn dequantize_q5_k(data: &[u8], out: &mut [f32]) {
    let bytes_per_block = 176;
    let num_blocks = out.len() / QK_K;

    for i in 0..num_blocks {
        let block = &data[i * bytes_per_block..(i + 1) * bytes_per_block];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..176];

        let mut ql_off = 0;
        let mut is = 0;
        let (mut u1, mut u2) = (1u8, 2u8);
        let mut idx = i * QK_K;

        for _j in 0..(QK_K / 64) {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let dl1 = d * sc1 as f32;
            let dl2 = d * sc2 as f32;
            let min1 = dmin * m1 as f32;
            let min2 = dmin * m2 as f32;

            for l in 0..32 {
                let mut q = (ql[ql_off + l] & 0x0F) as u8;
                if qh[l] & u1 != 0 { q += 16; }
                out[idx + l] = dl1 * q as f32 - min1;

                let mut q = (ql[ql_off + l] >> 4) as u8;
                if qh[l] & u2 != 0 { q += 16; }
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

/// Dequantize Q6_K blocks to f32
pub fn dequantize_q6_k(data: &[u8], out: &mut [f32]) {
    let bytes_per_block = 210;
    let num_blocks = out.len() / QK_K;

    for i in 0..num_blocks {
        let block = &data[i * bytes_per_block..(i + 1) * bytes_per_block];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

        let mut ql_off = 0;
        let mut qh_off = 0;
        let mut sc_off = 0;
        let mut idx = i * QK_K;

        for _n in 0..(QK_K / 128) {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0x0F) as i8 | (((qh[qh_off + l] >> 0) & 3) as i8) << 4) - 32;
                let q2 = ((ql[ql_off + l + 32] & 0x0F) as i8 | (((qh[qh_off + l] >> 2) & 3) as i8) << 4) - 32;
                let q3 = ((ql[ql_off + l] >> 4) as i8 | (((qh[qh_off + l] >> 4) & 3) as i8) << 4) - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) as i8 | (((qh[qh_off + l] >> 6) & 3) as i8) << 4) - 32;

                out[idx + l + 0] = d * scales[sc_off + is + 0] as i8 as f32 * q1 as f32;
                out[idx + l + 32] = d * scales[sc_off + is + 2] as i8 as f32 * q2 as f32;
                out[idx + l + 64] = d * scales[sc_off + is + 4] as i8 as f32 * q3 as f32;
                out[idx + l + 96] = d * scales[sc_off + is + 6] as i8 as f32 * q4 as f32;
            }
            idx += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

/// Non-linear lookup table for IQ4_NL (improved 4-bit quantization).
///
/// Source: llama.cpp ggml-common.h
/// These 16 values are optimized for typical weight distributions.
const IQ4NL_TABLE: [f32; 16] = [
    -1.0, -0.6962, -0.5251, -0.3952, -0.2893, -0.1957, -0.1107, -0.0322,
     0.0322,  0.1107,  0.1957,  0.2893,  0.3952,  0.5251,  0.6962,  1.0,
];

/// Dequantize IQ4_NL blocks to f32.
///
/// Block layout (18 bytes for 32 values = 4.5 bpw):
///   - bytes 0..1: scale as f16
///   - bytes 2..17: 32 nibbles packed into 16 bytes
///
/// Dequant: value = scale * iq4nl_table[nibble]
pub fn dequantize_iq4_nl(data: &[u8], out: &mut [f32]) {
    const BLOCK_SIZE: usize = 32;
    const GROUP_SIZE: usize = 18; // 2 (scale) + 16 (nibbles)
    let num_blocks = out.len() / BLOCK_SIZE;

    for i in 0..num_blocks {
        let start = i * GROUP_SIZE;
        let block = &data[start..start + GROUP_SIZE];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

        for j in 0..16 {
            let qs = block[2 + j];
            let nibble0 = (qs & 0x0F) as usize;
            let nibble1 = (qs >> 4) as usize;
            out[i * BLOCK_SIZE + j * 2]     = scale * IQ4NL_TABLE[nibble0];
            out[i * BLOCK_SIZE + j * 2 + 1] = scale * IQ4NL_TABLE[nibble1];
        }
    }
}

#[cfg(test)]

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_roundtrip() {
        let mut data = vec![0u8; 18];
        data[0] = 0x00; data[1] = 0x3C; // scale = 1.0 in f16
        data[2] = 0x89; // q0=9, q1=8
        let mut out = vec![0.0f32; 32];
        dequantize_q4_0(&data, &mut out);
        assert!((out[0] - 1.0).abs() < 0.01);  // 9 - 8 = 1
        assert!((out[16] - 0.0).abs() < 0.01); // 8 - 8 = 0
    }

    #[test]
    fn test_q4_k_block_size() {
        let data = vec![0u8; 144];
        let mut out = vec![0.0f32; 256];
        dequantize_q4_k(&data, &mut out);
        // With zero data, d=0, so all outputs should be 0
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_q6_k_block_size() {
        let data = vec![0u8; 210];
        let mut out = vec![0.0f32; 256];
        dequantize_q6_k(&data, &mut out);
        // With zero data, d=0, so all outputs should be 0
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_iq4_nl_basic() {
        let mut data = vec![0u8; 18];
        // scale = 1.0 in f16
        data[0] = 0x00; data[1] = 0x3C;
        // nibble0=0  (-1.0), nibble1=15 (1.0)
        data[2] = 0xF0;
        let mut out = vec![0.0f32; 32];
        dequantize_iq4_nl(&data, &mut out);
        assert!((out[0] - -1.0).abs() < 0.001);
        assert!((out[1] - 1.0).abs() < 0.001);
    }
}
