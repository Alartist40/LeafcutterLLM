//! Quantized kernel dequantization + SIMD compute kernels
//!
//! Converts GGUF quantized blocks to f32 tensors for computation.
//! simd.rs provides architecture-specific SIMD matmul and element-wise ops.

pub mod bitnet_lut;
pub mod int8_gemm;
pub mod iq4_nl;
pub mod iq4_nl_gemm;
pub mod q4_0;
pub mod q4_k;
pub mod q4_k_gemm;
pub mod q5_k;
pub mod q5_k_gemm;
pub mod q6_k;
pub mod q6_k_gemm;
pub mod q6_k_fused;
pub mod q8_0;
pub mod q8_k;
pub mod q8_k_gemm;
pub mod quant_gemm_common;
pub mod simd;

use half::{f16, bf16};

pub const QK_K: usize = 256;

/// Dequantize Q4_0 blocks to f32
/// Layout matches GGUF spec: each block is 18 bytes (scale f16 + 16 bytes of nibbles).
/// Within each byte, low nibble = first element, high nibble = second element.
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
            let low = (qs & 0x0F) as f32;
            let high = ((qs >> 4) & 0x0F) as f32;
            // Byte-interleaved: two consecutive elements come from one byte
            out[i * block_size + j * 2] = (low - 8.0) * scale;
            out[i * block_size + j * 2 + 1] = (high - 8.0) * scale;
        }
    }
}

/// Dequantize Q4_1 blocks to f32
/// Block layout (20 bytes for 32 values = 5.0 bpw):
///   - bytes 0..1: scale `d` as f16
///   - bytes 2..3: min `m` as f16
///   - bytes 4..19: 32 nibbles packed into 16 bytes
/// Dequant: value = d * nibble + m
pub fn dequantize_q4_1(data: &[u8], out: &mut [f32]) {
    let block_size = 32;
    let group_size = 20;
    let num_blocks = out.len() / block_size;

    for i in 0..num_blocks {
        let start = i * group_size;
        let block = &data[start..start + group_size];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let min = f16::from_le_bytes([block[2], block[3]]).to_f32();

        for j in 0..16 {
            let qs = block[4 + j];
            let q0 = (qs & 0x0F) as f32;
            let q1 = (qs >> 4) as f32;
            out[i * block_size + j] = q0 * scale + min;
            out[i * block_size + j + 16] = q1 * scale + min;
        }
    }
}

/// Dequantize BF16 elements to f32
pub fn dequantize_bf16(data: &[u8], out: &mut [f32]) {
    let count = out.len();
    for i in 0..count {
        let bytes = [data[i * 2], data[i * 2 + 1]];
        out[i] = bf16::from_le_bytes(bytes).to_f32();
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

/// Dequantize Q8_K blocks to f32
pub fn dequantize_q8_k(data: &[u8], out: &mut [f32]) {
    let bytes_per_block = 292;
    let num_blocks = out.len() / QK_K;

    for i in 0..num_blocks {
        let block = &data[i * bytes_per_block..(i + 1) * bytes_per_block];
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let qs = &block[4..260];

        let base = i * QK_K;
        for j in 0..QK_K {
            out[base + j] = d * (qs[j] as i8) as f32;
        }
    }
}

/// Non-linear lookup table for IQ4_NL (improved 4-bit quantization).
///
/// Source: llama.cpp ggml-common.h — kvalues_iq4nl
/// These 16 values are the raw integer-ish lattice points; the block scale
/// converts them to the final weight.
const IQ4NL_TABLE: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
       1.0,   13.0,  25.0,  38.0,  53.0,  69.0,  89.0, 113.0,
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

/// Non-linear lookup table for IQ4_NL / IQ4_XS quant types.
/// Values are symmetric-ish around zero, optimized for language model weights.
static KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Dequantize IQ4_XS blocks to f32.
///
/// Block layout (136 bytes for 256 values = 4.25 bpw):
///   - bytes 0..1:   scale `d` as f16
///   - bytes 2..3:   scales_h as uint16_t (2 bits per sub-block scale)
///   - bytes 4..7:   scales_l[4] (4 bits per sub-block scale)
///   - bytes 8..135: qs[128] (4-bit values, 2 per byte)
///
/// Each block has 8 sub-blocks of 32 values.
/// Sub-block scale: ls = (scales_l[ib/2] >> 4*(ib%2) & 0xf) | ((scales_h >> 2*ib) & 3) << 4
/// Dequantized value: dl = d * (ls - 32), value = dl * kvalues_iq4nl[q]
pub fn dequantize_iq4_xs(data: &[u8], out: &mut [f32]) {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 136;
    let num_blocks = out.len() / BLOCK_SIZE;

    for i in 0..num_blocks {
        let start = i * BLOCK_BYTES;
        let block = &data[start..start + BLOCK_BYTES];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];

        let mut y = &mut out[i * BLOCK_SIZE..];
        let mut qsp = 0usize;

        for ib in 0..8 {
            let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as u16
                | (((scales_h >> (2 * ib)) & 3) << 4);
            let dl = d * (ls as f32 - 32.0);

            for j in 0..16 {
                let q_lo = qs[qsp + j] & 0xf;
                let q_hi = qs[qsp + j] >> 4;
                y[j]      = dl * KVALUES_IQ4NL[q_lo as usize] as f32;
                y[j + 16] = dl * KVALUES_IQ4NL[q_hi as usize] as f32;
            }
            y = &mut y[32..];
            qsp += 16;
        }
    }
}

#[cfg(test)]

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_roundtrip() {
        // Q4_0 block: 2-byte f16 scale + 16 bytes of nibble-packed quants,
        // byte-interleaved (two consecutive elements come from one byte).
        let mut data = vec![0u8; 18];
        data[0] = 0x00; data[1] = 0x3C; // scale = 1.0 in f16
        data[2] = 0x89; // q0=9 (low nibble), q1=8 (high nibble)
        let mut out = vec![0.0f32; 32];
        dequantize_q4_0(&data, &mut out);
        // out[0] = (9 - 8) * 1.0 = 1.0, out[1] = (8 - 8) * 1.0 = 0.0
        assert!((out[0] - 1.0).abs() < 0.01);
        assert!((out[1] - 0.0).abs() < 0.01);
        // Un-touched bytes are all-zero quants: (0 - 8) * 1.0 = -8.0
        assert!((out[16] + 8.0).abs() < 0.01);
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
    fn test_q8_k_block_size() {
        let data = vec![0u8; 292];
        let mut out = vec![0.0f32; 256];
        dequantize_q8_k(&data, &mut out);
        // With zero data, d=0, so all outputs should be 0
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_iq4_nl_basic() {
        let mut data = vec![0u8; 18];
        // scale = 1.0 in f16
        data[0] = 0x00; data[1] = 0x3C;
        // nibble0=0 → table[0] = -127, nibble1=15 → table[15] = 113
        data[2] = 0xF0;
        let mut out = vec![0.0f32; 32];
        dequantize_iq4_nl(&data, &mut out);
        assert!((out[0] - -127.0).abs() < 0.001, "out[0] = {}", out[0]);
        assert!((out[1] - 113.0).abs() < 0.001, "out[1] = {}", out[1]);
    }
}
