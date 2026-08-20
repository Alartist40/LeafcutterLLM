//! Q8_K-activation GEMV kernels (m == 1).
//!
//! The activation vector `a` is quantized to Q8_K once per matmul
//! (`crate::kernels::q8_k::quantize_row_q8_k`), then each output column's
//! dot product is computed in the integer domain, ported from llama.cpp's
//! `ggml_vec_dot_q4_K_q8_K` / `ggml_vec_dot_q6_K_q8_K` (AVX2 paths).
//!
//! This trades 256 f32 MACs + dequant per column for 16 integer MACs per
//! `_mm256_maddubs_epi16` and is the hot GEMV path for FFN gate/up and
//! lm_head.

use super::q4_k::Block as Q4KBlock;
use super::q4_k::Matrix as Q4KMatrix;
use super::q6_k::Block as Q6KBlock;
use super::q6_k::Matrix as Q6KMatrix;
use super::q8_k::Q8KBlock;
use super::q8_k::{quantize_row_q8_k, q4_k_dot_q8_k_scalar, q6_k_dot_q8_k_scalar, QK_K};

/// Shuffle index table row for Q4_K: 32 bytes alternating [2*sc, 2*sc+1, ...].
/// Applied to a 16-lane i16 vector of scales, this broadcasts scale `sc`
/// into every i16 lane (llama.cpp `get_scale_shuffle_k4`).
fn get_scale_shuffle_k4(sc: usize) -> [u8; 32] {
    let mut v = [0u8; 32];
    for (i, byte) in v.iter_mut().enumerate() {
        *byte = (2 * sc + (i & 1)) as u8;
    }
    v
}

/// Shuffle index table row for Q6_K: 16 bytes, 8 copies of [2*is, 2*is+1].
/// Broadcasts scales[2*is] into lanes 0..7 and scales[2*is+1] into lanes
/// 8..15 (llama.cpp `get_scale_shuffle`).
fn get_scale_shuffle_k6(is: usize) -> [u8; 16] {
    let mut v = [0u8; 16];
    for (i, byte) in v.iter_mut().enumerate() {
        *byte = (2 * is + (i >> 3)) as u8;
    }
    v
}

/// AVX2 per-column Q4_K x Q8_K dot (llama.cpp `ggml_vec_dot_q4_K_q8_K`).
///
/// Accumulates `d * sumi - dmin_neg * sum(bsums * mins)` per block, with the
/// 8 scale bytes and 8 min bytes unpacked from the 12-byte scales field.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q4_k_gemv_col_q8_avx2(w_blocks: &[Q4KBlock], a_blocks: &[Q8KBlock]) -> f32 {
    use std::arch::x86_64::*;
    let nb = w_blocks.len();
    debug_assert_eq!(a_blocks.len(), nb);

    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_ps();
    let mut acc_m = _mm_setzero_ps();

    for i in 0..nb {
        let w = &w_blocks[i];
        let y = &a_blocks[i];

        let d = y.d * w.d;
        let dmin = -y.d * w.dmin;

        // Unpack the 12 scale bytes into 4 uint32 (llama.cpp utmp dance).
        let s = &w.scales;
        let u0 = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        let u1 = u32::from_le_bytes([s[4], s[5], s[6], s[7]]);
        let u2 = u32::from_le_bytes([s[8], s[9], s[10], s[11]]);
        let u3 = ((u2 >> 4) & 0x0f0f_0f0f) | (((u1 >> 6) & 0x0303_0303) << 4);
        let uaux = u1 & 0x3f3f_3f3f;
        let u1 = (u2 & 0x0f0f_0f0f) | (((u0 >> 6) & 0x0303_0303) << 4);
        let u2 = uaux;
        let u0 = u0 & 0x3f3f_3f3f;

        // Bytes [sc0..sc7, m0..m7] zero-extended to 16 i16 lanes.
        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(u3 as i32, u2 as i32, u1 as i32, u0 as i32));

        // min correction: sum over 32-value chunks of mins[chunk] * bsums_sum.
        let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1),
        );
        let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        // Low 128 bits = 8 scales; broadcast to both halves.
        let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        let scales = _mm256_inserti128_si256(_mm256_castsi128_si256(sc128), sc128, 1);

        let mut sumi = _mm256_setzero_si256();
        let mut q4 = w.qs.as_ptr();
        let mut q8 = y.qs.as_ptr();

        for j in 0..(QK_K / 64) {
            let shuf_l = _mm256_loadu_si256(get_scale_shuffle_k4(2 * j).as_ptr() as *const __m256i);
            let shuf_h = _mm256_loadu_si256(get_scale_shuffle_k4(2 * j + 1).as_ptr() as *const __m256i);
            let scale_l = _mm256_shuffle_epi8(scales, shuf_l);
            let scale_h = _mm256_shuffle_epi8(scales, shuf_h);

            let q4bits = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            let q8l = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let p16l = _mm256_maddubs_epi16(q4l, q8l);
            let p16l = _mm256_madd_epi16(scale_l, p16l);

            let q8h = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let p16h = _mm256_maddubs_epi16(q4h, q8h);
            let p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7] + _mm_cvtss_f32(acc_m)
}

/// AVX2 per-column Q6_K x Q8_K dot (llama.cpp `ggml_vec_dot_q6_K_q8_K`).
///
/// The -32 offset of each 6-bit value is folded into a fixed correction
/// `32 * sum(scales * bsums)` (q8sclsub) subtracted after the maddubs chain.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q6_k_gemv_col_q8_avx2(w_blocks: &[Q6KBlock], a_blocks: &[Q8KBlock]) -> f32 {
    use std::arch::x86_64::*;
    let nb = w_blocks.len();
    debug_assert_eq!(a_blocks.len(), nb);

    let m3 = _mm256_set1_epi8(3);
    let m15 = _mm256_set1_epi8(15);
    let mut acc = _mm256_setzero_ps();

    for i in 0..nb {
        let w = &w_blocks[i];
        let y = &a_blocks[i];
        let d = y.d * w.d;

        let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
        let scales = _mm_loadu_si128(w.scales.as_ptr() as *const __m128i);
        let scales_16 = _mm256_cvtepi8_epi16(scales);
        let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        let mut sumi = _mm256_setzero_si256();
        let mut q4 = w.ql.as_ptr();
        let mut qh = w.qh.as_ptr();
        let mut q8 = y.qs.as_ptr();
        let mut is = 0usize;

        for _j in 0..(QK_K / 128) {
            let q4bits1 = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4bits2 = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
            qh = qh.add(32);

            let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4);
            let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(12)), 2);
            let q4h_2 = _mm256_and_si256(q4bits_h, _mm256_set1_epi8(48));
            let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(-64)), 2);

            let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            let q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            let q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);

            let p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            let p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            let p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            let p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            let s0 = _mm_shuffle_epi8(scales, _mm_loadu_si128(get_scale_shuffle_k6(is).as_ptr() as *const __m128i));
            let s1 = _mm_shuffle_epi8(scales, _mm_loadu_si128(get_scale_shuffle_k6(is + 1).as_ptr() as *const __m128i));
            let s2 = _mm_shuffle_epi8(scales, _mm_loadu_si128(get_scale_shuffle_k6(is + 2).as_ptr() as *const __m128i));
            let s3 = _mm_shuffle_epi8(scales, _mm_loadu_si128(get_scale_shuffle_k6(is + 3).as_ptr() as *const __m128i));
            is += 4;

            let p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s0), p16_0);
            let p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s1), p16_1);
            let p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s2), p16_2);
            let p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(s3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    let mut buf = [0.0f32; 8];
    _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]
}

/// Quantize `a` once, then compute every output column as an integer dot.
fn run_q4_k_q8_gemv(a: &[f32], b: &Q4KMatrix, c: &mut [f32], k: usize, n: usize) {
    let bpr = b.blocks_per_row();
    let mut a_blocks = vec![Q8KBlock::default(); k / QK_K];
    quantize_row_q8_k(&a[..k], &mut a_blocks);

    let compute = |j: usize| -> f32 {
        let row_base = j * bpr;
        let w = &b.blocks[row_base..row_base + bpr];
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                return unsafe { crate::kernels::q8_k::q4_k_dot_q8_k_sdot(w, &a_blocks) };
            }
            return unsafe { crate::kernels::q8_k::q4_k_dot_q8_k_neon(w, &a_blocks) };
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                return unsafe { q4_k_gemv_col_q8_avx2(w, &a_blocks) };
            }
        }
        q4_k_dot_q8_k_scalar(w, &a_blocks)
    };

    // Chunked parallelization: split the output into a few contiguous ranges
    // (one per worker, 2-way oversubscribed) instead of one Rayon task per
    // column. Per-column tasks are tiny (~k/256 block dots) and their dispatch
    // overhead dominates large-n GEMVs (e.g. lm_head, n = 248K).
    if n >= 1024 {
        use rayon::prelude::*;
        let nthreads = rayon::current_num_threads().max(1);
        let chunk = (n / (nthreads * 2)).max(1);
        c.par_chunks_mut(chunk).enumerate().for_each(|(ci, out)| {
            let start = ci * chunk;
            for (jj, o) in out.iter_mut().enumerate() {
                *o = compute(start + jj);
            }
        });
        return;
    }
    for j in 0..n {
        c[j] = compute(j);
    }
}

/// Quantize `a` once, then compute every output column as an integer dot.
fn run_q6_k_q8_gemv(a: &[f32], b: &Q6KMatrix, c: &mut [f32], k: usize, n: usize) {
    let bpr = b.blocks_per_row();
    let mut a_blocks = vec![Q8KBlock::default(); k / QK_K];
    quantize_row_q8_k(&a[..k], &mut a_blocks);

    let compute = |j: usize| -> f32 {
        let row_base = j * bpr;
        let w = &b.blocks[row_base..row_base + bpr];
                #[cfg(target_arch = "aarch64")]
        {
            return unsafe { crate::kernels::q8_k::q6_k_dot_q8_k_neon(w, &a_blocks) };
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                return unsafe { q6_k_gemv_col_q8_avx2(w, &a_blocks) };
            }
        }
        q6_k_dot_q8_k_scalar(w, &a_blocks)
    };

    // Chunked parallelization (see `run_q4_k_q8_gemv`).
    if n >= 1024 {
        use rayon::prelude::*;
        let nthreads = rayon::current_num_threads().max(1);
        let chunk = (n / (nthreads * 2)).max(1);
        c.par_chunks_mut(chunk).enumerate().for_each(|(ci, out)| {
            let start = ci * chunk;
            for (jj, o) in out.iter_mut().enumerate() {
                *o = compute(start + jj);
            }
        });
    } else {
        for j in 0..n {
            c[j] = compute(j);
        }
    }
}

/// Q8_K-activation GEMV for Q4_K (m == 1 only).
pub fn q4_k_matmul_transposed_b_q8(a: &[f32], b: &Q4KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(b.cols, k);
    assert_eq!(b.rows, n);
    for v in c.iter_mut() { *v = 0.0; }
    if m == 1 {
        run_q4_k_q8_gemv(a, b, c, k, n);
    } else {
        run_q4_k_q8_gemm(a, b, c, m, k, n);
    }
}

/// Batched Q8_K-activation GEMM for Q4_K (m > 1, prefill).
///
/// Quantizes every activation row to Q8_K once, then for each output column
/// reconstructs the weight `aux8` nibbles once per block and reuses them
/// across all `m` rows via `sdot`.
fn run_q4_k_q8_gemm(a: &[f32], b: &Q4KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    let bpr = b.blocks_per_row();
    let abpr = k / QK_K;
    let mut a_q = vec![Q8KBlock::default(); m * abpr];
    for i in 0..m {
        quantize_row_q8_k(&a[i * k..(i + 1) * k], &mut a_q[i * abpr..(i + 1) * abpr]);
    }

    #[cfg(target_arch = "aarch64")]
    {
        use crate::kernels::q4_k::QK_K as Q4QK;
        use crate::kernels::q8_k::build_q4_aux8;
        use crate::kernels::q8_k::q4_block_dot_sdot;
        use rayon::prelude::*;
        let nthreads = rayon::current_num_threads().max(1);
        let chunkp = (n / (nthreads * 2)).max(1);
        let mut col_results = vec![0.0f32; n * m];
        col_results.par_chunks_mut(m).enumerate().for_each(|(j, col)| {
            let row_base = j * bpr;
            let mut aux = [0u8; Q4QK];
            unsafe {
                for blk in 0..bpr {
                    let block = &b.blocks[row_base + blk];
                    build_q4_aux8(block, &mut aux);
                    for i in 0..m {
                        col[i] += q4_block_dot_sdot(block, &aux, &a_q[i * abpr + blk]);
                    }
                }
            }
        });
        for j in 0..n {
            for i in 0..m {
                c[i * n + j] = col_results[j * m + i];
            }
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        use rayon::prelude::*;
        let nthreads = rayon::current_num_threads().max(1);
        let chunkp = (n / (nthreads * 2)).max(1);
        let mut col_results = vec![0.0f32; n * m];
        col_results.par_chunks_mut(m).enumerate().for_each(|(j, col)| {
            let row_base = j * bpr;
            let w = &b.blocks[row_base..row_base + bpr];
            for i in 0..m {
                let arow = &a_q[i * abpr..(i + 1) * abpr];
                col[i] += q4_k_dot_q8_k_scalar(w, arow);
            }
        });
        for j in 0..n {
            for i in 0..m {
                c[i * n + j] = col_results[j * m + i];
            }
        }
    }
}

/// Q8_K-activation GEMV for Q6_K (m == 1 only).
pub fn q6_k_matmul_transposed_b_q8(a: &[f32], b: &Q6KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(b.cols, k);
    assert_eq!(b.rows, n);
    for v in c.iter_mut() { *v = 0.0; }
    if m == 1 {
        run_q6_k_q8_gemv(a, b, c, k, n);
    } else {
        run_q6_k_q8_gemm(a, b, c, m, k, n);
    }
}

/// Batched Q8_K-activation GEMM for Q6_K (m > 1, prefill).
fn run_q6_k_q8_gemm(a: &[f32], b: &Q6KMatrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    let bpr = b.blocks_per_row();
    let abpr = k / QK_K;
    let mut a_q = vec![Q8KBlock::default(); m * abpr];
    for i in 0..m {
        quantize_row_q8_k(&a[i * k..(i + 1) * k], &mut a_q[i * abpr..(i + 1) * abpr]);
    }

    #[cfg(target_arch = "aarch64")]
    {
        use crate::kernels::q6_k::QK_K as Q6QK;
        use crate::kernels::q8_k::build_q6_aux8;
        use crate::kernels::q8_k::q6_block_dot_sdot;
        use rayon::prelude::*;
        let nthreads = rayon::current_num_threads().max(1);
        let chunkp = (n / (nthreads * 2)).max(1);
        let mut col_results = vec![0.0f32; n * m];
        col_results.par_chunks_mut(m).enumerate().for_each(|(j, col)| {
            let row_base = j * bpr;
            let mut aux = [0i8; Q6QK];
            unsafe {
                for blk in 0..bpr {
                    let block = &b.blocks[row_base + blk];
                    build_q6_aux8(block, &mut aux);
                    for i in 0..m {
                        col[i] += q6_block_dot_sdot(block, &aux, &a_q[i * abpr + blk]);
                    }
                }
            }
        });
        for j in 0..n {
            for i in 0..m {
                c[i * n + j] = col_results[j * m + i];
            }
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        use rayon::prelude::*;
        let nthreads = rayon::current_num_threads().max(1);
        let chunkp = (n / (nthreads * 2)).max(1);
        let mut col_results = vec![0.0f32; n * m];
        col_results.par_chunks_mut(m).enumerate().for_each(|(j, col)| {
            let row_base = j * bpr;
            let w = &b.blocks[row_base..row_base + bpr];
            for i in 0..m {
                let arow = &a_q[i * abpr..(i + 1) * abpr];
                col[i] += q6_k_dot_q8_k_scalar(w, arow);
            }
        });
        for j in 0..n {
            for i in 0..m {
                c[i * n + j] = col_results[j * m + i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q4_matrix(rows: usize, cols: usize) -> Q4KMatrix {
        let bpr = cols / QK_K;
        let mut blocks = Vec::with_capacity(rows * bpr);
        for row in 0..rows {
            for b in 0..bpr {
                let d = 0.03f32 * ((row as f32) + 1.0) / (rows as f32);
                let dmin = 0.01f32 * ((row as f32) + 1.0) / (rows as f32);
                let mut scales = [0u8; 12];
                for s in 0..12 {
                    scales[s] = (((s * 3 + row) as i32) % 64) as u8;
                }
                let mut qs = [0u8; 128];
                for s in 0..128 {
                    qs[s] = ((s * 5 + row * 7 + b * 11) % 256) as u8;
                }
                blocks.push(Q4KBlock { d, dmin, scales, qs });
            }
        }
        Q4KMatrix { rows, cols, blocks }
    }

    fn make_q6_matrix(rows: usize, cols: usize) -> Q6KMatrix {
        let bpr = cols / QK_K;
        let mut blocks = Vec::with_capacity(rows * bpr);
        for row in 0..rows {
            for b in 0..bpr {
                let d = 0.02f32 * ((row as f32) + 1.0) / (rows as f32);
                let mut ql = [0u8; 128];
                let mut qh = [0u8; 64];
                for s in 0..128 {
                    ql[s] = ((s * 3 + row + b * 13) % 256) as u8;
                }
                for s in 0..64 {
                    qh[s] = ((s * 7 + row * 5) % 256) as u8;
                }
                let mut scales = [0u8; 16];
                for s in 0..16 {
                    scales[s] = (((s * 5 + row) as i32) % 61 - 30) as u8;
                }
                blocks.push(Q6KBlock { d, ql, qh, scales });
            }
        }
        Q6KMatrix { rows, cols, blocks }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_q4_k_gemv_col_q8_avx2_matches_scalar() {
        let k = 1024;
        let n = 512;
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.37).cos() * 1.5).collect();
        let b = make_q4_matrix(n, k);

        let mut a_blocks = vec![Q8KBlock::default(); k / QK_K];
        quantize_row_q8_k(&a, &mut a_blocks);

        for j in (0..n).step_by(64) {
            let row_base = j * (k / QK_K);
            let w = &b.blocks[row_base..row_base + (k / QK_K)];
            let expected = q4_k_dot_q8_k_scalar(w, &a_blocks);
            let got = unsafe { q4_k_gemv_col_q8_avx2(w, &a_blocks) };
            let rel = (got - expected).abs() / expected.abs().max(1e-6);
            assert!(rel < 1e-3,
                "q4 col {} avx2 vs scalar: got={}, expected={}, rel={:.6}", j, got, expected, rel);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_q6_k_gemv_col_q8_avx2_matches_scalar() {
        let k = 1024;
        let n = 512;
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.29).sin() * 1.5).collect();
        let b = make_q6_matrix(n, k);

        let mut a_blocks = vec![Q8KBlock::default(); k / QK_K];
        quantize_row_q8_k(&a, &mut a_blocks);

        for j in (0..n).step_by(64) {
            let row_base = j * (k / QK_K);
            let w = &b.blocks[row_base..row_base + (k / QK_K)];
            let expected = q6_k_dot_q8_k_scalar(w, &a_blocks);
            let got = unsafe { q6_k_gemv_col_q8_avx2(w, &a_blocks) };
            let rel = (got - expected).abs() / expected.abs().max(1e-6);
            assert!(rel < 1e-3,
                "q6 col {} avx2 vs scalar: got={}, expected={}, rel={:.6}", j, got, expected, rel);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_q4_k_q8_gemm_m4_matches_dequant() {
        let k = 1024;
        let m = 4;
        let n = 512;
        let a: Vec<f32> = (0..m * k).map(|idx| ((idx as f32) * 0.13).sin() * 2.0).collect();
        let b = make_q4_matrix(n, k);

        // Q8_K batched path (sdot on aarch64)
        let mut c_q8 = vec![0.0f32; m * n];
        crate::kernels::q8_k_gemm::q4_k_matmul_transposed_b_q8(&a, &b, &mut c_q8, m, k, n);

        // Reference: per-row m=1 Q8_K gemv (same activation quantization).
        for i in 0..m {
            let mut c_row = vec![0.0f32; n];
            crate::kernels::q8_k_gemm::q4_k_matmul_transposed_b_q8(
                &a[i * k..(i + 1) * k], &b, &mut c_row, 1, k, n,
            );
            for j in 0..n {
                let diff = (c_q8[i * n + j] - c_row[j]).abs();
                assert!(diff < 1e-4,
                    "q4 batched ({i},{j}): batched={}, row={}, diff={:.6}", c_q8[i * n + j], c_row[j], diff);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_q6_k_q8_gemm_m4_matches_dequant() {
        let k = 1024;
        let m = 4;
        let n = 512;
        let a: Vec<f32> = (0..m * k).map(|idx| ((idx as f32) * 0.17).cos() * 1.5).collect();
        let b = make_q6_matrix(n, k);

        let mut c_q8 = vec![0.0f32; m * n];
        crate::kernels::q8_k_gemm::q6_k_matmul_transposed_b_q8(&a, &b, &mut c_q8, m, k, n);

        // Reference: per-row m=1 Q8_K gemv (same activation quantization).
        for i in 0..m {
            let mut c_row = vec![0.0f32; n];
            crate::kernels::q8_k_gemm::q6_k_matmul_transposed_b_q8(
                &a[i * k..(i + 1) * k], &b, &mut c_row, 1, k, n,
            );
            for j in 0..n {
                let diff = (c_q8[i * n + j] - c_row[j]).abs();
                assert!(diff < 1e-4,
                    "q6 batched ({i},{j}): batched={}, row={}, diff={:.6}", c_q8[i * n + j], c_row[j], diff);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_q4_k_dot_q8_neon_2col_matches_scalar() {
        let k = 1024;
        let n = 512;
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.37).cos() * 1.5).collect();
        let b = make_q4_matrix(n, k);

        let mut a_blocks = vec![Q8KBlock::default(); k / QK_K];
        quantize_row_q8_k(&a, &mut a_blocks);

        for j in (0..n).step_by(2) {
            let row_base = j * (k / QK_K);
            let w1 = &b.blocks[row_base..row_base + (k / QK_K)];
            let w2 = &b.blocks[row_base + (k / QK_K)..row_base + 2 * (k / QK_K)];
            let (got1, got2) = unsafe {
                crate::kernels::q8_k::q4_k_dot_q8_k_2col_neon(w1, w2, &a_blocks)
            };
            let expected1 = q4_k_dot_q8_k_scalar(w1, &a_blocks);
            let expected2 = q4_k_dot_q8_k_scalar(w2, &a_blocks);
            let diff1 = (got1 - expected1).abs();
            let diff2 = (got2 - expected2).abs();
            assert!(diff1 < 1e-4,
                "q4 2col col {} got1={}, expected1={}, diff={:.6}", j, got1, expected1, diff1);
            assert!(diff2 < 1e-4,
                "q4 2col col {} got2={}, expected2={}, diff={:.6}", j + 1, got2, expected2, diff2);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_q4_k_dot_q8_sdot_2col_matches_scalar() {
        let k = 1024;
        let n = 512;
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.37).cos() * 1.5).collect();
        let b = make_q4_matrix(n, k);

        let mut a_blocks = vec![Q8KBlock::default(); k / QK_K];
        quantize_row_q8_k(&a, &mut a_blocks);

        for j in (0..n).step_by(2) {
            let row_base = j * (k / QK_K);
            let w1 = &b.blocks[row_base..row_base + (k / QK_K)];
            let w2 = &b.blocks[row_base + (k / QK_K)..row_base + 2 * (k / QK_K)];
            let (got1, got2) = unsafe {
                crate::kernels::q8_k::q4_k_dot_q8_k_2col_sdot(w1, w2, &a_blocks)
            };
            let expected1 = q4_k_dot_q8_k_scalar(w1, &a_blocks);
            let expected2 = q4_k_dot_q8_k_scalar(w2, &a_blocks);
            let diff1 = (got1 - expected1).abs();
            let diff2 = (got2 - expected2).abs();
            assert!(diff1 < 1e-4,
                "q4 2col sdot col {} got1={}, expected1={}, diff={:.6}", j, got1, expected1, diff1);
            assert!(diff2 < 1e-4,
                "q4 2col sdot col {} got2={}, expected2={}, diff={:.6}", j + 1, got2, expected2, diff2);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_q4_k_dot_q8_neon_matches_scalar() {
        let k = 1024;
        let n = 512;
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.37).cos() * 1.5).collect();
        let b = make_q4_matrix(n, k);

        let mut a_blocks = vec![Q8KBlock::default(); k / QK_K];
        quantize_row_q8_k(&a, &mut a_blocks);

        for j in 0..n {
            let row_base = j * (k / QK_K);
            let w = &b.blocks[row_base..row_base + (k / QK_K)];
            let expected = q4_k_dot_q8_k_scalar(w, &a_blocks);
            let got = unsafe { crate::kernels::q8_k::q4_k_dot_q8_k_neon(w, &a_blocks) };
            let diff = (got - expected).abs();
            assert!(diff < 1e-4,
                "q4 col {} neon vs scalar: got={}, expected={}, diff={:.6}", j, got, expected, diff);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_q6_k_dot_q8_neon_matches_scalar() {
        let k = 1024;
        let n = 512;
        let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.29).sin() * 1.5).collect();
        let b = make_q6_matrix(n, k);

        let mut a_blocks = vec![Q8KBlock::default(); k / QK_K];
        quantize_row_q8_k(&a, &mut a_blocks);

        for j in 0..n {
            let row_base = j * (k / QK_K);
            let w = &b.blocks[row_base..row_base + (k / QK_K)];
            let expected = q6_k_dot_q8_k_scalar(w, &a_blocks);
            let got = unsafe { crate::kernels::q8_k::q6_k_dot_q8_k_neon(w, &a_blocks) };
            let diff = (got - expected).abs();
            assert!(diff < 1e-4,
                "q6 col {} neon vs scalar: got={}, expected={}, diff={:.6}", j, got, expected, diff);
        }
    }

    /// Microbenchmark: time a full GEMV at production shapes so the Q4_K vs
    /// Q6_K per-block throughput gap can be measured without running the model.
    /// Run with `cargo test --release -- --ignored bench_gemv --nocapture`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn bench_gemv() {
        use std::time::Instant;

        let run4 = |k: usize, n: usize| {
            let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.13).cos() * 2.0).collect();
            let b = make_q4_matrix(n, k);
            let mut c = vec![0.0f32; n];
            let nblocks = n * (k / QK_K);
            let t0 = Instant::now();
            for _ in 0..8 {
                run_q4_k_q8_gemv(&a, &b, &mut c, k, n);
            }
            let dt = t0.elapsed().as_secs_f64() / 8.0;
            let bytes = nblocks as f64 * 128.0;
            println!(
                "Q4_K GEMV k={} n={} blocks={}M: {:.2} ms  ({:.1} GB/s)",
                k, n, nblocks as f64 / 1e6, dt * 1e3, bytes / dt / 1e9
            );
        };
        let run6 = |k: usize, n: usize| {
            let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.13).cos() * 2.0).collect();
            let b = make_q6_matrix(n, k);
            let mut c = vec![0.0f32; n];
            let nblocks = n * (k / QK_K);
            let t0 = Instant::now();
            for _ in 0..8 {
                run_q6_k_q8_gemv(&a, &b, &mut c, k, n);
            }
            let dt = t0.elapsed().as_secs_f64() / 8.0;
            let bytes = nblocks as f64 * 210.0;
            println!(
                "Q6_K GEMV k={} n={} blocks={}M: {:.2} ms  ({:.1} GB/s)",
                k, n, nblocks as f64 / 1e6, dt * 1e3, bytes / dt / 1e9
            );
        };

        run4(4096, 12288);
        run4(4096, 8192);
        run4(12288, 4096);
        run6(4096, 8192);
        run6(12288, 4096);
        run6(4096, 248320);
    }
}
