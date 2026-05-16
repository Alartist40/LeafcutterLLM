//! CPU backend with architecture-specific SIMD kernels

use super::Backend;
use crate::kernels::simd;

/// Global singleton CPU backend.
/// Uses the best available instruction set for the target architecture.
pub static CPU_BACKEND: CpuBackend = CpuBackend;

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; m * n];
        // Use parallel matmul for large matrices to utilize all CPU cores.
        // Threshold: 4096 output elements (~64×64) is where threading pays off.
        if m * n >= 4096 {
            simd::simd_matmul_parallel(a, b, &mut result, m, k, n);
        } else {
            simd::simd_matmul(a, b, &mut result, m, k, n);
        }
        result
    }

    fn vec_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; a.len()];
        simd::simd_vec_add(a, b, &mut out);
        out
    }

    fn vec_mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; a.len()];
        simd::simd_vec_mul(a, b, &mut out);
        out
    }

    fn vec_scale(&self, a: &[f32], scale: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; a.len()];
        simd::simd_vec_scale(a, scale, &mut out);
        out
    }

    fn vec_scale_mul(&self, a: &[f32], scale: f32, b: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; a.len()];
        simd::simd_vec_scale_mul(a, scale, b, &mut out);
        out
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
        let rows = x.len() / hidden_size;
        let mut result = vec![0.0f32; x.len()];
        for r in 0..rows {
            let start = r * hidden_size;
            let slice = &x[start..start + hidden_size];
            let mean_sq = simd::simd_sum_sq(slice) / hidden_size as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            simd::simd_vec_scale_mul(slice, scale, &weight[..hidden_size], &mut result[start..start + hidden_size]);
        }
        result
    }

    fn silu(&self, x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v * (1.0 / (1.0 + (-v).exp()))).collect()
    }

    fn softmax(&self, x: &[f32], hidden_size: usize) -> Vec<f32> {
        let rows = x.len() / hidden_size;
        let mut result = vec![0.0f32; x.len()];
        for r in 0..rows {
            let start = r * hidden_size;
            let end = start + hidden_size;
            let max_val = x[start..end].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f32 = x[start..end].iter().map(|&v| (v - max_val).exp()).sum();
            for i in 0..hidden_size {
                result[start + i] = (x[start + i] - max_val).exp() / sum_exp;
            }
        }
        result
    }

    fn sum_sq(&self, x: &[f32]) -> f32 {
        simd::simd_sum_sq(x)
    }
}
