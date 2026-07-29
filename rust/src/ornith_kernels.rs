//! Math kernels for native Rust inference on safetensors models.
//!
//! All kernels operate on `&mut [f32]` for cache-friendly access.
//! Pure Rust — no SIMD intrinsics yet (Colibri has SIMD but we focus on
//! correctness first, performance later).
//!
//! Kernel list:
//! - rmsnorm:  y[i] = (x[i] / rms) * w[i] * (1 + scale_offset)
//! - matmul:   y[s,o] = sum_i x[s,i] * W[o,i]
//! - matmul_addto:  y[s,o] += sum_i x[s,i] * W[o,i]
//! - swiglu:   y[i] = silu(gate[i]) * up[i]
//! - silu:     x[i] = x[i] / (1 + exp(-x[i]))
//! - softmax:  x[i] = exp(x[i] - max) / sum

/// RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * w
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), w.len());
    let n = x.len();
    let mut sumsq = 0.0f32;
    for &v in x {
        sumsq += v * v;
    }
    let inv_rms = 1.0 / ((sumsq / n as f32) + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv_rms * w[i];
    }
}

/// y[s, o] = sum_i x[s, i] * W[o, i]
/// W is row-major [O, I].
pub fn matmul(y: &mut [f32], x: &[f32], w: &[f32], s: usize, i: usize, o: usize) {
    debug_assert_eq!(y.len(), s * o);
    debug_assert_eq!(x.len(), s * i);
    debug_assert_eq!(w.len(), o * i);
    for si in 0..s {
        for oi in 0..o {
            let mut acc = 0.0f32;
            for ii in 0..i {
                acc += x[si * i + ii] * w[oi * i + ii];
            }
            y[si * o + oi] = acc;
        }
    }
}

/// y[s, o] += sum_i x[s, i] * W[o, i]   (accumulate into existing output)
pub fn matmul_addto(y: &mut [f32], x: &[f32], w: &[f32], s: usize, i: usize, o: usize) {
    debug_assert_eq!(y.len(), s * o);
    debug_assert_eq!(x.len(), s * i);
    debug_assert_eq!(w.len(), o * i);
    for si in 0..s {
        for oi in 0..o {
            let mut acc = 0.0f32;
            for ii in 0..i {
                acc += x[si * i + ii] * w[oi * i + ii];
            }
            y[si * o + oi] += acc;
        }
    }
}

/// SiLU: x[i] = x[i] * sigmoid(x[i])  (in-place)
pub fn silu(x: &mut [f32]) {
    for v in x {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// SwiGLU: out[i] = silu(gate[i]) * up[i]
pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    debug_assert_eq!(out.len(), gate.len());
    debug_assert_eq!(out.len(), up.len());
    for i in 0..out.len() {
        let g = gate[i] / (1.0 + (-gate[i]).exp()); // silu
        out[i] = g * up[i];
    }
}

/// Softmax over a row: x[i] = exp(x[i] - max) / sum
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut maxv = x[0];
    for &v in &x[1..] {
        if v > maxv {
            maxv = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - maxv).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// RoPE rotation: applies rotary embeddings to a slice of pairs (x[2i], x[2i+1])
/// cos/sin arrays are precomputed for the given position.
pub fn rope_inplace(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    debug_assert_eq!(x.len(), cos.len() * 2);
    let half = x.len() / 2;
    for i in 0..half {
        let c = cos[i];
        let s = sin[i];
        let x0 = x[i];
        let x1 = x[half + i];
        x[i] = x0 * c - x1 * s;
        x[half + i] = x1 * c + x0 * s;
    }
}

/// Compute cos/sin tables for rotary embeddings.
/// dim_pairs: number of (x, y) pairs; usually head_dim / 2.
/// Returns (cos[dim_pairs], sin[dim_pairs]).
pub fn rope_tables(theta: f32, dim_pairs: usize, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos = Vec::with_capacity(dim_pairs);
    let mut sin = Vec::with_capacity(dim_pairs);
    for i in 0..dim_pairs {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / (2.0 * dim_pairs as f32));
        let angle = pos as f32 * freq;
        cos.push(angle.cos());
        sin.push(angle.sin());
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rmsnorm(&mut out, &x, &w, 1e-6);
        let sumsq: f32 = 1.0 + 4.0 + 9.0 + 16.0;
        let rms = (sumsq).sqrt() / 2.0;
        let expected: Vec<f32> = x.iter().map(|&v| v / rms).collect();
        for i in 0..4 {
            assert!((out[i] - expected[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu() {
        let mut x = vec![0.0f32, 1.0, 2.0, -1.0];
        silu(&mut x);
        // silu(0) = 0
        // silu(1) = 1 * sigmoid(1) ≈ 0.7311
        // silu(2) = 2 * sigmoid(2) ≈ 1.7616
        // silu(-1) = -1 * sigmoid(-1) ≈ -0.2689
        assert!((x[0] - 0.0).abs() < 1e-5);
        assert!((x[1] - 0.7311).abs() < 1e-3);
        assert!((x[2] - 1.7616).abs() < 1e-3);
        assert!((x[3] - -0.2689).abs() < 1e-3);
    }

    #[test]
    fn test_matmul_simple() {
        // x = [[1, 2], [3, 4]]   (s=2, i=2)
        // W = [[1, 0], [0, 1], [1, 1]]   (o=3, i=2)
        // y = [[1, 2, 3], [3, 4, 7]]
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0; 6];
        matmul(&mut y, &x, &w, 2, 2, 3);
        let expected = vec![1.0, 2.0, 3.0, 3.0, 4.0, 7.0];
        for i in 0..6 {
            assert!((y[i] - expected[i]).abs() < 1e-5, "i={i} got {} expected {}", y[i], expected[i]);
        }
    }

    #[test]
    fn test_swiglu() {
        let gate = vec![0.0f32, 1.0, 2.0];
        let up = vec![1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; 3];
        swiglu(&mut out, &gate, &up);
        // silu(0) * 1 = 0
        // silu(1) * 2 ≈ 1.462
        // silu(2) * 3 ≈ 5.285
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!((out[1] - 1.4621).abs() < 1e-3);
        assert!((out[2] - 5.2848).abs() < 1e-3);
    }
}
