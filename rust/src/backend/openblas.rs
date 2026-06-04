//! OpenBLAS backend — delegates `matmul` to highly-optimized `cblas_sgemm`.
//!
//! All non-matmul ops (vec_add, rms_norm, silu, softmax, etc.) are delegated
//! to `CpuBackend` so we don't need to reimplement them.
//!
//! Enabled via the `openblas` Cargo feature.  When enabled, the global
//! default backend automatically prefers OpenBLAS if the library is
//! available at link time.

use super::Backend;
use super::cpu::CPU_BACKEND;

#[link(name = "openblas")]
extern "C" {
    pub fn cblas_sgemm(
        order: libc::c_int,
        trans_a: libc::c_int,
        trans_b: libc::c_int,
        m: libc::c_int,
        n: libc::c_int,
        k: libc::c_int,
        alpha: libc::c_float,
        a: *const libc::c_float,
        lda: libc::c_int,
        b: *const libc::c_float,
        ldb: libc::c_int,
        beta: libc::c_float,
        c: *mut libc::c_float,
        ldc: libc::c_int,
    );
}

pub const CBLAS_ROW_MAJOR: libc::c_int = 101;
pub const CBLAS_NO_TRANS: libc::c_int = 111;
pub const CBLAS_TRANS: libc::c_int = 112;

/// Global singleton OpenBLAS backend.
pub static OPENBLAS_BACKEND: OpenBlasBackend = OpenBlasBackend;

pub struct OpenBlasBackend;

impl Backend for OpenBlasBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; m * n];
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as libc::c_int,
                n as libc::c_int,
                k as libc::c_int,
                1.0,
                a.as_ptr(),
                k as libc::c_int,   // lda = k (row-major stride)
                b.as_ptr(),
                n as libc::c_int,   // ldb = n (row-major stride)
                0.0,
                result.as_mut_ptr(),
                n as libc::c_int,   // ldc = n (row-major stride)
            );
        }
        result
    }

    // Delegate every other operation to the CPU backend.
    // OpenBLAS does not provide speedups for element-wise ops or activations.

    fn vec_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        CPU_BACKEND.vec_add(a, b)
    }

    fn vec_mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        CPU_BACKEND.vec_mul(a, b)
    }

    fn vec_scale(&self, a: &[f32], scale: f32) -> Vec<f32> {
        CPU_BACKEND.vec_scale(a, scale)
    }

    fn vec_scale_mul(&self, a: &[f32], scale: f32, b: &[f32]) -> Vec<f32> {
        CPU_BACKEND.vec_scale_mul(a, scale, b)
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
        CPU_BACKEND.rms_norm(x, weight, eps, hidden_size)
    }

    fn rms_norm_with_offset(&self, x: &[f32], weight: &[f32], eps: f32, hidden_size: usize, weight_offset: f32) -> Vec<f32> {
        CPU_BACKEND.rms_norm_with_offset(x, weight, eps, hidden_size, weight_offset)
    }

    fn silu(&self, x: &[f32]) -> Vec<f32> {
        CPU_BACKEND.silu(x)
    }

    fn softmax(&self, x: &[f32], hidden_size: usize) -> Vec<f32> {
        CPU_BACKEND.softmax(x, hidden_size)
    }

    fn sum_sq(&self, x: &[f32]) -> f32 {
        CPU_BACKEND.sum_sq(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify OpenBLAS matmul produces the same result as the CPU backend.
    #[test]
    fn test_openblas_matmul_correctness() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]; // [3, 2]
        // Expected: [[58, 64], [139, 154]]
        let expected = vec![58.0f32, 64.0, 139.0, 154.0];

        let openblas_result = OPENBLAS_BACKEND.matmul(&a, &b, 2, 3, 2);
        let cpu_result = CPU_BACKEND.matmul(&a, &b, 2, 3, 2);

        assert_eq!(openblas_result, expected, "OpenBLAS result mismatch");
        assert_eq!(cpu_result, expected, "CPU result mismatch");
        assert_eq!(openblas_result, cpu_result, "OpenBLAS vs CPU mismatch");
    }

    /// Larger random matrix to stress the BLAS path.
    #[test]
    fn test_openblas_matmul_large() {
        let m = 64;
        let k = 128;
        let n = 64;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i % 5) as f32 * 0.1).collect();

        let openblas_result = OPENBLAS_BACKEND.matmul(&a, &b, m, k, n);
        let cpu_result = CPU_BACKEND.matmul(&a, &b, m, k, n);

        assert_eq!(openblas_result.len(), cpu_result.len());
        for (i, (ob, cpu)) in openblas_result.iter().zip(cpu_result.iter()).enumerate() {
            let diff = (ob - cpu).abs();
            assert!(
                diff < 1e-4,
                "Mismatch at index {}: OpenBLAS={}, CPU={}, diff={}",
                i, ob, cpu, diff
            );
        }
    }
}
