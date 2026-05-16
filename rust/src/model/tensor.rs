//! f32 tensor implementation with pluggable compute backend

use crate::backend::{default_backend, set_global_backend, Backend};
use crate::kernels::q4_0::Matrix as Q4Matrix;
use crate::kernels::q8_0::Matrix as Q8Matrix;

/// Native quantized weight data attached to a Tensor.
/// When present, matmul dispatches to a format-specific kernel
/// for memory-bandwidth savings. All other ops use `data` (f32).
#[derive(Clone)]
pub enum QuantizedData {
    Q4_0(Q4Matrix),
    Q8_0(Q8Matrix),
}

#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    /// Optional quantized weights for fast matmul.
    q_data: Option<QuantizedData>,
    backend: &'static dyn Backend,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("data_len", &self.data.len())
            .field("quantized", &self.q_data.is_some())
            .field("backend", &"<dyn Backend>")
            .finish()
    }
}

impl Tensor {
    /// Create a tensor filled with zeros, using the global backend.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; size],
            q_data: None,
            backend: default_backend(),
        }
    }

    /// Create a tensor from raw data, using the global backend.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self {
            shape,
            data,
            q_data: None,
            backend: default_backend(),
        }
    }

    /// Create a tensor with a specific backend.
    pub fn from_vec_with_backend(data: Vec<f32>, shape: Vec<usize>, backend: &'static dyn Backend) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { shape, data, q_data: None, backend }
    }

    /// Create a tensor from Q8_0 quantized weights.
    /// Stores both Q8_0 (for fast INT8 matmul) and f32 (for other ops).
    pub fn from_q8_0(q8: Q8Matrix, shape: Vec<usize>) -> Self {
        assert_eq!(q8.rows * q8.cols, shape.iter().product::<usize>());
        let data = q8.dequantize();
        Self {
            shape,
            data,
            q_data: Some(QuantizedData::Q8_0(q8)),
            backend: default_backend(),
        }
    }

    /// Create a tensor from Q4_0 quantized weights.
    /// Stores both Q4_0 (for fast INT4 matmul) and f32 (for other ops).
    pub fn from_q4_0(q4: Q4Matrix, shape: Vec<usize>) -> Self {
        assert_eq!(q4.rows * q4.cols, shape.iter().product::<usize>());
        let data = q4.dequantize();
        Self {
            shape,
            data,
            q_data: Some(QuantizedData::Q4_0(q4)),
            backend: default_backend(),
        }
    }

    /// Returns true if this tensor has native quantized data for fast matmul.
    pub fn is_quantized(&self) -> bool {
        self.q_data.is_some()
    }

    /// Set the global backend for all new Tensors.
    pub fn set_global_backend(backend: &'static dyn Backend) {
        set_global_backend(backend);
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Matrix multiplication: self @ other
    /// self: [m, k], other: [k, n], result: [m, n]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];
        assert_eq!(k, other.shape[0]);

        // Fast path: if other has quantized weights, use native GEMM
        if let Some(ref q) = other.q_data {
            let mut result = vec![0.0f32; m * n];
            match q {
                QuantizedData::Q8_0(q8) => {
                    assert_eq!(q8.rows, k);
                    assert_eq!(q8.cols, n);
                    crate::kernels::int8_gemm::q8_0_matmul(&self.data, q8, &mut result, m, k, n);
                }
                QuantizedData::Q4_0(q4) => {
                    assert_eq!(q4.rows, k);
                    assert_eq!(q4.cols, n);
                    crate::kernels::int8_gemm::q4_0_matmul(&self.data, q4, &mut result, m, k, n);
                }
            }
            return Self::from_vec_with_backend(result, vec![m, n], self.backend);
        }

        let result = self.backend.matmul(&self.data, &other.data, m, k, n);
        Self::from_vec_with_backend(result, vec![m, n], self.backend)
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.size(), other.size());
        let data = self.backend.vec_add(&self.data, &other.data);
        Self::from_vec_with_backend(data, self.shape.clone(), self.backend)
    }

    /// RMSNorm: x * rsqrt(mean(x^2) + epsilon) * weight
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Tensor {
        let hidden_size = self.shape.last().copied().unwrap_or(1);
        let data = self.backend.rms_norm(&self.data, &weight.data, eps, hidden_size);
        Self::from_vec_with_backend(data, self.shape.clone(), self.backend)
    }

    /// SiLU activation: x * sigmoid(x)
    pub fn silu(&self) -> Tensor {
        let data = self.backend.silu(&self.data);
        Self::from_vec_with_backend(data, self.shape.clone(), self.backend)
    }

    /// Softmax over last dimension
    pub fn softmax_last_dim(&self) -> Tensor {
        let hidden_size = self.shape.last().copied().unwrap_or(1);
        let data = self.backend.softmax(&self.data, hidden_size);
        Self::from_vec_with_backend(data, self.shape.clone(), self.backend)
    }

    /// Reshape to new shape (total size must match)
    pub fn reshape(&self, shape: Vec<usize>) -> Tensor {
        assert_eq!(self.size(), shape.iter().product::<usize>());
        Self {
            shape,
            data: self.data.clone(),
            q_data: None, // reshape invalidates quantized block layout
            backend: self.backend,
        }
    }

    /// Transpose a 2D tensor: [m, n] -> [n, m]
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        let m = self.shape[0];
        let n = self.shape[1];
        let mut result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = self.data[i * n + j];
            }
        }
        // Transpose invalidates quantized block layout
        Self::from_vec_with_backend(result, vec![n, m], self.backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rms_norm() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let w = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]);
        let y = x.rms_norm(&w, 1e-5);
        assert!(y.data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_softmax() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let y = x.softmax_last_dim();
        let sum: f32 = y.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
