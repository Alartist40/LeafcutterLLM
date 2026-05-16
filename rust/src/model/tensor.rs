//! f32 tensor implementation with pluggable compute backend

use crate::backend::{default_backend, set_global_backend, Backend};

#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    backend: &'static dyn Backend,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("data_len", &self.data.len())
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
            backend: default_backend(),
        }
    }

    /// Create a tensor from raw data, using the global backend.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self {
            shape,
            data,
            backend: default_backend(),
        }
    }

    /// Create a tensor with a specific backend.
    pub fn from_vec_with_backend(data: Vec<f32>, shape: Vec<usize>, backend: &'static dyn Backend) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { shape, data, backend }
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
        Self::from_vec_with_backend(self.data.clone(), shape, self.backend)
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
