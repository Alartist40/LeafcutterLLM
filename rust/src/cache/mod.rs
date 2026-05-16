//! KV Cache for transformer inference
//!
//! Stores Key and Value tensors for each layer to avoid recomputation
//! during autoregressive generation.

use crate::model::tensor::Tensor;

pub struct KVCache {
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            k: Vec::with_capacity(num_layers),
            v: Vec::with_capacity(num_layers),
        }
    }

    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
    }

    pub fn append(&mut self, layer_idx: usize, k: Tensor, v: Tensor) {
        if layer_idx >= self.k.len() {
            self.k.push(k);
            self.v.push(v);
        } else {
            // Concatenate along sequence dimension
            self.k[layer_idx] = self.k[layer_idx].concat(&k, 0);
            self.v[layer_idx] = self.v[layer_idx].concat(&v, 0);
        }
    }

    pub fn get(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        if layer_idx < self.k.len() {
            Some((&self.k[layer_idx], &self.v[layer_idx]))
        } else {
            None
        }
    }
}

impl Tensor {
    /// Concatenate two tensors along a dimension
    pub fn concat(&self, other: &Tensor, dim: usize) -> Tensor {
        assert_eq!(self.shape.len(), other.shape.len());
        assert!(dim < self.shape.len());

        let mut new_shape = self.shape.clone();
        new_shape[dim] += other.shape[dim];

        if dim == 1 {
            // Concat along sequence dimension (most common for KV cache)
            let batch = self.shape[0];
            let head_dim = self.shape[2];
            let seq1 = self.shape[1];
            let seq2 = other.shape[1];

            let mut result = vec![0.0f32; batch * (seq1 + seq2) * head_dim];
            for b in 0..batch {
                for s in 0..seq1 {
                    for h in 0..head_dim {
                        result[b * (seq1 + seq2) * head_dim + s * head_dim + h] =
                            self.data[b * seq1 * head_dim + s * head_dim + h];
                    }
                }
                for s in 0..seq2 {
                    for h in 0..head_dim {
                        result[b * (seq1 + seq2) * head_dim + (seq1 + s) * head_dim + h] =
                            other.data[b * seq2 * head_dim + s * head_dim + h];
                    }
                }
            }
            Tensor::from_vec(result, new_shape)
        } else {
            // Generic fallback
            let mut result = self.data.clone();
            result.extend_from_slice(&other.data);
            Tensor::from_vec(result, new_shape)
        }
    }
}
