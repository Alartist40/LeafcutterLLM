//! KV Cache for transformer inference
//!
//! Stores Key and Value tensors for each layer in f16 format to reduce
//! RAM usage by 2×. Decompresses to f32 on demand for computation.

use crate::model::tensor::Tensor;
use half::f16;

pub struct KVCache {
    /// Compressed f16 storage per layer
    k_compressed: Vec<Vec<f16>>,
    v_compressed: Vec<Vec<f16>>,
    /// Shape for each layer's K/V tensor: [seq_len, num_kv_heads, head_dim]
    shapes: Vec<Vec<usize>>,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            k_compressed: Vec::with_capacity(num_layers),
            v_compressed: Vec::with_capacity(num_layers),
            shapes: Vec::with_capacity(num_layers),
        }
    }

    pub fn clear(&mut self) {
        self.k_compressed.clear();
        self.v_compressed.clear();
        self.shapes.clear();
    }

    /// Append K and V tensors for a layer. Input is f32; stored as f16.
    pub fn append(&mut self, layer_idx: usize, k: Tensor, v: Tensor) {
        // Convert f32 to f16 for storage
        let k_f16: Vec<f16> = k.data.iter().map(|&x| f16::from_f32(x)).collect();
        let v_f16: Vec<f16> = v.data.iter().map(|&x| f16::from_f32(x)).collect();

        if layer_idx >= self.k_compressed.len() {
            // First time for this layer
            self.k_compressed.push(k_f16);
            self.v_compressed.push(v_f16);
            self.shapes.push(k.shape.clone());
        } else {
            // Concatenate: decompress existing, append new, recompress
            let mut existing_k: Vec<f32> =
                self.k_compressed[layer_idx].iter().map(|&x| x.to_f32()).collect();
            let mut existing_v: Vec<f32> =
                self.v_compressed[layer_idx].iter().map(|&x| x.to_f32()).collect();

            existing_k.extend_from_slice(&k.data);
            existing_v.extend_from_slice(&v.data);

            self.k_compressed[layer_idx] =
                existing_k.iter().map(|&x| f16::from_f32(x)).collect();
            self.v_compressed[layer_idx] =
                existing_v.iter().map(|&x| f16::from_f32(x)).collect();

            // Update shape along sequence dimension (dim 0)
            self.shapes[layer_idx][0] += k.shape[0];
        }
    }

    /// Get decompressed f32 K and V tensors for a layer.
    /// Returns owned Tensors (decompresses from f16 on demand).
    pub fn get(&self, layer_idx: usize) -> Option<(Tensor, Tensor)> {
        if layer_idx >= self.k_compressed.len() {
            return None;
        }

        let k_f32: Vec<f32> =
            self.k_compressed[layer_idx].iter().map(|&x| x.to_f32()).collect();
        let v_f32: Vec<f32> =
            self.v_compressed[layer_idx].iter().map(|&x| x.to_f32()).collect();

        let shape = self.shapes[layer_idx].clone();
        Some((Tensor::from_vec(k_f32, shape.clone()), Tensor::from_vec(v_f32, shape)))
    }

    /// Report memory usage in bytes (compressed f16 storage).
    pub fn memory_bytes(&self) -> usize {
        let k_bytes: usize = self.k_compressed.iter().map(|v| v.len() * 2).sum();
        let v_bytes: usize = self.v_compressed.iter().map(|v| v.len() * 2).sum();
        k_bytes + v_bytes
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_f16_roundtrip() {
        let mut cache = KVCache::new(2);
        let k = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 1, 2]);
        let v = Tensor::from_vec(vec![0.5f32, 1.5, 2.5, 3.5], vec![2, 1, 2]);

        cache.append(0, k.clone(), v.clone());
        let (k_out, v_out) = cache.get(0).unwrap();

        assert_eq!(k_out.data, k.data);
        assert_eq!(v_out.data, v.data);

        // Memory should be ~half of f32
        let f32_bytes = k.data.len() * 4 + v.data.len() * 4;
        let f16_bytes = cache.memory_bytes();
        assert_eq!(f16_bytes, f32_bytes / 2);
    }

    #[test]
    fn test_kv_cache_append() {
        let mut cache = KVCache::new(2);
        let k1 = Tensor::from_vec(vec![1.0f32, 2.0], vec![1, 1, 2]);
        let v1 = Tensor::from_vec(vec![3.0f32, 4.0], vec![1, 1, 2]);
        let k2 = Tensor::from_vec(vec![5.0f32, 6.0], vec![1, 1, 2]);
        let v2 = Tensor::from_vec(vec![7.0f32, 8.0], vec![1, 1, 2]);

        cache.append(0, k1, v1);
        cache.append(0, k2, v2);

        let (k_out, v_out) = cache.get(0).unwrap();
        assert_eq!(k_out.shape, vec![2, 1, 2]);
        assert_eq!(k_out.data, vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(v_out.data, vec![3.0, 4.0, 7.0, 8.0]);
    }
}
