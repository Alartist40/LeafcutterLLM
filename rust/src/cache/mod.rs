//! KV Cache for transformer inference
//!
//! Stores Key and Value tensors for each layer in f16 format to reduce
//! RAM usage by 2×. Decompresses to f32 on demand for computation.
//! Uses HashMap keyed by layer index to support sparse hybrid architectures
//! (e.g. Qwen3.5 where only some layers use attention).

pub mod ssm_state;

use crate::model::tensor::Tensor;
use half::f16;
use std::collections::HashMap;

pub struct KVCache {
    /// Compressed f16 storage per layer
    k_compressed: HashMap<usize, Vec<f16>>,
    v_compressed: HashMap<usize, Vec<f16>>,
    /// Shape for each layer's K/V tensor: [seq_len, num_kv_heads, head_dim]
    shapes: HashMap<usize, Vec<usize>>,
}

impl KVCache {
    pub fn new(_num_layers: usize) -> Self {
        Self {
            k_compressed: HashMap::new(),
            v_compressed: HashMap::new(),
            shapes: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.k_compressed.clear();
        self.v_compressed.clear();
        self.shapes.clear();
    }

    /// Append K and V tensors for a layer. Input is f32; stored as f16.
    pub fn append(&mut self, layer_idx: usize, k: Tensor, v: Tensor) {
        let k_f16: Vec<f16> = k.data.iter().map(|&x| f16::from_f32(x)).collect();
        let v_f16: Vec<f16> = v.data.iter().map(|&x| f16::from_f32(x)).collect();

        if let Some(existing_k) = self.k_compressed.get_mut(&layer_idx) {
            // Concatenate: decompress existing, append new, recompress
            let mut existing_k_f32: Vec<f32> = existing_k.iter().map(|&x| x.to_f32()).collect();
            let mut existing_v_f32: Vec<f32> =
                self.v_compressed.get(&layer_idx).unwrap().iter().map(|&x| x.to_f32()).collect();

            existing_k_f32.extend_from_slice(&k.data);
            existing_v_f32.extend_from_slice(&v.data);

            *existing_k = existing_k_f32.iter().map(|&x| f16::from_f32(x)).collect();
            self.v_compressed.insert(layer_idx, existing_v_f32.iter().map(|&x| f16::from_f32(x)).collect());

            // Update shape along sequence dimension (dim 0)
            self.shapes.get_mut(&layer_idx).unwrap()[0] += k.shape[0];
        } else {
            // First time for this layer
            self.k_compressed.insert(layer_idx, k_f16);
            self.v_compressed.insert(layer_idx, v_f16);
            self.shapes.insert(layer_idx, k.shape.clone());
        }
    }

    /// Get decompressed f32 K and V tensors for a layer.
    /// Returns owned Tensors (decompresses from f16 on demand).
    pub fn get(&self, layer_idx: usize) -> Option<(Tensor, Tensor)> {
        let k_f16 = self.k_compressed.get(&layer_idx)?;
        let v_f16 = self.v_compressed.get(&layer_idx)?;

        let k_f32: Vec<f32> = k_f16.iter().map(|&x| x.to_f32()).collect();
        let v_f32: Vec<f32> = v_f16.iter().map(|&x| x.to_f32()).collect();

        let shape = self.shapes.get(&layer_idx)?.clone();
        Some((Tensor::from_vec(k_f32, shape.clone()), Tensor::from_vec(v_f32, shape)))
    }

    /// Report memory usage in bytes (compressed f16 storage).
    pub fn memory_bytes(&self) -> usize {
        let k_bytes: usize = self.k_compressed.values().map(|v| v.len() * 2).sum();
        let v_bytes: usize = self.v_compressed.values().map(|v| v.len() * 2).sum();
        k_bytes + v_bytes
    }

    /// Total sequence length cached across all layers.
    pub fn total_seq_len(&self) -> usize {
        self.shapes.values().map(|s| s.get(0).copied().unwrap_or(0)).sum()
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
        assert!(cache.memory_bytes() <= f32_bytes / 2 + 16); // small overhead
    }

    #[test]
    fn test_kv_cache_sparse_layers() {
        let mut cache = KVCache::new(4);
        let k = Tensor::from_vec(vec![1.0f32, 2.0], vec![1, 1, 2]);
        let v = Tensor::from_vec(vec![0.5f32, 1.5], vec![1, 1, 2]);

        // Only append to layers 1 and 3
        cache.append(1, k.clone(), v.clone());
        cache.append(3, k.clone(), v.clone());

        assert!(cache.get(0).is_none());
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_kv_cache_append_twice() {
        let mut cache = KVCache::new(2);
        let k1 = Tensor::from_vec(vec![1.0f32, 2.0], vec![1, 1, 2]);
        let v1 = Tensor::from_vec(vec![0.5f32, 1.5], vec![1, 1, 2]);
        let k2 = Tensor::from_vec(vec![3.0f32, 4.0], vec![1, 1, 2]);
        let v2 = Tensor::from_vec(vec![2.5f32, 3.5], vec![1, 1, 2]);

        cache.append(0, k1, v1);
        cache.append(0, k2, v2);

        let (k_out, v_out) = cache.get(0).unwrap();
        assert_eq!(k_out.shape, vec![2, 1, 2]);
        assert_eq!(k_out.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v_out.data, vec![0.5, 1.5, 2.5, 3.5]);
    }
}
