//! KV Cache for transformer inference
//!
//! Stores Key and Value tensors for each layer in f32 format.
//! Previous versions used f16 to reduce RAM by 2×, but the f16→f32→f16
//! round-trip on every append accumulated quantization noise that degraded
//! generation quality over long sequences.  f32 storage eliminates this.
//!
//! Uses HashMap keyed by layer index to support sparse hybrid architectures
//! (e.g. Qwen3.5 where only some layers use attention).

pub mod ssm_state;
pub mod deltanet_state;

use crate::model::tensor::Tensor;
use std::collections::HashMap;

/// Per-layer KV cache entry. K, V, and shape are stored **atomically** in one
/// struct so that they can never desync (the previous design used three
/// separate HashMaps, which risked panic if K was inserted but V was missing).
#[derive(Clone)]
struct KVEntry {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Shape: [seq_len, num_kv_heads, head_dim]
    shape: Vec<usize>,
}

pub struct KVCache {
    /// f32 storage per layer (atomic K/V/shape trio)
    layers: HashMap<usize, KVEntry>,
}

impl KVCache {
    pub fn new(_num_layers: usize) -> Self {
        Self {
            layers: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.layers.clear();
    }

    /// Append K and V tensors for a layer. Input is f32; stored as f32.
    /// K, V, and shape are updated atomically — no panics from desync.
    pub fn append(&mut self, layer_idx: usize, k: Tensor, v: Tensor) {
        if let Some(entry) = self.layers.get_mut(&layer_idx) {
            entry.k.extend_from_slice(&k.data);
            entry.v.extend_from_slice(&v.data);
            entry.shape[0] += k.shape[0];
        } else {
            self.layers.insert(
                layer_idx,
                KVEntry {
                    k: k.data,
                    v: v.data,
                    shape: k.shape,
                },
            );
        }
    }

    /// Get f32 K and V tensors for a layer.
    pub fn get(&self, layer_idx: usize) -> Option<(Tensor, Tensor)> {
        let entry = self.layers.get(&layer_idx)?;
        Some((
            Tensor::from_vec(entry.k.clone(), entry.shape.clone()),
            Tensor::from_vec(entry.v.clone(), entry.shape.clone()),
        ))
    }

    /// Report memory usage in bytes (f32 storage).
    pub fn memory_bytes(&self) -> usize {
        self.layers
            .values()
            .map(|e| (e.k.len() + e.v.len()) * 4)
            .sum()
    }

    /// Total sequence length cached (same for all layers; returns first layer's seq_len).
    pub fn total_seq_len(&self) -> usize {
        self.layers
            .values()
            .next()
            .map(|e| e.shape.get(0).copied().unwrap_or(0))
            .unwrap_or(0)
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
    fn test_kv_cache_f32_exact() {
        let mut cache = KVCache::new(2);
        let k = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 1, 2]);
        let v = Tensor::from_vec(vec![0.5f32, 1.5, 2.5, 3.5], vec![2, 1, 2]);

        cache.append(0, k.clone(), v.clone());
        let (k_out, v_out) = cache.get(0).unwrap();

        assert_eq!(k_out.data, k.data);
        assert_eq!(v_out.data, v.data);
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
    fn test_kv_cache_append_twice_exact() {
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
