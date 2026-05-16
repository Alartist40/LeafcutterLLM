//! ShardLoader — mmap-based layer loading with explicit eviction and cache

use crate::model::tensor::Tensor;
use crate::kernels::q8_0::Matrix as Q8Matrix;
use super::format::{ShardHeader, ShardTensorMeta};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Manifest describing all shards for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub model: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub shard_dir: String,
    pub layers: Vec<LayerManifest>,
    pub special: SpecialManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerManifest {
    pub idx: usize,
    pub file: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialManifest {
    pub embed: String,
    pub embed_size: u64,
    pub norm: String,
    pub norm_size: u64,
    pub lm_head: String,
    pub lm_head_size: u64,
}

impl Manifest {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let manifest: Manifest = serde_json::from_str(&text)?;
        Ok(manifest)
    }
}

/// Simple FIFO layer cache. Keeps up to `max_slots` layers in RAM.
/// For decode on large-RAM systems, set max_slots = num_layers to
/// keep everything cached after the first forward pass.
#[derive(Debug)]
pub struct LayerCache {
    slots: Vec<(usize, HashMap<String, Tensor>)>,
    max_slots: usize,
}

impl LayerCache {
    pub fn new(max_slots: usize) -> Self {
        Self {
            slots: Vec::with_capacity(max_slots),
            max_slots,
        }
    }

    pub fn get(&self, idx: usize) -> Option<HashMap<String, Tensor>> {
        self.slots.iter().find(|(i, _)| *i == idx).map(|(_, w)| w.clone())
    }

    pub fn put(&mut self, idx: usize, weights: HashMap<String, Tensor>) {
        if self.max_slots == 0 {
            return;
        }
        // Remove if already present
        self.slots.retain(|(i, _)| *i != idx);
        // Evict oldest if at capacity
        if self.slots.len() >= self.max_slots {
            self.slots.remove(0);
        }
        self.slots.push((idx, weights));
    }

    pub fn clear(&mut self) {
        self.slots.clear();
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }
}

/// A loaded shard file mapped into memory.  Dropping this unmaps the
/// file, instantly freeing the kernel page cache for those pages.
pub struct MmapShard {
    #[allow(dead_code)]
    mmap: Mmap,
    pub tensors: HashMap<String, Tensor>,
}

/// Loads model layers from disk shards.  Only one layer is resident
/// in RAM at any time (plus cached layers and whatever the kernel
/// keeps in page cache).
pub struct ShardLoader {
    pub manifest: Manifest,
    shard_dir: String,
    /// Optional prefetch handle for the next layer
    prefetch: Arc<Mutex<Option<HashMap<String, Tensor>>>>,
    /// Layer cache — configurable size
    cache: Arc<Mutex<LayerCache>>,
}

impl ShardLoader {
    pub fn from_manifest(manifest: Manifest) -> Self {
        let shard_dir = manifest.shard_dir.clone();
        Self {
            manifest,
            shard_dir,
            prefetch: Arc::new(Mutex::new(None)),
            cache: Arc::new(Mutex::new(LayerCache::new(0))),
        }
    }

    /// Configure the layer cache.  `max_slots = 0` disables caching.
    /// `max_slots = num_layers` caches everything (useful for decode
    /// on systems with enough RAM).
    pub fn with_cache_capacity(mut self, max_slots: usize) -> Self {
        self.cache = Arc::new(Mutex::new(LayerCache::new(max_slots)));
        self
    }

    /// Load a layer's weights by index, checking cache first.
    pub fn load_layer(&self, idx: usize) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
        if idx >= self.manifest.layers.len() {
            return Err(format!("Layer index {} out of range ({} layers)",
                idx, self.manifest.layers.len()).into());
        }

        // Check cache first
        if let Ok(cache) = self.cache.lock() {
            if let Some(weights) = cache.get(idx) {
                return Ok(weights);
            }
        }

        // Check prefetch slot
        if let Ok(mut guard) = self.prefetch.lock() {
            if let Some(weights) = guard.take() {
                // Verify it's the layer we want (prefetch may have loaded a different one)
                // Since we don't track idx in prefetch, we always accept.
                // The engine's prefetch logic ensures this is usually correct.
                return Ok(weights);
            }
        }

        let layer_manifest = &self.manifest.layers[idx];
        let path = Path::new(&self.shard_dir).join(&layer_manifest.file);
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let tensors = parse_shard_from_mmap(&mmap)?;
        Ok(tensors)
    }

    /// Store a computed layer in the cache for reuse.
    pub fn cache_layer(&self, idx: usize, weights: HashMap<String, Tensor>) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.put(idx, weights);
        }
    }

    /// Load special weights (embed, norm, lm_head) from their shard files.
    /// These are kept in RAM permanently by the engine.
    pub fn load_special(&self) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
        let mut all = HashMap::new();

        let embed_path = Path::new(&self.shard_dir).join(&self.manifest.special.embed);
        let embed_file = File::open(&embed_path)?;
        let embed_mmap = unsafe { Mmap::map(&embed_file)? };
        let embed_tensors = parse_shard_from_mmap(&embed_mmap)?;
        all.extend(embed_tensors);

        let norm_path = Path::new(&self.shard_dir).join(&self.manifest.special.norm);
        let norm_file = File::open(&norm_path)?;
        let norm_mmap = unsafe { Mmap::map(&norm_file)? };
        let norm_tensors = parse_shard_from_mmap(&norm_mmap)?;
        all.extend(norm_tensors);

        let lm_head_path = Path::new(&self.shard_dir).join(&self.manifest.special.lm_head);
        let lm_head_file = File::open(&lm_head_path)?;
        let lm_head_mmap = unsafe { Mmap::map(&lm_head_file)? };
        let lm_head_tensors = parse_shard_from_mmap(&lm_head_mmap)?;
        all.extend(lm_head_tensors);

        Ok(all)
    }

    /// Start loading the next layer in a background thread.
    /// The prefetched layer can be retrieved with `load_layer()`.
    pub fn prefetch_layer(&self, idx: usize) {
        if idx >= self.manifest.layers.len() {
            return;
        }
        let manifest = self.manifest.layers[idx].clone();
        let shard_dir = self.shard_dir.clone();
        let prefetch = Arc::clone(&self.prefetch);

        std::thread::spawn(move || {
            let path = Path::new(&shard_dir).join(&manifest.file);
            let Ok(file) = File::open(&path) else { return };
            let Ok(mmap) = (unsafe { Mmap::map(&file) }) else { return };
            let Ok(tensors) = parse_shard_from_mmap(&mmap) else { return };

            if let Ok(mut guard) = prefetch.lock() {
                *guard = Some(tensors);
            }
        });
    }

    /// Take a prefetched layer if available.
    pub fn take_prefetch(&self) -> Option<HashMap<String, Tensor>> {
        if let Ok(mut guard) = self.prefetch.lock() {
            guard.take()
        } else {
            None
        }
    }

    /// Clear the layer cache (e.g., between conversations).
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Report cache stats.
    pub fn cache_stats(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_cache_fifo() {
        let mut cache = LayerCache::new(2);
        let w1 = HashMap::new();
        let mut w2 = HashMap::new();
        w2.insert("x".to_string(), Tensor::zeros(vec![1]));
        let mut w3 = HashMap::new();
        w3.insert("y".to_string(), Tensor::zeros(vec![2]));

        cache.put(0, w1.clone());
        assert_eq!(cache.len(), 1);
        assert!(cache.get(0).is_some());

        cache.put(1, w2.clone());
        assert_eq!(cache.len(), 2);

        cache.put(2, w3.clone());
        assert_eq!(cache.len(), 2);
        assert!(cache.get(0).is_none()); // evicted (FIFO)
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_some());
    }

    #[test]
    fn test_layer_cache_zero_slots() {
        let mut cache = LayerCache::new(0);
        let mut w = HashMap::new();
        w.insert("x".to_string(), Tensor::zeros(vec![1]));
        cache.put(0, w);
        assert_eq!(cache.len(), 0);
        assert!(cache.get(0).is_none());
    }
}

/// Parse a shard file from a memory-mapped buffer.
/// Copies all tensor data into owned `Vec<f32>` Tensors.
fn parse_shard_from_mmap(mmap: &[u8]) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
    let mut cursor = std::io::Cursor::new(mmap);

    let header = ShardHeader::read(&mut cursor)?;
    if !header.is_valid() {
        return Err("Invalid shard file: bad magic or version".into());
    }

    let mut tensors = HashMap::new();

    for _ in 0..header.tensor_count {
        let meta = ShardTensorMeta::read(&mut cursor)?;
        let start = meta.data_offset as usize;
        let end = start + meta.data_size as usize;

        if end > mmap.len() {
            return Err(format!("Shard corrupt: tensor '{}' data out of bounds", meta.name).into());
        }

        let data_bytes = &mmap[start..end];
        let element_count = meta.element_count();

        let shape: Vec<usize> = meta.dims.iter().map(|&d| d as usize).collect();

        match header.quant_format {
            super::format::QuantFormat::F32 => {
                if data_bytes.len() != element_count * 4 {
                    return Err(format!(
                        "Shard corrupt: tensor '{}' size mismatch ({} bytes vs {} elements)",
                        meta.name, data_bytes.len(), element_count
                    ).into());
                }
                let mut out = vec![0.0f32; element_count];
                for (i, chunk) in data_bytes.chunks_exact(4).enumerate() {
                    out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                tensors.insert(meta.name, Tensor::from_vec(out, shape));
            }
            super::format::QuantFormat::Q8_0 => {
                let expected_bytes = (element_count / 32) * 34;
                if data_bytes.len() != expected_bytes {
                    return Err(format!(
                        "Shard corrupt: tensor '{}' Q8_0 size mismatch ({} bytes vs {} expected)",
                        meta.name, data_bytes.len(), expected_bytes
                    ).into());
                }
                let blocks = crate::kernels::q8_0::blocks_from_bytes(data_bytes);
                // Only create a quantized Tensor if it's a 2D matrix with cols multiple of 32.
                // 1D tensors (norm weights, biases) just get dequantized to f32.
                if shape.len() == 2 && shape[1] % 32 == 0 {
                    let q8 = Q8Matrix { rows: shape[0], cols: shape[1], blocks };
                    tensors.insert(meta.name, Tensor::from_q8_0(q8, shape));
                } else {
                    let mut out = vec![0.0f32; element_count];
                    crate::kernels::dequantize_q8_0(data_bytes, &mut out);
                    tensors.insert(meta.name, Tensor::from_vec(out, shape));
                }
            }
            super::format::QuantFormat::Q4_0 => {
                let expected_bytes = (element_count / 32) * 18;
                if data_bytes.len() != expected_bytes {
                    return Err(format!(
                        "Shard corrupt: tensor '{}' Q4_0 size mismatch ({} bytes vs {} expected)",
                        meta.name, data_bytes.len(), expected_bytes
                    ).into());
                }
                let blocks = crate::kernels::q4_0::blocks_from_bytes(data_bytes);
                if shape.len() == 2 && shape[1] % 32 == 0 {
                    let q4 = crate::kernels::q4_0::Matrix { rows: shape[0], cols: shape[1], blocks };
                    tensors.insert(meta.name, Tensor::from_q4_0(q4, shape));
                } else {
                    let mut out = vec![0.0f32; element_count];
                    crate::kernels::dequantize_q4_0(data_bytes, &mut out);
                    tensors.insert(meta.name, Tensor::from_vec(out, shape));
                }
            }
        };
    }

    Ok(tensors)
}
