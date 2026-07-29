//! Safetensor-backed tensor map.
//!
//! Wraps our `Shards` loader to provide a `HashMap`-style interface
//! for the existing leafcutter inference code (which expects
//! `HashMap<String, Tensor>`).
//!
//! Tensors are read on first access (via safetensors `pread`) and
//! cached for subsequent lookups. Cache size is bounded to avoid
//! unbounded RAM growth on small machines.

use crate::model::tensor::Tensor;
use crate::safetensors_loader::Shards;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// Lazily reads and caches tensors from a safetensors model directory.
pub struct SafetensorTensors {
    shards: Mutex<Shards>,
    cache: Mutex<HashMap<String, Tensor>>,
    /// Set of tensor names we've already warned about (to avoid spam).
    missing: Mutex<std::collections::HashSet<String>>,
}

impl SafetensorTensors {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let shards = Shards::open_dir(dir)?;
        Ok(Self {
            shards: Mutex::new(shards),
            cache: Mutex::new(HashMap::new()),
            missing: Mutex::new(std::collections::HashSet::new()),
        })
    }

    /// Get a tensor by name. Returns None if missing.
    pub fn get(&self, name: &str) -> Option<Tensor> {
        // Fast path: cache hit.
        {
            let cache = self.cache.lock().ok()?;
            if let Some(t) = cache.get(name) {
                return Some(clone_tensor(t));
            }
        }
        // Slow path: read from safetensors, then cache.
        let data = {
            let mut shards = self.shards.lock().ok()?;
            match shards.read_tensor_f32(name) {
                Ok(d) => d,
                Err(_) => {
                    let mut missing = self.missing.lock().ok()?;
                    if missing.insert(name.to_string()) {
                        eprintln!("[safetensor-tensors] missing tensor: {name}");
                    }
                    return None;
                }
            }
        };
        // Look up shape from the index.
        let shape = {
            let shards = self.shards.lock().ok()?;
            match shards.lookup(name) {
                Some(meta) => meta.shape.iter().map(|&x| x as usize).collect(),
                None => vec![data.len()],
            }
        };
        let tensor = Tensor::from_vec(data, shape);
        {
            let mut cache = self.cache.lock().ok()?;
            cache.insert(name.to_string(), clone_tensor(&tensor));
        }
        Some(tensor)
    }

    /// List all available tensor names.
    pub fn tensor_names(&self) -> Vec<String> {
        self.shards
            .lock()
            .ok()
            .map(|s| s.tensor_names().iter().map(|x| x.to_string()).collect())
            .unwrap_or_default()
    }
}

/// Clone a Tensor — since Tensor's fields are mostly public but the
/// struct doesn't impl Clone, we manually replicate.
fn clone_tensor(t: &Tensor) -> Tensor {
    Tensor::from_vec(t.data.clone(), t.shape.clone())
}
