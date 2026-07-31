//! Safetensors loader (Rust port of colibri's st.h).
//!
//! Loads tensor metadata from safetensors files (the JSON header) and
//! provides on-demand reading of tensor data via pread.  Unlike
//! HuggingFace's loader (which loads everything into RAM), we read
//! tensors one at a time and can advise the kernel to evict pages
//! after reading, keeping RSS low.
//!
//! Supported dtypes: BF16, F16, F32, U8/I8 (quantized data).
//! All reads return f32 (dequantized for int types).

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::Path;

/// Generic weight provider trait.
///
/// Abstracts over safetensor shards and GGUF files so the streaming engine
/// can load weights from either format without knowing the underlying storage.
pub trait WeightProvider: Send + Sync {
    /// Read an entire tensor as f32.
    fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, String>;
    /// Read a slice of tensor data (offset elements from start, count elements).
    fn read_tensor_slice_f32(&self, name: &str, offset: usize, count: usize) -> Result<Vec<f32>, String>;

    /// Load all weights for one transformer layer.
    ///
    /// The default implementation reads each tensor by its full safetensor-style
    /// name.  Implementations for different weight formats override this to
    /// map names appropriately.
    fn load_layer_weights(
        &self,
        layer_idx: usize,
        layer_type: &str,
        _layer_names: &[&str],
        prefix: &str,
    ) -> Result<HashMap<String, Vec<f32>>, String>
    where
        Self: Sync,
    {
        use rayon::prelude::*;
        let results: Vec<(&str, Result<Vec<f32>, String>)> = _layer_names
            .par_iter()
            .map(|suffix| {
                let full = format!("{prefix}{suffix}");
                let data = self.read_tensor_f32(&full);
                (*suffix, data)
            })
            .collect();
        let mut w = HashMap::new();
        for (suffix, data) in results {
            w.insert(suffix.to_string(), data?);
        }
        Ok(w)
    }
}

/// Tensor metadata for a single stored tensor.
#[derive(Debug, Clone)]
pub struct StTensor {
    pub name: String,
    pub offset: u64,
    pub nbytes: u64,
    pub dtype: StDtype,
    pub numel: u64,
    /// Shape of the tensor (from the JSON header).
    pub shape: Vec<u64>,
}

/// Safetensor dtype codes (matching colibri's st_dtype_code).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDtype {
    Bf16,
    F16,
    F32,
    U8, // also I8 — raw quantized bytes
}

impl StDtype {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "BF16" => Some(Self::Bf16),
            "F16" => Some(Self::F16),
            "F32" => Some(Self::F32),
            "U8" | "I8" => Some(Self::U8),
            _ => None,
        }
    }

    pub fn bytes_per_elem(&self) -> u64 {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
            Self::U8 => 1,
        }
    }
}

/// A collection of safetensors shards with indexed tensor metadata.
///
/// Mirrors Colibri's `shards` struct.  Each shard is a .safetensors
/// file; we keep its file handle open and an index of tensor name →
/// (file_index, offset, size, dtype).
pub struct Shards {
    files: Vec<File>,
    paths: Vec<String>,
    /// name → (file_index, tensor metadata)
    index: HashMap<String, (usize, StTensor)>,
}

impl Shards {
    /// Load all .safetensors files from a directory.
    /// If the directory has a model.safetensors.index.json, uses it to
    /// determine which shards contain which tensors.  Otherwise looks
    /// for a single model.safetensors file.
    pub fn open_dir(dir: &Path) -> Result<Self, String> {
        let index_path = dir.join("model.safetensors.index.json");
        let mut files = Vec::new();
        let mut paths = Vec::new();
        let mut index = HashMap::new();

        if index_path.exists() {
            // Multi-shard: read the index JSON to map tensor names → files.
            let idx_content = std::fs::read_to_string(&index_path)
                .map_err(|e| format!("read index: {e}"))?;
            let idx: serde_json::Value = serde_json::from_str(&idx_content)
                .map_err(|e| format!("parse index json: {e}"))?;
            let map = idx.get("weight_map")
                .ok_or_else(|| "index.json has no weight_map".to_string())?
                .as_object()
                .ok_or_else(|| "weight_map is not an object".to_string())?;

            // Collect unique shard file names.
            let mut shard_names: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            shard_names.sort();
            shard_names.dedup();

            for name in &shard_names {
                let path = dir.join(name);
                let f = File::open(&path).map_err(|e| format!("open {path:?}: {e}"))?;
                paths.push(name.clone());
                files.push(f);
            }

            // Build tensor index.
            for (tensor_name, shard_val) in map {
                if let Some(shard_name) = shard_val.as_str() {
                    let tensor_name = tensor_name.to_string();
                    let file_idx = paths
                        .iter()
                        .position(|p| p == shard_name)
                        .ok_or_else(|| format!("shard {shard_name} not in file list"))?;
                    // We need offset and dtype from the shard's own header.
                    // For now, store a placeholder; we'll read the header below.
                    index.insert(tensor_name.clone(), (file_idx, StTensor {
                        name: tensor_name,
                        offset: 0,
                        nbytes: 0,
                        dtype: StDtype::F32,
                        numel: 0,
                        shape: Vec::new(),
                    }));
                }
            }

            // Read each shard's header to fill in offset/dtype/shape.
            for (fi, file) in files.iter_mut().enumerate() {
                let header = read_safetensors_header(file)
                    .map_err(|e| format!("read header for {:?}: {e}", paths[fi]))?;
                for (name, meta) in &header {
                    if let Some(entry) = index.get_mut(name) {
                        entry.1.offset = meta.offset;
                        entry.1.nbytes = meta.nbytes;
                        entry.1.dtype = meta.dtype;
                        entry.1.numel = meta.numel;
                        entry.1.shape = meta.shape.clone();
                    }
                }
            }
        } else {
            // Single-file model.safetensors
            let single = dir.join("model.safetensors");
            if !single.exists() {
                return Err(format!(
                    "no safetensors found in {dir:?} (no model.safetensors or index.json)"
                ));
            }
            let mut f = File::open(&single).map_err(|e| format!("open {single:?}: {e}"))?;
            let header = read_safetensors_header(&mut f)?;
            for (name, meta) in header {
                index.insert(name.clone(), (0, meta));
            }
            paths.push("model.safetensors".to_string());
            files.push(f);
        }

        Ok(Self { files, paths, index })
    }

    /// Read a SLICE of a tensor: `offset..offset+count` elements, returned as f32.
    /// Reads only `count * dtype_bytes` from disk — does NOT load the whole tensor.
    /// This is the key to streaming: for embedding lookup we read 4096 elements
    /// (8KB BF16) instead of the entire 2GB table.
    pub fn read_tensor_slice_f32(
        &self,
        name: &str,
        offset: usize, // in elements
        count: usize,  // in elements
    ) -> Result<Vec<f32>, String> {
        let (file_idx, meta) = self
            .index
            .get(name)
            .ok_or_else(|| format!("tensor {name:?} not found"))?
            .clone();

        let file = &self.files[file_idx];
        let bytes_per_elem = meta.dtype.bytes_per_elem() as usize;
        let byte_offset = meta.offset as usize + offset * bytes_per_elem;
        let nbytes = count * bytes_per_elem;

        let mut buf = vec![0u8; nbytes];
        file.read_exact_at(&mut buf, byte_offset as u64)
            .map_err(|e| format!("pread: {e}"))?;

        dequant_slice(&buf, meta.dtype, count)
    }

    /// Read entire tensor as f32 (existing behavior — loads whole tensor).
    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let (file_idx, meta) = self
            .index
            .get(name)
            .ok_or_else(|| format!("tensor {name:?} not found"))?
            .clone();

        let file = &self.files[file_idx];
        let mut buf = vec![0u8; meta.nbytes as usize];
        file.read_exact_at(&mut buf, meta.offset)
            .map_err(|e| format!("pread: {e}"))?;

        // Dequantize to f32 based on dtype.
        match meta.dtype {
            StDtype::F32 => {
                let mut out = vec![0f32; meta.numel as usize];
                for (i, chunk) in buf.chunks_exact(4).enumerate() {
                    out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Ok(out)
            }
            StDtype::Bf16 => {
                let mut out = vec![0f32; meta.numel as usize];
                for (i, chunk) in buf.chunks_exact(2).enumerate() {
                    let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out[i] = bf16_to_f32(h);
                }
                Ok(out)
            }
            StDtype::F16 => {
                let mut out = vec![0f32; meta.numel as usize];
                for (i, chunk) in buf.chunks_exact(2).enumerate() {
                    let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out[i] = f16_to_f32(h);
                }
                Ok(out)
            }
            StDtype::U8 => {
                // Raw bytes — return as f32 for uniformity.
                Ok(buf.iter().map(|&b| b as f32).collect())
            }
        }
    }

    /// Get the metadata for a tensor without reading its data.
    pub fn lookup(&self, name: &str) -> Option<&StTensor> {
        self.index.get(name).map(|(_, m)| m)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.index.keys().map(|k| k.as_str()).collect()
    }
}

impl WeightProvider for Shards {
    fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        self.read_tensor_f32(name)
    }

    fn read_tensor_slice_f32(&self, name: &str, offset: usize, count: usize) -> Result<Vec<f32>, String> {
        self.read_tensor_slice_f32(name, offset, count)
    }
}

/// Read the JSON header of a safetensors file.
/// Returns a map of tensor_name → StTensor (offset, dtype, numel, shape).
fn read_safetensors_header(file: &mut File) -> Result<HashMap<String, StTensor>, String> {
    // First 8 bytes: u64 LE header length.
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|e| format!("read header len: {e}"))?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    if header_len > (512 << 20) {
        return Err(format!("safetensors header too large: {header_len} bytes"));
    }

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf).map_err(|e| format!("read header: {e}"))?;

    let header: serde_json::Value = serde_json::from_slice(&header_buf)
        .map_err(|e| format!("parse header json: {e}"))?;

    // Data section starts right after the header.
    let data_start = 8 + header_len as u64;

    let map = header
        .as_object()
        .ok_or_else(|| "safetensors header is not an object".to_string())?;

    let mut out: HashMap<String, StTensor> = HashMap::new();
    for (name, val) in map {
        // Skip metadata keys like "__metadata__".
        if name == "__metadata__" {
            continue;
        }
        let name = name.to_string();
        let obj = val
            .as_object()
            .ok_or_else(|| format!("{name}: not an object"))?;
        let dtype_str = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{name}: no dtype"))?;
        let dtype = StDtype::from_str(dtype_str)
            .ok_or_else(|| format!("{name}: unknown dtype {dtype_str}"))?;
        let shape: Vec<u64> = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{name}: no data_offsets"))?
            .iter()
            .filter_map(|v| v.as_u64())
            .collect();
        // Actually shape is a separate field; data_offsets has [start, end].
        let offsets = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{name}: no data_offsets"))?;
        let off_start = offsets
            .get(0)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("{name}: bad offset[0]"))?;
        let off_end = offsets
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("{name}: bad offset[1]"))?;

        let shape_field = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|v| v.as_u64())
            .collect::<Vec<u64>>();

        let numel: u64 = if shape_field.is_empty() {
            // Scalar
            (off_end - off_start) / dtype.bytes_per_elem()
        } else {
            shape_field.iter().product()
        };

        out.insert(
            name.clone(),
            StTensor {
                name,
                offset: data_start + off_start,
                nbytes: off_end - off_start,
                dtype,
                numel,
                shape: shape_field,
            },
        );
    }

    Ok(out)
}

/// Convert BF16 (brain float16) bits to f32.
#[inline]
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Convert IEEE f16 bits to f32.
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h as u32) >> 10) & 0x1F;
    let mant = (h as u32) & 0x3FF;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            // Subnormal
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            f32::from_bits(sign | (e << 23) | (m << 13))
        }
    } else if exp == 0x1F {
        f32::from_bits(sign | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
    }
}

/// Dequantize a byte slice to `count` f32 values. Same logic as
/// `read_tensor_f32` but works on an already-read buffer.
fn dequant_slice(buf: &[u8], dtype: StDtype, count: usize) -> Result<Vec<f32>, String> {
    match dtype {
        StDtype::F32 => {
            if buf.len() < count * 4 {
                return Err(format!(
                    "F32 slice too short: {} bytes for {} elements",
                    buf.len(),
                    count
                ));
            }
            let mut out = vec![0f32; count];
            for (i, chunk) in buf.chunks_exact(4).take(count).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(out)
        }
        StDtype::Bf16 => {
            if buf.len() < count * 2 {
                return Err(format!(
                    "BF16 slice too short: {} bytes for {} elements",
                    buf.len(),
                    count
                ));
            }
            let mut out = vec![0f32; count];
            for (i, chunk) in buf.chunks_exact(2).take(count).enumerate() {
                let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                out[i] = bf16_to_f32(h);
            }
            Ok(out)
        }
        StDtype::F16 => {
            if buf.len() < count * 2 {
                return Err(format!(
                    "F16 slice too short: {} bytes for {} elements",
                    buf.len(),
                    count
                ));
            }
            let mut out = vec![0f32; count];
            for (i, chunk) in buf.chunks_exact(2).take(count).enumerate() {
                let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                out[i] = f16_to_f32(h);
            }
            Ok(out)
        }
        StDtype::U8 => {
            if buf.len() < count {
                return Err(format!(
                    "U8 slice too short: {} bytes for {} elements",
                    buf.len(),
                    count
                ));
            }
            Ok(buf.iter().take(count).map(|&b| b as f32).collect())
        }
    }
}
