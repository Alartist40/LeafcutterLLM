//! GGUF v3 parser with K-quant support
//!
//! Implements zero-copy mmap-based loading of GGUF files.
//! Supports Q4_0, Q8_0, Q4_K, Q5_K, Q6_K tensor types.

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

use super::quant::{QuantType, QuantSummary};
// use crate::kernels;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian
pub const GGUF_VERSION: u32 = 3;
pub const QK_K: usize = 256;

#[derive(Debug, Error)]
pub enum GGUError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid GGUF magic: {0:#x}")]
    InvalidMagic(u32),
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("Invalid tensor type: {0}")]
    InvalidTensorType(u32),
    #[error("Truncated data")]
    TruncatedData,
    #[error("Missing tensor: {0}")]
    MissingTensor(String),
    /// Quant type `{1}` (which we don't have a kernel for: `{0}`).
    /// Returned instead of silently failing dequant when the user picks an
    /// unsupported format. See quant.rs `is_supported()`.
    #[error("Unsupported quant type: {0} (numeric={1})")]
    UnsupportedQuantType(String, u32),
}

#[derive(Debug, Clone, Hash)]
pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
}

#[derive(Debug, Clone, Hash)]
pub struct GGUFTensor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub typ: u32,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GGUFile {
    pub header: GGUFHeader,
    pub metadata: std::collections::HashMap<String, GGUFValue>,
    pub tensors: Vec<GGUFTensor>,
    pub data_offset: u64,
    mmap: Mmap,
}

#[derive(Debug, Clone)]
pub enum GGUFValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
}

impl std::hash::Hash for GGUFValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            GGUFValue::U8(v) => v.hash(state),
            GGUFValue::I8(v) => v.hash(state),
            GGUFValue::U16(v) => v.hash(state),
            GGUFValue::I16(v) => v.hash(state),
            GGUFValue::U32(v) => v.hash(state),
            GGUFValue::I32(v) => v.hash(state),
            // Hash the raw bits so f32/f64 have a stable hash regardless of
            // NaN payload handling.
            GGUFValue::F32(v) => v.to_bits().hash(state),
            GGUFValue::U64(v) => v.hash(state),
            GGUFValue::I64(v) => v.hash(state),
            GGUFValue::F64(v) => v.to_bits().hash(state),
            GGUFValue::Bool(v) => v.hash(state),
            GGUFValue::String(v) => v.hash(state),
            GGUFValue::Array(v) => v.hash(state),
        }
    }
}

impl GGUFile {
    /// Total size of the GGUF file in bytes (= length of the mmap region).
    pub fn file_size_bytes(&self) -> u64 {
        self.mmap.len() as u64
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GGUError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut reader = GGUFReader::new(&mmap);

        let header = reader.read_header()?;
        if header.magic != GGUF_MAGIC {
            return Err(GGUError::InvalidMagic(header.magic));
        }
        if header.version != GGUF_VERSION {
            return Err(GGUError::UnsupportedVersion(header.version));
        }

        let mut metadata = std::collections::HashMap::new();
        for _ in 0..header.metadata_count {
            let (key, value) = reader.read_metadata_kv()?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(header.tensor_count as usize);
        for _ in 0..header.tensor_count {
            tensors.push(reader.read_tensor_info()?);
        }

        // Align data section to boundary (default 32 bytes)
        let alignment = metadata.get("general.alignment")
            .and_then(|v| match v {
                GGUFValue::U32(v) => Some(*v as u64),
                GGUFValue::U64(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(32);
        let pos = reader.pos as u64;
        let padding = (alignment - (pos % alignment)) % alignment;
        let data_offset = pos + padding;

        Ok(Self {
            header,
            metadata,
            tensors,
            data_offset,
            mmap,
        })
    }

    pub fn get_tensor_raw(&self, name: &str) -> Option<&[u8]> {
        let t = self.tensors.iter().find(|t| t.name == name)?;
        let size = calculate_tensor_size(&t.dimensions, t.typ);
        let start = self.data_offset + t.offset;
        let end = start + size as u64;
        // Bounds-check against the actual mmap length to prevent OOB panic
        // on truncated or crafted GGUF files.
        if end > self.mmap.len() as u64 {
            eprintln!(
                "Leafcutter: tensor '{}' data extends past mmap boundary (end={}, mmap_len={})",
                name, end, self.mmap.len()
            );
            return None;
        }
        Some(&self.mmap[start as usize..end as usize])
    }

    /// Stable config fingerprint of this model file: hashes the GGUF metadata
    /// key/value pairs (sorted) plus every tensor's name/type/dims.
    ///
    /// This is the "config fingerprint" idea from the kimi-k3-in-c analysis:
    /// any change to the model file that affects how it should be loaded
    /// (architecture, quant type, tensor layout, even the tokenizer template)
    /// changes the fingerprint, so cached/saved state keyed on it can be
    /// detected as stale.
    pub fn fingerprint(&self) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.header.version.hash(&mut hasher);
        self.header.metadata_count.hash(&mut hasher);
        self.header.tensor_count.hash(&mut hasher);
        let mut keys: Vec<&String> = self.metadata.keys().collect();
        keys.sort();
        for k in keys {
            k.hash(&mut hasher);
            self.metadata[k].hash(&mut hasher);
        }
        let mut tensors: Vec<&GGUFTensor> = self.tensors.iter().collect();
        tensors.sort_by_key(|t| t.name.as_str());
        for t in tensors {
            t.name.hash(&mut hasher);
            t.typ.hash(&mut hasher);
            t.dimensions.hash(&mut hasher);
        }
        format!("{:016x}", hasher.finish())
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<&GGUFTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Read and dequantize a single row from a 2D tensor.
    ///
    /// GGUF stores 2D tensors in row-major order.  For quantized types,
    /// each row consists of an integral number of blocks.
    pub fn get_tensor_row_f32(&self, name: &str, row_idx: usize) -> Option<Vec<f32>> {
        let info = self.get_tensor_info(name)?;
        let qtype = QuantType::from_u32(info.typ)?;
        // GGUF stores 2D weight matrices as [inner_dim, outer_dim].
        // Each "row" we read is a contiguous chunk of inner_dim elements,
        // and there are outer_dim such rows.
        let cols = info.dimensions[0] as usize;
        let rows = info.dimensions.get(1).copied().unwrap_or(1) as usize;
        if row_idx >= rows {
            return None;
        }

        let block_size = qtype.block_size();
        let block_bytes = qtype.block_bytes();
        let blocks_per_row = (cols + block_size - 1) / block_size;
        let row_bytes = blocks_per_row * block_bytes;

        let tensor_start = (self.data_offset + info.offset) as usize;
        let row_start = tensor_start + row_idx * row_bytes;
        let row_end = row_start + row_bytes;
        // Bounds-check against actual mmap length to prevent OOB panic
        // on truncated or crafted GGUF files.
        if row_end > self.mmap.len() {
            eprintln!(
                "Leafcutter: row {} of '{}' extends past mmap boundary (end={}, mmap_len={})",
                row_idx, name, row_end, self.mmap.len()
            );
            return None;
        }
        let raw = &self.mmap[row_start..row_end];

        let mut out = vec![0.0f32; cols];
        match qtype {
            QuantType::F32 => {
                for i in 0..cols {
                    let b = [raw[i*4], raw[i*4+1], raw[i*4+2], raw[i*4+3]];
                    out[i] = f32::from_le_bytes(b);
                }
            }
            QuantType::F16 => {
                for i in 0..cols {
                    let b = [raw[i*2], raw[i*2+1]];
                    out[i] = half::f16::from_le_bytes(b).to_f32();
                }
            }
            QuantType::BF16 => {
                for i in 0..cols {
                    let b = [raw[i*2], raw[i*2+1]];
                    out[i] = half::bf16::from_le_bytes(b).to_f32();
                }
            }
            QuantType::Q8_0 => crate::kernels::dequantize_q8_0(raw, &mut out),
            QuantType::Q4_0 => crate::kernels::dequantize_q4_0(raw, &mut out),
            QuantType::Q4_1 => crate::kernels::dequantize_q4_1(raw, &mut out),
            QuantType::Q4_K => crate::kernels::dequantize_q4_k(raw, &mut out),
            QuantType::Q5_K => crate::kernels::dequantize_q5_k(raw, &mut out),
            QuantType::Q6_K => crate::kernels::dequantize_q6_k(raw, &mut out),
            QuantType::Q8_K => crate::kernels::dequantize_q8_k(raw, &mut out),
            QuantType::IQ4_NL => crate::kernels::dequantize_iq4_nl(raw, &mut out),
            QuantType::IQ4_XS => crate::kernels::dequantize_iq4_xs(raw, &mut out),
            _ => {
                // Unsupported quant type — log once and return None rather than
                // crashing the whole run.  Callers propagate this as an error.
                eprintln!(
                    "Leafcutter: unsupported quant type {:?} for tensor '{}'; skipping",
                    qtype, name
                );
                return None;
            }
        }
        Some(out)
    }

    /// Read a single row from a 2D tensor into a pre-allocated buffer.
    /// Avoids allocating a new Vec on every call — use with a thread-local buffer
    /// in hot loops like lm_head projection.
    pub fn get_tensor_row_f32_into(&self, name: &str, row_idx: usize, out: &mut [f32]) -> Option<()> {
        let info = self.get_tensor_info(name)?;
        let qtype = QuantType::from_u32(info.typ)?;
        let cols = info.dimensions[0] as usize;
        let rows = info.dimensions.get(1).copied().unwrap_or(1) as usize;
        if row_idx >= rows || out.len() < cols {
            return None;
        }

        let block_size = qtype.block_size();
        let block_bytes = qtype.block_bytes();
        let blocks_per_row = (cols + block_size - 1) / block_size;
        let row_bytes = blocks_per_row * block_bytes;

        let tensor_start = (self.data_offset + info.offset) as usize;
        let row_start = tensor_start + row_idx * row_bytes;
        let raw = &self.mmap[row_start..row_start + row_bytes];

        match qtype {
            QuantType::F32 => {
                for i in 0..cols {
                    let b = [raw[i*4], raw[i*4+1], raw[i*4+2], raw[i*4+3]];
                    out[i] = f32::from_le_bytes(b);
                }
            }
            QuantType::F16 => {
                for i in 0..cols {
                    let b = [raw[i*2], raw[i*2+1]];
                    out[i] = half::f16::from_le_bytes(b).to_f32();
                }
            }
            QuantType::BF16 => {
                for i in 0..cols {
                    let b = [raw[i*2], raw[i*2+1]];
                    out[i] = half::bf16::from_le_bytes(b).to_f32();
                }
            }
            QuantType::Q8_0 => crate::kernels::dequantize_q8_0(raw, &mut out[..cols]),
            QuantType::Q4_0 => crate::kernels::dequantize_q4_0(raw, &mut out[..cols]),
            QuantType::Q4_1 => crate::kernels::dequantize_q4_1(raw, &mut out[..cols]),
            QuantType::Q4_K => crate::kernels::dequantize_q4_k(raw, &mut out[..cols]),
            QuantType::Q5_K => crate::kernels::dequantize_q5_k(raw, &mut out[..cols]),
            QuantType::Q6_K => crate::kernels::dequantize_q6_k(raw, &mut out[..cols]),
            QuantType::Q8_K => crate::kernels::dequantize_q8_k(raw, &mut out[..cols]),
            QuantType::IQ4_NL => crate::kernels::dequantize_iq4_nl(raw, &mut out[..cols]),
            QuantType::IQ4_XS => crate::kernels::dequantize_iq4_xs(raw, &mut out[..cols]),
            _ => {
                eprintln!(
                    "Leafcutter: unsupported quant type {:?} for tensor '{}'; skipping",
                    qtype, name
                );
                return None;
            }
        }
        Some(())
    }

    /// Drop all mmap pages from the OS page cache.
    /// Call after processing a layer to prevent RSS from growing
    /// as the entire file gets paged in.
    #[cfg(target_os = "linux")]
    pub fn drop_pages_from_cache(&self) {
        let ptr = self.mmap.as_ptr();
        let len = self.mmap.len();
        unsafe {
            // MADV_DONTNEED: Linux frees pages immediately;
            // they will be re-faulted from disk on next access.
            libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_DONTNEED);
        }
    }

    /// Drop a sub-range of mmap pages from the OS page cache.
    /// `start_abs` is the absolute byte offset into the mmap (data_offset +
    /// tensor offset).  Used to release each tensor's file pages as soon as it
    /// has been parsed into an owned quantized block cache, so the model is
    /// never double-resident (raw mmap pages + cache copies) at once.
    #[cfg(target_os = "linux")]
    pub fn drop_pages_in_range(&self, start_abs: u64, len: usize) {
        let ptr = self.mmap.as_ptr() as *mut libc::c_void;
        // madvise requires a page-aligned address; round start down and
        // length up to cover the same pages.
        let page = 4096u64;
        let start_aligned = (start_abs / page) * page;
        let end = start_abs + len as u64;
        let end_aligned = ((end + page - 1) / page) * page;
        let len_aligned = (end_aligned - start_aligned) as usize;
        unsafe {
            libc::madvise(
                ptr.add(start_aligned as usize) as *mut libc::c_void,
                len_aligned,
                libc::MADV_DONTNEED,
            );
        }
    }

    #[cfg(target_os = "macos")]
    pub fn drop_pages_in_range(&self, start_abs: u64, len: usize) {
        let ptr = self.mmap.as_ptr() as *mut libc::c_void;
        unsafe {
            libc::madvise(
                ptr.add(start_abs as usize) as *mut libc::c_void,
                len,
                libc::MADV_FREE,
            );
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn drop_pages_in_range(&self, _start_abs: u64, _len: usize) {
        // No-op on unsupported platforms
    }

    #[cfg(target_os = "macos")]
    pub fn drop_pages_from_cache(&self) {
        let ptr = self.mmap.as_ptr();
        let len = self.mmap.len();
        unsafe {
            // MADV_FREE: macOS marks pages as eligible for reuse immediately,
            // but keeps their content until the kernel needs the memory.
            libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_FREE);
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub fn drop_pages_from_cache(&self) {
        // No-op on unsupported platforms
    }

    pub fn get_metadata_int(&self, key: &str) -> Option<i64> {
        match self.metadata.get(key)? {
            GGUFValue::U8(v) => Some(*v as i64),
            GGUFValue::I8(v) => Some(*v as i64),
            GGUFValue::U16(v) => Some(*v as i64),
            GGUFValue::I16(v) => Some(*v as i64),
            GGUFValue::U32(v) => Some(*v as i64),
            GGUFValue::I32(v) => Some(*v as i64),
            GGUFValue::U64(v) => Some(*v as i64),
            GGUFValue::I64(v) => Some(*v as i64),
            GGUFValue::F32(v) => Some(*v as i64),
            GGUFValue::F64(v) => Some(*v as i64),
            GGUFValue::Array(arr) if arr.len() == 1 => {
                match &arr[0] {
                    GGUFValue::U32(v) => Some(*v as i64),
                    GGUFValue::I32(v) => Some(*v as i64),
                    GGUFValue::U64(v) => Some(*v as i64),
                    GGUFValue::I64(v) => Some(*v as i64),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub fn get_metadata_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            GGUFValue::F32(v) => Some(*v),
            GGUFValue::F64(v) => Some(*v as f32),
            GGUFValue::U8(v) => Some(*v as f32),
            GGUFValue::I8(v) => Some(*v as f32),
            GGUFValue::U16(v) => Some(*v as f32),
            GGUFValue::I16(v) => Some(*v as f32),
            GGUFValue::U32(v) => Some(*v as f32),
            GGUFValue::I32(v) => Some(*v as f32),
            GGUFValue::U64(v) => Some(*v as f32),
            GGUFValue::I64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Raw pointer to the mmap'd file data.
    pub fn mmap_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }

    /// Total length of the mmap'd file in bytes.
    pub fn mmap_len(&self) -> usize {
        self.mmap.len()
    }
}

struct GGUFReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> GGUFReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_header(&mut self) -> Result<GGUFHeader, GGUError> {
        Ok(GGUFHeader {
            magic: self.read_u32()?,
            version: self.read_u32()?,
            tensor_count: self.read_u64()?,
            metadata_count: self.read_u64()?,
        })
    }

    fn read_metadata_kv(&mut self) -> Result<(String, GGUFValue), GGUError> {
        let key = self.read_string()?;
        let typ = self.read_u32()?;
        let value = self.read_value(typ)?;
        Ok((key, value))
    }

    fn read_value(&mut self, typ: u32) -> Result<GGUFValue, GGUError> {
        match typ {
            0 => Ok(GGUFValue::U8(self.read_u8()?)),
            1 => Ok(GGUFValue::I8(self.read_i8()?)),
            2 => Ok(GGUFValue::U16(self.read_u16()?)),
            3 => Ok(GGUFValue::I16(self.read_i16()?)),
            4 => Ok(GGUFValue::U32(self.read_u32()?)),
            5 => Ok(GGUFValue::I32(self.read_i32()?)),
            6 => Ok(GGUFValue::F32(self.read_f32()?)),
            7 => Ok(GGUFValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GGUFValue::String(self.read_string()?)),
            9 => {
                let arr_type = self.read_u32()?;
                let len = self.read_u64()?;
                let mut arr = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    arr.push(self.read_value(arr_type)?);
                }
                Ok(GGUFValue::Array(arr))
            }
            10 => Ok(GGUFValue::U64(self.read_u64()?)),
            11 => Ok(GGUFValue::I64(self.read_i64()?)),
            12 => Ok(GGUFValue::F64(self.read_f64()?)),
            _ => Err(GGUError::InvalidTensorType(typ)),
        }
    }

    fn read_tensor_info(&mut self) -> Result<GGUFTensor, GGUError> {
        let name = self.read_string()?;
        let n_dims = self.read_u32()? as usize;
        let mut dimensions = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dimensions.push(self.read_u64()?);
        }
        let typ = self.read_u32()?;
        let offset = self.read_u64()?;
        Ok(GGUFTensor {
            name,
            dimensions,
            typ,
            offset,
        })
    }

    fn read_u8(&mut self) -> Result<u8, GGUError> {
        if self.pos + 1 > self.data.len() {
            return Err(GGUError::TruncatedData);
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8, GGUError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GGUError> {
        if self.pos + 2 > self.data.len() {
            return Err(GGUError::TruncatedData);
        }
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16, GGUError> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32, GGUError> {
        if self.pos + 4 > self.data.len() {
            return Err(GGUError::TruncatedData);
        }
        let v = u32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> Result<i32, GGUError> {
        Ok(self.read_u32()? as i32)
    }

    fn read_f32(&mut self) -> Result<f32, GGUError> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_u64(&mut self) -> Result<u64, GGUError> {
        if self.pos + 8 > self.data.len() {
            return Err(GGUError::TruncatedData);
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[self.pos..self.pos + 8]);
        let v = u64::from_le_bytes(bytes);
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> Result<i64, GGUError> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f64(&mut self) -> Result<f64, GGUError> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self) -> Result<String, GGUError> {
        let len = self.read_u64()? as usize;
        if self.pos + len > self.data.len() {
            return Err(GGUError::TruncatedData);
        }
        let s = String::from_utf8_lossy(&self.data[self.pos..self.pos + len]).to_string();
        self.pos += len;
        Ok(s)
    }
}

pub fn calculate_tensor_size(dims: &[u64], typ: u32) -> usize {
    let count: u64 = dims.iter().product();
    match QuantType::from_u32(typ) {
        Some(qt) => qt.tensor_size(count as usize),
        None => count as usize,
    }
}

impl GGUFile {
    /// Build a quantization-type summary for all tensors in the file.
    pub fn quant_summary(&self) -> QuantSummary {
        let mut summary = QuantSummary::default();
        summary.total_tensors = self.tensors.len();
        for t in &self.tensors {
            if let Some(qt) = QuantType::from_u32(t.typ) {
                *summary.types.entry(qt).or_insert(0) += 1;
                if !qt.is_supported() {
                    if !summary.unsupported.contains(&qt) {
                        summary.unsupported.push(qt);
                    }
                }
            }
        }
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_tensor_size() {
        assert_eq!(calculate_tensor_size(&[256], 12), 144);  // Q4_K
        assert_eq!(calculate_tensor_size(&[256], 13), 176);  // Q5_K
        assert_eq!(calculate_tensor_size(&[256], 14), 210);  // Q6_K
        assert_eq!(calculate_tensor_size(&[256], 15), 292);  // Q8_K
        assert_eq!(calculate_tensor_size(&[256], 2), 144);   // Q4_0 (32*18/4 = 144 for 256)
        assert_eq!(calculate_tensor_size(&[256], 8), 272);   // Q8_0 (8*34 = 272 for 256)
    }

    #[test]
    fn test_load_real_gguf() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let file = GGUFile::open(path).expect("Failed to open GGUF");
        println!("Loaded GGUF with {} tensors", file.tensors.len());
        assert!(!file.tensors.is_empty());
    }
}


#[cfg(test)]
mod eos_tests {
    use super::*;

    #[test]
    fn debug_eos_token() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let file = GGUFile::open(path).expect("Failed to open GGUF");
        
        // Check EOS token
        if let Some(eos) = file.get_metadata_int("tokenizer.ggml.eos_token_id") {
            println!("EOS token ID: {}", eos);
        }
        if let Some(bos) = file.get_metadata_int("tokenizer.ggml.bos_token_id") {
            println!("BOS token ID: {}", bos);
        }
        if let Some(pad) = file.get_metadata_int("tokenizer.ggml.padding_token_id") {
            println!("PAD token ID: {}", pad);
        }
        
        // Check vocab size
        if let Some(vs) = file.get_metadata_int("qwen2.vocab_size") {
            println!("qwen2.vocab_size: {}", vs);
        }
        if let Some(GGUFValue::Array(arr)) = file.metadata.get("tokenizer.ggml.tokens") {
            println!("tokenizer.ggml.tokens len: {}", arr.len());
        }
        
        // List added tokens
        if let Some(GGUFValue::Array(arr)) = file.metadata.get("tokenizer.ggml.added_tokens") {
            println!("Added tokens count: {}", arr.len());
        }
    }
}

#[cfg(test)]
mod weight_shape_tests {
    use super::*;

    #[test]
    fn debug_attention_weight_shapes() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let file = GGUFile::open(path).expect("Failed to open GGUF");
        
        for name in ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight"] {
            if let Some(info) = file.get_tensor_info(name) {
                println!("{}: GGUF dims={:?} -> reversed={:?}", name, info.dimensions, info.dimensions.iter().rev().collect::<Vec<_>>());
            }
        }
    }
}

#[cfg(test)]
mod ffn_tests {
    
    use crate::model::loader::GGUFModel;

    #[test]
    fn debug_ffn_weight_shapes() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let model = GGUFModel::load(path).unwrap();
        let layer0 = model.load_layer(0).unwrap();
        for (name, tensor) in &layer0 {
            if name.contains("mlp") {
                println!("{}: shape={:?}", name, tensor.shape);
            }
        }
    }
}

#[cfg(test)]
mod all_shapes {
    use crate::model::loader::GGUFModel;

    #[test]
    fn debug_all_layer_shapes() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let model = GGUFModel::load(path).unwrap();
        let layer0 = model.load_layer(0).unwrap();
        for (name, tensor) in &layer0 {
            println!("{}: shape={:?}", name, tensor.shape);
        }
    }
}

#[cfg(test)]
mod token_lookup {
    use super::*;

    #[test]
    fn debug_token_151935() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let file = GGUFile::open(path).unwrap();
        
        if let Some(GGUFValue::Array(arr)) = file.metadata.get("tokenizer.ggml.tokens") {
            if let Some(GGUFValue::String(tok)) = arr.get(151935) {
                println!("Token 151935: '{}'", tok);
            }
            if let Some(GGUFValue::String(tok)) = arr.get(151643) {
                println!("Token 151643 (BOS): '{}'", tok);
            }
            if let Some(GGUFValue::String(tok)) = arr.get(151645) {
                println!("Token 151645 (EOS): '{}'", tok);
            }
            if let Some(GGUFValue::String(tok)) = arr.get(151644) {
                println!("Token 151644 (IM_START): '{}'", tok);
            }
            // Check a few around 151935
            for i in [151930, 151931, 151932, 151933, 151934, 151935] {
                if let Some(GGUFValue::String(tok)) = arr.get(i) {
                    println!("Token {}: '{}'", i, tok);
                }
            }
        }
    }
}

#[cfg(test)]
mod ffn_gguf_dims {
    use super::*;

    #[test]
    fn debug_ffn_gguf_dims() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let file = GGUFile::open(path).unwrap();
        for name in ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"] {
            if let Some(info) = file.get_tensor_info(name) {
                println!("{}: GGUF dims={:?}", name, info.dimensions);
            }
        }
    }
}

#[cfg(test)]
mod special_shapes {
    use super::*;

    #[test]
    fn debug_special_gguf_dims() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let file = GGUFile::open(path).unwrap();
        for name in ["output.weight", "output_norm.weight", "token_embd.weight"] {
            if let Some(info) = file.get_tensor_info(name) {
                println!("{}: GGUF dims={:?}", name, info.dimensions);
            }
        }
    }
}

#[cfg(test)]
mod layer1_types {
    use super::*;

    #[test]
    fn debug_layer1_tensor_types() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let file = GGUFile::open(path).unwrap();
        for name in ["blk.1.attn_q.weight", "blk.1.attn_k.weight", "blk.1.attn_v.weight", 
                     "blk.1.ffn_gate.weight", "blk.1.ffn_up.weight", "blk.1.ffn_down.weight"] {
            if let Some(info) = file.get_tensor_info(name) {
                println!("{}: dims={:?} type={}", name, info.dimensions, info.typ);
            }
        }
    }
}

#[cfg(test)]
mod offsets {
    use super::*;

    #[test]
    fn debug_tensor_offsets() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let file = GGUFile::open(path).unwrap();
        println!("data_offset = {}", file.data_offset);
        for name in ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                     "blk.1.ffn_gate.weight", "blk.1.ffn_up.weight", "blk.1.ffn_down.weight"] {
            if let Some(t) = file.tensors.iter().find(|t| t.name == name) {
                let size = calculate_tensor_size(&t.dimensions, t.typ);
                println!("{}: offset={} size={} end={}", name, t.offset, size, t.offset + size as u64);
            }
        }
    }
}

#[cfg(test)]
mod first_offset {
    use super::*;

    #[test]
    fn debug_first_tensor_offset() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let file = GGUFile::open(path).unwrap();
        println!("data_offset = {}", file.data_offset);
        for i in 0..10.min(file.tensors.len()) {
            let t = &file.tensors[i];
            println!("{}: offset={}", t.name, t.offset);
        }
    }
}

#[cfg(test)]
mod embed_type {
    use super::*;

    #[test]
    fn debug_token_embd() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let file = GGUFile::open(path).unwrap();
        if let Some(t) = file.tensors.iter().find(|t| t.name == "token_embd.weight") {
            let size = calculate_tensor_size(&t.dimensions, t.typ);
            println!("token_embd.weight: dims={:?} type={} size={} offset={}", t.dimensions, t.typ, size, t.offset);
        }
        // Print first 20 tensor names and offsets
        for i in 0..20.min(file.tensors.len()) {
            let t = &file.tensors[i];
            let size = calculate_tensor_size(&t.dimensions, t.typ);
            println!("{}: offset={} size={}", t.name, t.offset, size);
        }
    }
}

#[cfg(test)]
mod alignment {
    use super::*;

    #[test]
    fn debug_alignment() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let file = GGUFile::open(path).unwrap();
        println!("data_offset = {}", file.data_offset);
        println!("alignment from metadata: {:?}", file.metadata.get("general.alignment"));
    }
}

#[cfg(test)]
mod header_counts {
    use super::*;

    #[test]
    fn debug_header_counts() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() { return; }
        let file = GGUFile::open(path).unwrap();
        println!("tensor_count = {}", file.header.tensor_count);
        println!("metadata_count = {}", file.header.metadata_count);
        println!("actual tensors parsed = {}", file.tensors.len());
        println!("actual metadata keys = {}", file.metadata.len());
    }
}

#[cfg(test)]
mod fingerprint_tests {
    use super::*;

    fn sample_file() -> GGUFile {
        let tmp = std::env::temp_dir().join("leafcutter_fp_test.bin");
        std::fs::write(&tmp, vec![0u8; 4096]).unwrap();
        GGUFile {
            header: GGUFHeader {
                magic: GGUF_MAGIC,
                version: 3,
                tensor_count: 2,
                metadata_count: 1,
            },
            metadata: std::collections::HashMap::from([
                ("general.architecture".to_string(), GGUFValue::String("qwen2".to_string())),
                ("tokenizer.ggml.model".to_string(), GGUFValue::String("gpt2".to_string())),
            ]),
            tensors: vec![
                GGUFTensor { name: "token_embd.weight".into(), dimensions: vec![4096, 2048], typ: 3, offset: 0 },
                GGUFTensor { name: "blk.0.attn_q.weight".into(), dimensions: vec![4096, 4096], typ: 3, offset: 100 },
            ],
            data_offset: 0,
            mmap: unsafe { memmap2::Mmap::map(&std::fs::File::open(&tmp).unwrap()).unwrap() },
        }
    }

    #[test]
    fn fingerprint_is_stable() {
        let a = sample_file();
        let b = sample_file();
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_changes_with_metadata() {
        let mut a = sample_file();
        a.metadata.insert("general.name".into(), GGUFValue::String("v2".into()));
        let b = sample_file();
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_changes_with_tensor_layout() {
        let mut a = sample_file();
        a.tensors[1].dimensions = vec![8192, 8192];
        let b = sample_file();
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_ignores_metadata_order() {
        let mut a = sample_file();
        let mut b = sample_file();
        std::mem::swap(
            a.metadata.get_mut("general.architecture").unwrap(),
            b.metadata.get_mut("tokenizer.ggml.model").unwrap(),
        );
        std::mem::swap(
            a.metadata.get_mut("tokenizer.ggml.model").unwrap(),
            b.metadata.get_mut("general.architecture").unwrap(),
        );
        assert_eq!(a.fingerprint(), b.fingerprint());
    }
}
