//! GGUF v3 parser with K-quant support
//!
//! Implements zero-copy mmap-based loading of GGUF files.
//! Supports Q4_0, Q8_0, Q4_K, Q5_K, Q6_K tensor types.

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

use super::quant::{QuantType, QuantSummary};

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
}

#[derive(Debug, Clone)]
pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
}

#[derive(Debug, Clone)]
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

impl GGUFile {
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
        Some(&self.mmap[start as usize..end as usize])
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<&GGUFTensor> {
        self.tensors.iter().find(|t| t.name == name)
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
    use super::*;
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
