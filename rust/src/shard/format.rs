//! Binary shard file format
//!
//! Layout on disk (little-endian, native f32):
//! ```text
//! [Header] 32 bytes
//!   magic:            [u8; 8]  = "LEAFSHDR"
//!   version:          u32      = 1
//!   tensor_count:     u32
//!   data_start:       u64      // file offset where data section begins
//!   reserved:         [u8; 8]
//!
//! [Tensor Metadata] × tensor_count
//!   name_len:         u32
//!   name:             [u8; name_len]
//!   rank:             u32
//!   dims:             [u64; rank]
//!   data_offset:      u64      // absolute file offset
//!   data_size:        u64      // bytes (= elements × 4)
//!
//! [Padding] to align data_start to 4096 bytes
//!
//! [Data Section]
//!   raw f32 bytes for each tensor, starting at data_offset
//! ```

use std::io::{Read, Write};

pub const SHARD_MAGIC: &[u8; 8] = b"LEAFSHDR";
pub const SHARD_VERSION: u32 = 1;
pub const DATA_ALIGN: u64 = 4096;

/// Quantization format stored in the shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// Raw f32 values (no quantization)
    F32 = 0,
    /// Q8_0 per-block quantization (32 int8 + f16 scale per block)
    Q8_0 = 1,
}

impl QuantFormat {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(QuantFormat::F32),
            1 => Some(QuantFormat::Q8_0),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShardHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub tensor_count: u32,
    pub data_start: u64,
}

impl ShardHeader {
    pub const SIZE: usize = 32;

    pub fn new(tensor_count: u32, data_start: u64) -> Self {
        Self {
            magic: *SHARD_MAGIC,
            version: SHARD_VERSION,
            tensor_count,
            data_start,
        }
    }

    pub fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.magic)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.tensor_count.to_le_bytes())?;
        writer.write_all(&self.data_start.to_le_bytes())?;
        writer.write_all(&[0u8; 8])?; // reserved
        Ok(())
    }

    pub fn read<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;

        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let tensor_count = u32::from_le_bytes(buf4);

        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let data_start = u64::from_le_bytes(buf8);

        let mut reserved = [0u8; 8];
        reader.read_exact(&mut reserved)?;

        Ok(Self {
            magic,
            version,
            tensor_count,
            data_start,
        })
    }

    pub fn is_valid(&self) -> bool {
        &self.magic == SHARD_MAGIC && self.version == SHARD_VERSION
    }
}

#[derive(Debug, Clone)]
pub struct ShardTensorMeta {
    pub name: String,
    pub rank: u32,
    pub dims: Vec<u64>,
    pub data_offset: u64,
    pub data_size: u64,
}

impl ShardTensorMeta {
    pub fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let name_bytes = self.name.as_bytes();
        writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(name_bytes)?;
        writer.write_all(&self.rank.to_le_bytes())?;
        for dim in &self.dims {
            writer.write_all(&dim.to_le_bytes())?;
        }
        writer.write_all(&self.data_offset.to_le_bytes())?;
        writer.write_all(&self.data_size.to_le_bytes())?;
        Ok(())
    }

    pub fn read<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let name_len = u32::from_le_bytes(buf4) as usize;

        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;

        reader.read_exact(&mut buf4)?;
        let rank = u32::from_le_bytes(buf4);

        let mut dims = Vec::with_capacity(rank as usize);
        let mut buf8 = [0u8; 8];
        for _ in 0..rank {
            reader.read_exact(&mut buf8)?;
            dims.push(u64::from_le_bytes(buf8));
        }

        reader.read_exact(&mut buf8)?;
        let data_offset = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let data_size = u64::from_le_bytes(buf8);

        Ok(Self {
            name,
            rank,
            dims,
            data_offset,
            data_size,
        })
    }

    /// Number of f32 elements
    pub fn element_count(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
}

/// Align an offset up to the nearest multiple of `align`
pub fn align_up(offset: u64, align: u64) -> u64 {
    (offset + align - 1) / align * align
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = ShardHeader::new(5, 4096);
        let mut buf = Vec::new();
        header.write(&mut buf).unwrap();
        assert_eq!(buf.len(), ShardHeader::SIZE);

        let mut cursor = std::io::Cursor::new(&buf);
        let parsed = ShardHeader::read(&mut cursor).unwrap();
        assert!(parsed.is_valid());
        assert_eq!(parsed.tensor_count, 5);
        assert_eq!(parsed.data_start, 4096);
    }

    #[test]
    fn test_tensor_meta_roundtrip() {
        let meta = ShardTensorMeta {
            name: "mlp.gate_proj.weight".to_string(),
            rank: 2,
            dims: vec![11008, 2048],
            data_offset: 8192,
            data_size: 11008 * 2048 * 4,
        };
        let mut buf = Vec::new();
        meta.write(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let parsed = ShardTensorMeta::read(&mut cursor).unwrap();
        assert_eq!(parsed.name, "mlp.gate_proj.weight");
        assert_eq!(parsed.rank, 2);
        assert_eq!(parsed.dims, vec![11008, 2048]);
        assert_eq!(parsed.data_offset, 8192);
        assert_eq!(parsed.data_size, 11008 * 2048 * 4);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
    }
}
