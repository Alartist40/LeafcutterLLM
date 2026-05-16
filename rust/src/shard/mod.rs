//! Layer shard format — disk-offloaded per-layer weights
//!
//! A "shard" is a single layer's dequantized f32 weights stored in a
//! lightweight mmap-friendly binary format.  During inference only one
//! shard is resident in RAM at a time.

pub mod format;
pub mod loader;
pub mod writer;

pub use format::{ShardHeader, ShardTensorMeta, QuantFormat};
pub use loader::{ShardLoader, Manifest};
pub use writer::{ShardWriter, split_gguf_model};
