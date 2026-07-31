//! Layer shard format — disk-offloaded per-layer weights
//!
//! A "shard" is a single layer's dequantized f32 weights stored in a
//! lightweight mmap-friendly binary format.  During inference only one
//! shard is resident in RAM at a time.

pub mod format;
pub mod loader;
pub mod writer;
pub mod lfru_cache;

pub use format::{ShardHeader, ShardTensorMeta, QuantFormat};
pub use loader::{ShardLoader, Manifest, LayerCache, CachePolicy, ShardCache};
pub use lfru_cache::{LfruCache, CacheStats};
pub use writer::split_gguf_model;
