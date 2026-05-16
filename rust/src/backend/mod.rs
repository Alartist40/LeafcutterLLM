//! Hardware abstraction layer — pluggable compute backends
//!
//! LeafcutterLLM is designed to run on any hardware: ARM SBCs, x86_64
//! desktops, Apple Silicon, and (future) GPUs.  The `Backend` trait
//! abstracts all compute operations so the engine logic never changes.
//!
//! Usage:
//!   Tensor::set_global_backend(&CPU_BACKEND); // default
//!   Tensor::set_global_backend(&Q8_CPU_BACKEND); // quantized

pub mod cpu;
pub mod wgpu;

/// Abstract compute backend.
///
/// All methods take slices and return owned `Vec<f32>` results.
/// This keeps the backend decoupled from `Tensor`'s shape logic.
pub trait Backend: Send + Sync {
    /// Matrix multiplication: C = A × B
    /// A: [m, k], B: [k, n], result: [m * n]
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32>;

    /// Element-wise addition: out = a + b
    fn vec_add(&self, a: &[f32], b: &[f32]) -> Vec<f32>;

    /// Element-wise multiply: out = a * b
    fn vec_mul(&self, a: &[f32], b: &[f32]) -> Vec<f32>;

    /// Scale a vector: out = a * scale
    fn vec_scale(&self, a: &[f32], scale: f32) -> Vec<f32>;

    /// Scale then multiply: out = a * scale * b
    fn vec_scale_mul(&self, a: &[f32], scale: f32, b: &[f32]) -> Vec<f32>;

    /// RMSNorm over rows: out = x * rsqrt(mean(x^2) + eps) * weight
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32>;

    /// SiLU activation: x * sigmoid(x)
    fn silu(&self, x: &[f32]) -> Vec<f32>;

    /// Softmax over last dimension
    fn softmax(&self, x: &[f32], hidden_size: usize) -> Vec<f32>;

    /// Sum of squares
    fn sum_sq(&self, x: &[f32]) -> f32;
}

use std::sync::OnceLock;

static GLOBAL_BACKEND: OnceLock<&'static dyn Backend> = OnceLock::new();

/// Get the current global backend.  Defaults to `CpuBackend`.
pub fn default_backend() -> &'static dyn Backend {
    *GLOBAL_BACKEND.get_or_init(|| &cpu::CPU_BACKEND)
}

/// Set the global backend for all new Tensors.
pub fn set_global_backend(backend: &'static dyn Backend) {
    let _ = GLOBAL_BACKEND.set(backend);
}
