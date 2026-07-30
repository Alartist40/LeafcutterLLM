//! DeltaNet State Cache — matrix states for Gated DeltaNet layers
//!
//! Unlike Mamba SSM which uses a vector state per channel, DeltaNet maintains
//! a matrix state S_h = [head_v_dim, head_k_dim] per head.  The delta rule is:
//!   S_t = decay_t * S_{t-1} + beta_t * (v_t ⊗ k_t)
//! where ⊗ is the outer product.

use std::collections::HashMap;

pub struct DeltaNetStateCache {
    /// Per-layer matrix states: layer_idx -> flat Vec of [num_v_heads, head_v_dim, head_k_dim]
    states: HashMap<usize, Vec<f32>>,
    /// Per-layer conv1d cached inputs: layer_idx -> Vec<cached_inputs>
    conv_states: HashMap<usize, Vec<f32>>,
}

impl DeltaNetStateCache {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            conv_states: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.conv_states.clear();
    }

    /// Get the matrix state for a layer, returning None if not yet initialized.
    pub fn get(&self, layer_idx: usize) -> Option<&Vec<f32>> {
        self.states.get(&layer_idx)
    }

    /// Get a mutable reference to the matrix state for a layer.
    pub fn get_mut(&mut self, layer_idx: usize) -> Option<&mut Vec<f32>> {
        self.states.get_mut(&layer_idx)
    }

    /// Initialize the matrix state for a layer with zeros.
    pub fn init_layer(&mut self, layer_idx: usize, num_v_heads: usize, head_v_dim: usize, head_k_dim: usize) {
        let size = num_v_heads * head_v_dim * head_k_dim;
        self.states.insert(layer_idx, vec![0.0f32; size]);
    }

    /// Get the conv1d cached inputs for a layer.
    pub fn get_conv(&self, layer_idx: usize) -> Vec<f32> {
        self.conv_states.get(&layer_idx).cloned().unwrap_or_default()
    }

    /// Set the conv1d cached inputs for a layer.
    pub fn set_conv(&mut self, layer_idx: usize, state: Vec<f32>) {
        self.conv_states.insert(layer_idx, state);
    }

    /// Get a mutable reference to the conv buffer for a layer, initializing if needed.
    /// The conv buffer has shape [conv_dim, conv_k] stored flat as [c * conv_k + k] where
    /// k=0 is the oldest input and k=conv_k-1 is the newest (current) input.
    pub fn get_conv_buf_mut(&mut self, layer_idx: usize, conv_dim: usize, conv_k: usize) -> &mut Vec<f32> {
        let entry = self.conv_states.entry(layer_idx).or_insert_with(|| {
            vec![0.0f32; conv_dim * conv_k]
        });
        entry
    }

    /// Report memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let state_bytes: usize = self.states.values().map(|v| v.len() * 4).sum();
        let conv_bytes: usize = self.conv_states.values().map(|v| v.len() * 4).sum();
        state_bytes + conv_bytes
    }
}
