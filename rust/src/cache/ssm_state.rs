//! SSM State Cache for Mamba-style state space models
//!
//! Unlike transformers which use KV cache, SSM layers maintain a persistent
//! hidden state `h` per channel that must be carried across forward passes.
//! This cache stores `h` for each SSM layer, plus the causal conv1d history.

use std::collections::HashMap;

pub struct SSMStateCache {
    /// Per-layer SSM state vectors: layer_idx -> Vec<h_per_channel>
    states: HashMap<usize, Vec<f32>>,
    /// Per-layer conv1d cached inputs: layer_idx -> Vec<cached_inputs>
    /// Stores the last (kernel_size - 1) inputs for each channel,
    /// flattened as [channel0_oldest, ..., channel0_newest, channel1_oldest, ...]
    conv_states: HashMap<usize, Vec<f32>>,
}

impl SSMStateCache {
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

    /// Get the SSM state for a layer, returning zeros if not yet initialized.
    pub fn get(&self, layer_idx: usize, inner_size: usize) -> Vec<f32> {
        self.states.get(&layer_idx)
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; inner_size])
    }

    /// Set the SSM state for a layer.
    pub fn set(&mut self, layer_idx: usize, state: Vec<f32>) {
        self.states.insert(layer_idx, state);
    }

    /// Get the conv1d cached inputs for a layer.
    /// Returns empty vec if not initialized.
    pub fn get_conv(&self, layer_idx: usize) -> Vec<f32> {
        self.conv_states.get(&layer_idx)
            .cloned()
            .unwrap_or_default()
    }

    /// Set the conv1d cached inputs for a layer.
    pub fn set_conv(&mut self, layer_idx: usize, state: Vec<f32>) {
        self.conv_states.insert(layer_idx, state);
    }

    /// Report memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let state_bytes: usize = self.states.values().map(|v| v.len() * 4).sum();
        let conv_bytes: usize = self.conv_states.values().map(|v| v.len() * 4).sum();
        state_bytes + conv_bytes
    }
}
