//! Layer streaming loader for GGUF models
//!
//! Only one layer's weights are resident in RAM at any time.

use super::arch::{CapabilityReport, ModelArchitecture};
use super::gguf::{GGUFile, GGUError, calculate_tensor_size};
use super::quant::QuantType;
use super::tensor::Tensor;
use crate::kernels;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            intermediate_size: 11008,
            max_seq_len: 4096,
            vocab_size: 32000,
            rope_theta: 10000.0,
        }
    }
}

pub struct GGUFModel {
    pub file: GGUFile,
    pub config: ModelConfig,
    pub architecture: ModelArchitecture,
}

impl GGUFModel {
    pub fn load(path: &str) -> Result<Self, GGUError> {
        let file = GGUFile::open(path)?;
        let architecture = ModelArchitecture::detect(&file);
        let config = Self::extract_config(&file, architecture);
        Ok(Self { file, config, architecture })
    }

    /// Generate a pre-flight capability report without loading any weights.
    pub fn capability_report(&self) -> CapabilityReport {
        let quant_summary = self.file.quant_summary();
        let arch_supported = self.architecture.is_supported();
        let mappings = self.architecture.layer_mappings();

        // Check which required tensors are missing
        let mut missing = Vec::new();
        for layer_idx in 0..self.config.num_hidden_layers.min(4) {
            // Sample first few layers to keep it fast
            let prefix = format!("blk.{}", layer_idx);
            for (gguf_suffix, _engine_name) in mappings.iter() {
                let name = format!("{}.{}", prefix, gguf_suffix);
                if self.file.get_tensor_info(&name).is_none() {
                    missing.push(name);
                }
            }
        }

        // Check for extra / unrecognised tensors
        let known_suffixes: std::collections::HashSet<_> = mappings
            .iter()
            .map(|(s, _)| *s)
            .chain(self.architecture.known_extra_suffixes().iter().copied())
            .collect();

        let mut extra = Vec::new();
        for t in &self.file.tensors {
            if t.name.starts_with("blk.") && t.name.ends_with(".weight") {
                let _suffix = t.name.rsplitn(2, '.').next().unwrap_or("");
                let full_suffix = t.name.splitn(3, '.').nth(2).unwrap_or("");
                if !known_suffixes.contains(full_suffix) {
                    extra.push(t.name.clone());
                }
            }
        }
        extra.sort();
        extra.dedup();

        let can_run = arch_supported
            && quant_summary.is_fully_supported()
            && missing.is_empty();

        CapabilityReport {
            architecture: self.architecture,
            arch_supported,
            quant_summary,
            missing_tensors: missing,
            extra_tensors: extra,
            can_run,
        }
    }

    fn extract_config(file: &GGUFile, arch: ModelArchitecture) -> ModelConfig {
        let mut cfg = ModelConfig::default();
        let prefix = arch.metadata_prefix();

        cfg.hidden_size = Self::get_meta_int(file, &[&format!("{}.embedding_length", prefix), "llama.embedding_length", "qwen2.embedding_length", "qwen35.embedding_length"])
            .map(|v| v as usize).unwrap_or(cfg.hidden_size);
        cfg.num_hidden_layers = Self::get_meta_int(file, &[&format!("{}.block_count", prefix), "llama.block_count", "qwen2.block_count", "qwen35.block_count"])
            .map(|v| v as usize).unwrap_or(cfg.num_hidden_layers);
        cfg.num_attention_heads = Self::get_meta_int(file, &[&format!("{}.attention.head_count", prefix), "llama.attention.head_count", "qwen2.attention.head_count", "qwen35.attention.head_count"])
            .map(|v| v as usize).unwrap_or(cfg.num_attention_heads);
        cfg.num_key_value_heads = Self::get_meta_int(file, &[&format!("{}.attention.head_count_kv", prefix), "llama.attention.head_count_kv", "qwen2.attention.head_count_kv", "qwen35.attention.head_count_kv"])
            .map(|v| v as usize).unwrap_or(cfg.num_key_value_heads);
        cfg.intermediate_size = Self::get_meta_int(file, &[&format!("{}.feed_forward_length", prefix), "llama.feed_forward_length", "qwen2.feed_forward_length", "qwen35.feed_forward_length"])
            .map(|v| v as usize).unwrap_or(cfg.intermediate_size);
        cfg.max_seq_len = Self::get_meta_int(file, &[&format!("{}.context_length", prefix), "llama.context_length", "qwen2.context_length", "qwen35.context_length"])
            .map(|v| v as usize).unwrap_or(cfg.max_seq_len);
        cfg.vocab_size = Self::get_meta_int(file, &[&format!("{}.vocab_size", prefix), "tokenizer.ggml.tokens.length", "tokenizer.ggml.vocab_size"])
            .map(|v| v as usize)
            .or_else(|| {
                file.metadata.get("tokenizer.ggml.tokens")
                    .and_then(|v| if let crate::model::gguf::GGUFValue::Array(arr) = v { Some(arr.len()) } else { None })
            })
            .unwrap_or(cfg.vocab_size);
        cfg.rope_theta = Self::get_meta_int(file, &[&format!("{}.rope.freq_base", prefix), "llama.rope.freq_base", "qwen2.rope.freq_base", "qwen35.rope.freq_base"])
            .map(|v| v as f32).unwrap_or(cfg.rope_theta);

        cfg
    }

    fn get_meta_int(file: &GGUFile, keys: &[&str]) -> Option<i64> {
        for key in keys {
            if let Some(v) = file.get_metadata_int(key) {
                return Some(v);
            }
        }
        None
    }

    /// Load a specific transformer layer's weights
    pub fn load_layer(&self, idx: usize) -> Result<HashMap<String, Tensor>, GGUError> {
        let prefix = format!("blk.{}", idx);
        let mut weights = HashMap::new();

        let mappings = self.architecture.layer_mappings();

        for (gguf_suffix, engine_name) in mappings.iter() {
            let gguf_name = format!("{}.{}", prefix, gguf_suffix);
            if let Some(raw) = self.file.get_tensor_raw(&gguf_name) {
                if let Some(info) = self.file.get_tensor_info(&gguf_name) {
                    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).rev().collect();
                    let mut tensor = Self::dequantize(raw, info.typ, shape)?;
                    // Transpose 2D weights so matmul is A @ B (not A @ B^T)
                    if tensor.shape.len() == 2 {
                        tensor = tensor.transpose();
                    }
                    sanitize_weights(&mut tensor);
                    weights.insert(engine_name.to_string(), tensor);
                }
            }
        }

        Ok(weights)
    }

    /// Load embedding, final norm, and lm_head
    pub fn load_special(&self) -> Result<HashMap<String, Tensor>, GGUError> {
        let mut weights = HashMap::new();
        let mappings = [
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("output.weight", "lm_head.weight"),
        ];

        for (gguf_name, engine_name) in mappings.iter() {
            if let Some(raw) = self.file.get_tensor_raw(gguf_name) {
                if let Some(info) = self.file.get_tensor_info(gguf_name) {
                    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).rev().collect();
                    let mut tensor = Self::dequantize(raw, info.typ, shape)?;
                    // Transpose 2D weights, but NOT the embedding matrix
                    if tensor.shape.len() == 2 && *gguf_name != "token_embd.weight" {
                        tensor = tensor.transpose();
                    }
                    sanitize_weights(&mut tensor);
                    weights.insert(engine_name.to_string(), tensor);
                }
            }
        }

        // If lm_head is missing, use tied embeddings (common in Qwen models)
        if !weights.contains_key("lm_head.weight") {
            if let Some(embed) = weights.get("model.embed_tokens.weight") {
                // Embedding is [vocab, hidden]; lm_head needs [hidden, vocab]
                weights.insert("lm_head.weight".to_string(), embed.transpose());
            }
        }

        Ok(weights)
    }

    fn dequantize(data: &[u8], typ: u32, shape: Vec<usize>) -> Result<Tensor, GGUError> {
        let count: usize = shape.iter().product();
        let mut out = vec![0.0f32; count];

        let qtype = QuantType::from_u32(typ)
            .ok_or(GGUError::InvalidTensorType(typ))?;

        match qtype {
            QuantType::F32 => {
                for i in 0..count {
                    let bytes = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
                    out[i] = f32::from_le_bytes(bytes);
                }
            }
            QuantType::F16 => {
                for i in 0..count {
                    let bytes = [data[i * 2], data[i * 2 + 1]];
                    out[i] = half::f16::from_le_bytes(bytes).to_f32();
                }
            }
            QuantType::Q4_0 => kernels::dequantize_q4_0(data, &mut out),
            QuantType::Q8_0 => kernels::dequantize_q8_0(data, &mut out),
            QuantType::Q4_K => kernels::dequantize_q4_k(data, &mut out),
            QuantType::Q5_K => kernels::dequantize_q5_k(data, &mut out),
            QuantType::Q6_K => kernels::dequantize_q6_k(data, &mut out),
            QuantType::IQ4_NL => kernels::dequantize_iq4_nl(data, &mut out),
            _ => return Err(GGUError::InvalidTensorType(typ)),
        }

        Ok(Tensor::from_vec(out, shape))
    }
}

/// Sanitize dequantized weights by replacing NaN/Inf/outliers with 0.
/// Some GGUF files have corrupted quantization blocks (bad sectors, partial downloads).
/// For Q4_K, normal weights are typically |v| < 10. A threshold of 100 is conservative.
const WEIGHT_SANITY_THRESHOLD: f32 = 100.0;

fn sanitize_weights(tensor: &mut Tensor) {
    for v in &mut tensor.data {
        if v.is_nan() || v.is_infinite() || v.abs() > WEIGHT_SANITY_THRESHOLD {
            *v = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Corruption detector — scans raw tensor blocks for bad scales
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TensorCorruption {
    pub name: String,
    pub quant_type: String,
    pub blocks_total: usize,
    pub blocks_bad: usize,
    pub bad_percentage: f32,
}

#[derive(Debug, Clone)]
pub struct CorruptionReport {
    pub corrupted_tensors: Vec<TensorCorruption>,
    pub total_blocks_checked: usize,
    pub total_bad_blocks: usize,
}

impl CorruptionReport {
    pub fn is_clean(&self) -> bool {
        self.total_bad_blocks == 0
    }

    pub fn print(&self) -> String {
        if self.is_clean() {
            return "✓ No corruption detected in any tensor blocks.".to_string();
        }
        let mut s = format!(
            "⚠️  CORRUPTION DETECTED: {} bad blocks out of {} checked ({:.2}%)\n",
            self.total_bad_blocks,
            self.total_blocks_checked,
            100.0 * self.total_bad_blocks as f32 / self.total_blocks_checked.max(1) as f32
        );
        s.push_str("   Affected tensors:\n");
        for t in &self.corrupted_tensors {
            s.push_str(&format!(
                "     • {} ({}): {}/{} blocks bad ({:.2}%)\n",
                t.name, t.quant_type, t.blocks_bad, t.blocks_total, t.bad_percentage
            ));
        }
        s.push_str("   Recommendation: Re-download the model file. The current copy has corrupted data.\n");
        s.push_str("   (Inference will continue with corrupted weights zeroed out.)\n");
        s
    }
}

/// Scan raw tensor data for corrupted quantization blocks.
/// Checks each block's scale(s) — NaN, Inf, or absurdly large values indicate corruption.
pub fn scan_for_corruption(file: &GGUFile) -> CorruptionReport {
    let mut corrupted_tensors = Vec::new();
    let mut total_blocks = 0;
    let mut total_bad = 0;

    for t in &file.tensors {
        let qtype = match QuantType::from_u32(t.typ) {
            Some(q) => q,
            None => continue,
        };

        // Only check block-based quant types
        let block_size = match qtype {
            QuantType::Q4_0 | QuantType::Q4_1 | QuantType::IQ4_NL => 32,
            QuantType::Q5_0 | QuantType::Q5_1 => 32,
            QuantType::Q8_0 | QuantType::Q8_1 => 32,
            QuantType::Q2_K | QuantType::Q3_K | QuantType::Q4_K
            | QuantType::Q5_K | QuantType::Q6_K | QuantType::Q8_K
            | QuantType::IQ2_XXS | QuantType::IQ2_XS | QuantType::IQ3_XXS
            | QuantType::IQ3_S | QuantType::IQ4_XS | QuantType::IQ4_K
            | QuantType::IQ5_0 | QuantType::IQ5_NL | QuantType::IQ5_K => 256,
            _ => continue, // F32, F16, BF16 — no blocks to check
        };

        let count: usize = t.dimensions.iter().product::<u64>() as usize;
        let num_blocks = (count + block_size - 1) / block_size;
        let bb = qtype.block_bytes();

        let raw = match file.get_tensor_raw(&t.name) {
            Some(r) => r,
            None => continue,
        };

        let mut bad = 0usize;
        for i in 0..num_blocks {
            let block = &raw[i * bb..(i + 1).min(num_blocks) * bb];
            if block.len() < bb {
                break;
            }

            // Read scale(s) depending on block layout
            let (d, dmin_opt) = match qtype {
                QuantType::Q6_K => {
                    // Q6_K: scale is the last 2 bytes
                    if block.len() >= 210 {
                        let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();
                        (d, None)
                    } else {
                        continue;
                    }
                }
                QuantType::Q8_K => {
                    // Q8_K: scale is f32 at start
                    if block.len() >= 4 {
                        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
                        (d, None)
                    } else {
                        continue;
                    }
                }
                QuantType::Q8_1 => {
                    // Q8_1: d is f32 at 0, dmin is f32 at 4
                    if block.len() >= 8 {
                        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
                        let dmin = f32::from_le_bytes([block[4], block[5], block[6], block[7]]);
                        (d, Some(dmin))
                    } else {
                        continue;
                    }
                }
                _ => {
                    // Most types: d is f16 at bytes 0-1, dmin is f16 at bytes 2-3 (if present)
                    if block.len() >= 2 {
                        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                        let dmin = if block.len() >= 4 {
                            let v = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
                            Some(v)
                        } else {
                            None
                        };
                        (d, dmin)
                    } else {
                        continue;
                    }
                }
            };

            // Check for corruption
            let mut block_bad = false;
            if d.is_nan() || d.is_infinite() || d.abs() > 1e4 {
                block_bad = true;
            }
            if let Some(dmin) = dmin_opt {
                if dmin.is_nan() || dmin.is_infinite() || dmin.abs() > 1e4 {
                    block_bad = true;
                }
            }

            if block_bad {
                bad += 1;
            }
        }

        if bad > 0 {
            total_blocks += num_blocks;
            total_bad += bad;
            corrupted_tensors.push(TensorCorruption {
                name: t.name.clone(),
                quant_type: format!("{:?}", qtype),
                blocks_total: num_blocks,
                blocks_bad: bad,
                bad_percentage: 100.0 * bad as f32 / num_blocks.max(1) as f32,
            });
        }
    }

    CorruptionReport {
        corrupted_tensors,
        total_blocks_checked: total_blocks,
        total_bad_blocks: total_bad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_qwen_model() {
        let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let model = GGUFModel::load(path).expect("Failed to load model");
        println!("Config: {:?}", model.config);
        assert!(model.config.num_hidden_layers > 0);
        assert!(model.config.vocab_size > 0);

        let layer0 = model.load_layer(0).expect("Failed to load layer 0");
        assert!(!layer0.is_empty());
        println!("Layer 0 tensors: {}", layer0.len());

        let special = model.load_special().expect("Failed to load special layers");
        assert!(special.contains_key("model.embed_tokens.weight"));

        // Print capability report
        println!("\n{}", model.capability_report().print());
    }

    #[test]
    fn test_new_model_capability_report() {
        let path = "/home/xander/Documents/portfolio/LeafcutterLLM/Qwen3.5-9B-IQ4_NL.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let model = GGUFModel::load(path).expect("Failed to load model");
        let report = model.capability_report();
        println!("\n{}", report.print());

        assert_eq!(report.architecture, ModelArchitecture::Qwen35);
        assert!(!report.can_run); // IQ4_NL + qwen35 not fully supported yet
    }
}

#[test]
fn debug_check_q4k_values() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let model = GGUFModel::load(path).unwrap();
    let layer0 = model.load_layer(0).unwrap();
    for (name, tensor) in &layer0 {
        let nan_count = tensor.data.iter().filter(|&&v| v.is_nan()).count();
        let inf_count = tensor.data.iter().filter(|&&v| v.is_infinite()).count();
        let min = tensor.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = tensor.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("{}: shape={:?} nan={} inf={} min={} max={}", name, tensor.shape, nan_count, inf_count, min, max);
    }
}

#[test]
fn debug_check_layer1_weights() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() {
        return;
    }
    let model = GGUFModel::load(path).unwrap();
    let layer1 = model.load_layer(1).unwrap();
    for (name, tensor) in &layer1 {
        let nan_count = tensor.data.iter().filter(|&&v| v.is_nan()).count();
        let inf_count = tensor.data.iter().filter(|&&v| v.is_infinite()).count();
        let min = tensor.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = tensor.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("{}: shape={:?} nan={} inf={} min={} max={}", name, tensor.shape, nan_count, inf_count, min, max);
    }
}

#[test]
fn debug_nan_pattern_layer1() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() { return; }
    let model = GGUFModel::load(path).unwrap();
    let layer1 = model.load_layer(1).unwrap();
    
    // Check gate_proj NaN pattern
    if let Some(gate) = layer1.get("mlp.gate_proj.weight") {
        let mut block_nan_counts = vec![];
        for block_start in (0..gate.data.len()).step_by(256) {
            let block_end = (block_start + 256).min(gate.data.len());
            let nan_in_block = gate.data[block_start..block_end].iter().filter(|&&v| v.is_nan()).count();
            if nan_in_block > 0 {
                block_nan_counts.push((block_start / 256, nan_in_block));
            }
        }
        println!("gate_proj: {} blocks with NaN, first 10: {:?}", block_nan_counts.len(), &block_nan_counts[..10.min(block_nan_counts.len())]);
    }
    
    // Check down_proj NaN pattern  
    if let Some(down) = layer1.get("mlp.down_proj.weight") {
        let mut block_nan_counts = vec![];
        for block_start in (0..down.data.len()).step_by(256) {
            let block_end = (block_start + 256).min(down.data.len());
            let nan_in_block = down.data[block_start..block_end].iter().filter(|&&v| v.is_nan()).count();
            if nan_in_block > 0 {
                block_nan_counts.push((block_start / 256, nan_in_block));
            }
        }
        println!("down_proj: {} blocks with NaN, first 10: {:?}", block_nan_counts.len(), &block_nan_counts[..10.min(block_nan_counts.len())]);
    }
}

#[test]
fn debug_nan_positions_layer1() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() { return; }
    let model = GGUFModel::load(path).unwrap();
    let layer1 = model.load_layer(1).unwrap();
    
    if let Some(gate) = layer1.get("mlp.gate_proj.weight") {
        let nan_positions: Vec<usize> = gate.data.iter().enumerate()
            .filter(|(_, &v)| v.is_nan())
            .map(|(i, _)| i)
            .collect();
        println!("gate_proj NaN count: {}", nan_positions.len());
        println!("First 20 NaN positions: {:?}", &nan_positions[..20.min(nan_positions.len())]);
        // Check if they're at regular intervals
        if nan_positions.len() >= 2 {
            let diffs: Vec<usize> = nan_positions.windows(2).map(|w| w[1] - w[0]).collect();
            println!("First 20 intervals: {:?}", &diffs[..20.min(diffs.len())]);
        }
    }
    
    if let Some(down) = layer1.get("mlp.down_proj.weight") {
        let nan_positions: Vec<usize> = down.data.iter().enumerate()
            .filter(|(_, &v)| v.is_nan())
            .map(|(i, _)| i)
            .collect();
        println!("down_proj NaN count: {}", nan_positions.len());
        println!("First 20 NaN positions: {:?}", &nan_positions[..20.min(nan_positions.len())]);
        if nan_positions.len() >= 2 {
            let diffs: Vec<usize> = nan_positions.windows(2).map(|w| w[1] - w[0]).collect();
            println!("First 20 intervals: {:?}", &diffs[..20.min(diffs.len())]);
        }
    }
}

#[test]
fn debug_raw_bytes_layer1_gate() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() { return; }
    let file = GGUFile::open(path).unwrap();
    
    let t = file.tensors.iter().find(|t| t.name == "blk.1.ffn_gate.weight").unwrap();
    let raw = file.get_tensor_raw("blk.1.ffn_gate.weight").unwrap();
    println!("Raw data len: {}", raw.len());
    
    // Check blocks where NaN occurs: block 7, 50, 93, etc.
    // Block 7 starts at byte 7*144 = 1008
    for block_idx in [7, 50, 93, 136] {
        let start = block_idx * 144;
        let block = &raw[start..start+144];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        println!("Block {}: d={:?} dmin={:?}", block_idx, d, dmin);
        // Print first 16 bytes of scales
        println!("  scales first 16 bytes: {:?}", &block[4..20]);
    }
    
    // Compare with block 6 (no NaN, adjacent)
    for block_idx in [6, 49] {
        let start = block_idx * 144;
        let block = &raw[start..start+144];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        println!("Block {} (clean): d={:?} dmin={:?}", block_idx, d, dmin);
    }
}

#[test]
fn debug_values_around_nan() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() { return; }
    let model = GGUFModel::load(path).unwrap();
    let layer1 = model.load_layer(1).unwrap();
    
    if let Some(gate) = layer1.get("mlp.gate_proj.weight") {
        println!("Values around position 1930..1940:");
        for i in 1930..1940 {
            println!("  gate[{}] = {} (is_nan={}, bits={:08x})", i, gate.data[i], gate.data[i].is_nan(), gate.data[i].to_bits());
        }
        // Also check position 1934 in layer 0
    }
    
    let layer0 = model.load_layer(0).unwrap();
    if let Some(gate0) = layer0.get("mlp.gate_proj.weight") {
        println!("Layer 0 values around position 1930..1940:");
        for i in 1930..1940 {
            println!("  gate0[{}] = {} (is_nan={}, bits={:08x})", i, gate0.data[i], gate0.data[i].is_nan(), gate0.data[i].to_bits());
        }
    }
}

#[test]
fn debug_dequantize_sizes() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() { return; }
    let file = GGUFile::open(path).unwrap();
    
    for name in ["blk.0.ffn_gate.weight", "blk.1.ffn_gate.weight"] {
        let t = file.tensors.iter().find(|t| t.name == name).unwrap();
        let raw = file.get_tensor_raw(name).unwrap();
        let size = calculate_tensor_size(&t.dimensions, t.typ);
        println!("{}: dims={:?} type={} calc_size={} raw_len={}", name, t.dimensions, t.typ, size, raw.len());
    }
}

#[test]
fn debug_pre_transpose_nan() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() { return; }
    let file = GGUFile::open(path).unwrap();
    
    // Manually dequantize layer 1 gate WITHOUT transpose
    let t = file.tensors.iter().find(|t| t.name == "blk.1.ffn_gate.weight").unwrap();
    let raw = file.get_tensor_raw("blk.1.ffn_gate.weight").unwrap();
    let shape: Vec<usize> = t.dimensions.iter().map(|&d| d as usize).rev().collect();
    println!("Pre-transpose shape: {:?}", shape);
    
    let count: usize = shape.iter().product();
    let mut out = vec![0.0f32; count];
    crate::kernels::dequantize_q4_k(raw, &mut out);
    
    let nan_positions: Vec<usize> = out.iter().enumerate()
        .filter(|(_, &v)| v.is_nan())
        .map(|(i, _)| i)
        .collect();
    println!("Pre-transpose NaN count: {}", nan_positions.len());
    if nan_positions.len() >= 2 {
        let intervals: Vec<usize> = nan_positions.windows(2).map(|w| w[1] - w[0]).collect();
        println!("First 20 intervals: {:?}", &intervals[..20.min(intervals.len())]);
        // Check if contiguous
        let contiguous_blocks = intervals.iter().filter(|&&v| v == 1).count();
        println!("Contiguous pairs: {}", contiguous_blocks);
    }
}

#[test]
fn debug_single_block_dequant() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() { return; }
    let file = GGUFile::open(path).unwrap();
    let raw = file.get_tensor_raw("blk.1.ffn_gate.weight").unwrap();
    
    // Dequantize just block 7
    let block_data = &raw[7*144..8*144];
    let mut out = vec![0.0f32; 256];
    crate::kernels::dequantize_q4_k(block_data, &mut out);
    
    println!("Block 7 dequantized values around position 138 (1930-1792):");
    for i in 130..150 {
        println!("  out[{}] = {} (bits={:08x})", i, out[i], out[i].to_bits());
    }
    
    // Also dequantize block 6 for comparison
    let block6_data = &raw[6*144..7*144];
    let mut out6 = vec![0.0f32; 256];
    crate::kernels::dequantize_q4_k(block6_data, &mut out6);
    println!("Block 6 dequantized values around position 130..150:");
    for i in 130..150 {
        println!("  out6[{}] = {}", i, out6[i]);
    }
}

#[test]
fn debug_block7_assert() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() { return; }
    let file = GGUFile::open(path).unwrap();
    let raw = file.get_tensor_raw("blk.1.ffn_gate.weight").unwrap();
    
    let block_data = &raw[7*144..8*144];
    let mut out = vec![0.0f32; 256];
    crate::kernels::dequantize_q4_k(block_data, &mut out);
    
    println!("Block 7: d_bytes={:?}", &block_data[0..4]);
    println!("out[138] = {} nan={}", out[138], out[138].is_nan());
    println!("out[139] = {} nan={}", out[139], out[139].is_nan());
    println!("out[140] = {} nan={}", out[140], out[140].is_nan());
    println!("out[141] = {} nan={}", out[141], out[141].is_nan());
    println!("out[142] = {} nan={}", out[142], out[142].is_nan());
    
    // Check if dmin is NaN by manual decode
    let d = half::f16::from_le_bytes([block_data[0], block_data[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block_data[2], block_data[3]]).to_f32();
    println!("d={} dmin={} dmin_nan={}", d, dmin, dmin.is_nan());
    
    assert!(!out[138].is_nan(), "out[138] should not be NaN");
}

#[test]
fn debug_full_dequant_pre_transpose() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() { return; }
    let file = GGUFile::open(path).unwrap();
    let raw = file.get_tensor_raw("blk.1.ffn_gate.weight").unwrap();
    
    let shape: Vec<usize> = vec![11008, 2048];
    let count: usize = shape.iter().product();
    let mut out = vec![0.0f32; count];
    crate::kernels::dequantize_q4_k(raw, &mut out);
    
    println!("Pre-transpose values around 1930..1940:");
    for i in 1930..1940 {
        println!("  out[{}] = {}", i, out[i]);
    }
    
    // Now transpose manually
    let m = 11008;
    let n = 2048;
    let mut result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            result[j * m + i] = out[i * n + j];
        }
    }
    
    println!("Post-transpose values around 1930..1940:");
    for i in 1930..1940 {
        println!("  result[{}] = {}", i, result[i]);
    }
}

#[test]
fn debug_scan_blocks() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() { return; }
    let file = GGUFile::open(path).unwrap();
    let raw = file.get_tensor_raw("blk.1.ffn_gate.weight").unwrap();
    
    let mut huge_d_blocks = vec![];
    let mut nan_dmin_blocks = vec![];
    let num_blocks = raw.len() / 144;
    
    for i in 0..num_blocks {
        let block = &raw[i * 144..(i + 1) * 144];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        
        if dmin.is_nan() {
            nan_dmin_blocks.push(i);
        }
        if d.abs() > 10.0 || dmin.abs() > 10.0 {
            huge_d_blocks.push((i, d, dmin));
        }
    }
    
    println!("NaN dmin blocks: {} {:?}", nan_dmin_blocks.len(), &nan_dmin_blocks[..10.min(nan_dmin_blocks.len())]);
    println!("Huge d/dmin blocks: {}", huge_d_blocks.len());
    for (i, d, dmin) in &huge_d_blocks[..20.min(huge_d_blocks.len())] {
        println!("  block {}: d={} dmin={}", i, d, dmin);
    }
}

#[test]
fn debug_all_layer1_q4k_blocks() {
    let path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(path).exists() { return; }
    let file = GGUFile::open(path).unwrap();
    
    for name in ["blk.1.ffn_gate.weight", "blk.1.ffn_up.weight", "blk.1.attn_q.weight", "blk.1.attn_k.weight", "blk.1.attn_output.weight"] {
        let t = file.tensors.iter().find(|t| t.name == name).unwrap();
        if t.typ != 12 { continue; } // Q4_K
        let raw = file.get_tensor_raw(name).unwrap();
        let num_blocks = raw.len() / 144;
        let mut bad_blocks = 0;
        for i in 0..num_blocks {
            let block = &raw[i * 144..(i + 1) * 144];
            let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
            let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
            if d.abs() > 10.0 || dmin.is_nan() || dmin.abs() > 10.0 {
                bad_blocks += 1;
            }
        }
        println!("{}: blocks={} bad={}", name, num_blocks, bad_blocks);
    }
}
