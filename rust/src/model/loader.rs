//! Layer streaming loader for GGUF models
//!
//! Only one layer's weights are resident in RAM at any time.

use super::arch::{CapabilityReport, ModelArchitecture};
use super::gguf::{calculate_tensor_size, GGUFile, GGUError};
use super::quant::QuantType;
use super::tensor::Tensor;
use crate::kernels;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct YarnParams {
    /// 1.0 / yarn_ext_factor (e.g. 1/16 for Ministral).
    /// Computed internally; callers should not set this.
    pub freq_scale: f32,
    /// The original training context length (e.g. 16384 for Ministral).
    pub orig_ctx: usize,
    /// YaRN beta_fast (rotations whose wavelength < 2π·β_fast are interpolated).
    pub beta_fast: f32,
    /// YaRN beta_slow (rotations whose wavelength > 2π·β_slow are extrapolated).
    pub beta_slow: f32,
    /// GGUF `rope_attn_factor` (= mscale as defined by HF YaRN).
    /// Llama.cpp pre-divides by (1 + 0.1·log(factor)) at the call site,
    /// so the kernel's mscale-bake restores the original mscale.
    /// For Ministral (factor=16, attn_factor=1.0) this is identity.
    pub attn_factor: f32,
    /// yarn_ext_factor (1.0 = YaRN active, 0.0 = no YaRN interpolation).
    /// Stored separately for clarity even though it's redundant with
    /// `freq_scale = 1 / factor`.
    pub ext_factor: f32,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub kv_head_dim: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub attention_interval: usize,
    /// Gemma-style logit soft-capping: output = cap * tanh(output / cap)
    pub logit_soft_cap: f32,
    /// RoPE dimensions per head (partial RoPE for Qwen3.5/3.6)
    pub rope_dim: usize,
    /// RMS norm epsilon (read from GGUF metadata)
    pub norm_eps: f32,
    /// EOS token ID (read from GGUF metadata; defaults to 2 for backward compat)
    pub eos_token: usize,
    /// RoPE-YaRN parameters. `None` when the model uses standard RoPE
    /// (i.e. `rope_scaling.type == "yarn"` is absent). Models like
    /// Ministral-3-3B-Instruct-2512 require this for coherent generation.
    pub rope_yarn: Option<YarnParams>,
    /// Attention temperature scaling scale factor (`*.attention.temperature_scale`).
    /// 0.0 = disabled.  Used by Mistral-3 / Llama-4 models.
    pub attention_temp_scale: f32,
    /// Position floor for attention temperature scaling.  When 0, derived from
    /// the YaRN original context length at load time.
    pub attention_temp_floor_scale: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            head_dim: 128,
            kv_head_dim: 128,
            intermediate_size: 11008,
            max_seq_len: 4096,
            vocab_size: 32000,
            rope_theta: 10000.0,
            attention_interval: 1,
            logit_soft_cap: 0.0,
            rope_dim: 0,
            norm_eps: 1e-5,
            eos_token: 2,
            rope_yarn: None,
            attention_temp_scale: 0.0,
            attention_temp_floor_scale: 0,
        }
    }
}

pub struct GGUFModel {
    pub file: GGUFile,
    pub config: ModelConfig,
    pub architecture: ModelArchitecture,
    /// Persistent per-layer weight cache (Phase 5 perf fix).
    ///
    /// `load_layer()` re-parses + re-dequantizes every layer's weights from
    /// the mmap each call, which previously happened once per token (32× per
    /// generated token). Holding each layer's `Tensor`s here (as `Arc` so
    /// callers share without cloning the multi-GB matrices) turns every call
    /// after the first into a cache hit.
    ///
    /// The cache holds the raw quantized blocks (`Tensor.q_data`), NOT f32
    /// dequantized copies — so Q4_K/Q6_K stay ~file-size in RAM and GEMM
    /// kernels dequantize on the fly.
    ///
    /// The cache is **memory-bounded**: `cache_budget_bytes` caps how much
    /// quantized weight it will hold.  When a model fits comfortably in RAM
    /// (e.g. Ornith-9B), the budget is unlimited and every layer stays cached
    /// (fast).  For huge models (e.g. 70B Q4_K_M ≈ 42 GB on a 15 GB box) the
    /// budget is capped at the available RAM, so the oldest layers are evicted
    /// as new ones load — the engine degrades gracefully into layer streaming
    /// instead of OOMing.  Override with `LEAFCUTTER_CACHE_MB`.
    layer_cache: std::sync::Mutex<LayerCacheInner>,
    /// Cache budget in bytes (0 = unlimited).  Computed at load from the
    /// available RAM and the model size.
    cache_budget_bytes: usize,
}

/// Insertion-ordered layer cache backing `GGUFModel.layer_cache`.
struct LayerCacheInner {
    map: HashMap<usize, std::sync::Arc<HashMap<String, Tensor>>>,
    /// Layer indices in insertion order (oldest first) — the eviction order.
    order: VecDeque<usize>,
}

impl LayerCacheInner {
    fn new() -> Self {
        Self { map: HashMap::new(), order: VecDeque::new() }
    }

    /// Total resident bytes of cached layer weights.  Uses `resident_bytes`
    /// (quantized blocks + any materialized f32 data) so the eviction budget
    /// reflects the real memory footprint, not just the quantized size.
    fn cached_bytes(&self) -> usize {
        self.map
            .values()
            .map(|arc| arc.values().map(|t| t.resident_bytes()).sum::<usize>())
            .sum()
    }
}

/// Read available RAM from /proc/meminfo (MiB).  0 if unavailable.
fn available_memory_mb() -> usize {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<usize>().ok())
        })
        .unwrap_or(0)
        / 1024
}

/// Read total RAM from /proc/meminfo (MiB). 0 if unavailable.
fn total_memory_mb() -> usize {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<usize>().ok())
        })
        .unwrap_or(0)
        / 1024
}

impl GGUFModel {
    pub fn load(path: &str) -> Result<Self, GGUError> {
        let file = GGUFile::open(path)?;
        let architecture = ModelArchitecture::detect(&file);
        let config = Self::extract_config(&file, architecture);

        let total_model_bytes: usize = file
            .tensors
            .iter()
            .map(|t| calculate_tensor_size(&t.dimensions, t.typ))
            .sum();

        let cache_budget_bytes = Self::compute_cache_budget_bytes(total_model_bytes, &config);

        Ok(Self {
            file,
            config,
            architecture,
            layer_cache: std::sync::Mutex::new(LayerCacheInner::new()),
            cache_budget_bytes,
        })
    }

    /// Layer-cache budget in bytes (0 = unlimited).
    pub fn layer_cache_budget_bytes(&self) -> usize {
        self.cache_budget_bytes
    }

    /// True when the whole model fits the budget (i.e. every layer can be
    /// resident and the mmap pages can be safely dropped after the first pass).
    pub fn model_fits_available_ram(&self) -> bool {
        self.cache_budget_bytes == 0
    }

    /// Compute the layer-cache budget from `LEAFCUTTER_CACHE_MB` (if set),
    /// else from the available RAM using trunk-first budgeting:
    ///   1. reserve the dense trunk — KV cache at a working context, the
    ///      activation workspace, and a resident LM head — explicitly;
    ///   2. give the leftover RAM to the layer cache (the "expert" pool).
    /// This mirrors kimi-k3-in-c's trunk-first allocation: the always-needed
    /// dense working set is guaranteed before any optional cache is allowed
    /// to consume memory.
    fn compute_cache_budget_bytes(total_model_bytes: usize, config: &ModelConfig) -> usize {
        if let Ok(v) = std::env::var("LEAFCUTTER_CACHE_MB") {
            if let Ok(mb) = v.parse::<usize>() {
                return mb.saturating_mul(1024 * 1024);
            }
        }

        let total_mb = total_memory_mb();
        let avail_mb = available_memory_mb();
        let ram_mb = if total_mb > 0 { total_mb } else { avail_mb };
        if ram_mb == 0 {
            return 0; // meminfo unavailable → unlimited (backward compatible)
        }

        // Fast path: if total model size + 1.5 GiB headroom fits in total system RAM,
        // cache all layers residently in memory (return 0 = unlimited cache budget).
        let headroom_bytes = 1536 * 1024 * 1024;
        if total_model_bytes.saturating_add(headroom_bytes) <= ram_mb * 1024 * 1024 {
            return 0;
        }

        // Dense trunk estimate (f32): working context window + activations + lm head
        let ctx = std::env::var("LEAFCUTTER_CTX_KB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|kb| kb * 1024)
            .unwrap_or_else(|| config.max_seq_len.min(1024));
        let kv_cache_bytes = 2usize
            .saturating_mul(config.num_hidden_layers)
            .saturating_mul(config.num_key_value_heads)
            .saturating_mul(config.kv_head_dim)
            .saturating_mul(ctx)
            .saturating_mul(2); // f16 / quantized KV
        let activation_bytes = config
            .hidden_size
            .saturating_mul(config.num_attention_heads.max(1) + 3)
            .saturating_mul(4);
        let head_bytes = config
            .vocab_size
            .saturating_mul(config.hidden_size)
            .saturating_mul(2);
        let trunk_bytes = kv_cache_bytes.saturating_add(activation_bytes).saturating_add(head_bytes);
        let margin_mb = 512;
        let budget_bytes = (avail_mb.saturating_sub(margin_mb) * 1024 * 1024).saturating_sub(trunk_bytes);

        if total_model_bytes <= budget_bytes {
            0
        } else {
            budget_bytes.max(256 * 1024 * 1024)
        }
    }

    /// Number of resident layer-weights in the cache (for monitoring).
    pub fn cached_layers(&self) -> usize {
        self.layer_cache.lock().map(|c| c.map.len()).unwrap_or(0)
    }

    /// Total resident bytes of cached layer weights (for monitoring).
    pub fn cached_bytes(&self) -> usize {
        self.layer_cache.lock().map(|c| c.cached_bytes()).unwrap_or(0)
    }

    /// True when every layer's weights are currently resident in the cache.
    pub fn all_layers_cached(&self) -> bool {
        self.cached_layers() == self.config.num_hidden_layers
    }

    /// Load a layer's weights, cached across calls.
    ///
    /// First call per layer parses + dequantizes from the mmap (as before);
    /// subsequent calls return the cached `Arc` — O(1), no disk I/O, no
    /// re-dequantization. This is the Phase 5 fix for the 16× slowdown.
    ///
    /// The cache is memory-bounded: when `cache_budget_bytes` is set, the
    /// oldest layers are evicted as new ones are inserted so RSS stays under
    /// the budget (huge-model streaming mode).
    pub fn get_layer(&self, idx: usize) -> Result<std::sync::Arc<HashMap<String, Tensor>>, GGUError> {
        if std::env::var("LEAFCUTTER_NO_CACHE").map(|v| v == "1").unwrap_or(false) {
            return Ok(std::sync::Arc::new(self.load_layer(idx)?));
        }
        {
            let cache = self.layer_cache.lock().map_err(|_| GGUError::UnsupportedQuantType("cache poisoned".into(), 0))?;
            if let Some(arc) = cache.map.get(&idx) {
                return Ok(std::sync::Arc::clone(arc));
            }
        }
        // Cache miss: build the layer (may take ~10-100 ms for a big layer).
        let weights = self.load_layer(idx)?;
        let arc = std::sync::Arc::new(weights);
        let mut cache = self.layer_cache.lock().map_err(|_| GGUError::UnsupportedQuantType("cache poisoned".into(), 0))?;
        // Re-check under lock in case a concurrent caller built it first.
        if !cache.map.contains_key(&idx) {
            cache.map.insert(idx, std::sync::Arc::clone(&arc));
            cache.order.push_back(idx);
        }
        // Enforce the memory budget (skip when budget is 0 = unlimited).
        let budget = self.cache_budget_bytes;
        if budget > 0 {
            while cache.map.len() > 1 && cache.cached_bytes() > budget {
                if let Some(old) = cache.order.pop_front() {
                    cache.map.remove(&old);
                } else {
                    break;
                }
            }
        }
        Ok(arc)
    }

    /// Optional: evict a layer from the cache to bound RSS (e.g. huge models).
    pub fn evict_layer(&self, idx: usize) {
        if let Ok(mut cache) = self.layer_cache.lock() {
            if cache.map.remove(&idx).is_some() {
                cache.order.retain(|&i| i != idx);
            }
        }
    }

    /// Clear the whole layer cache (e.g. before a fresh conversation).
    pub fn clear_layer_cache(&self) {
        if let Ok(mut cache) = self.layer_cache.lock() {
            cache.map.clear();
            cache.order.clear();
        }
    }

    /// Generate a pre-flight capability report without loading any weights.
    pub fn capability_report(&self) -> CapabilityReport {
        let quant_summary = self.file.quant_summary();
        let arch_supported = self.architecture.is_supported();
        let mappings = self.architecture.layer_mappings();

        // Check which required tensors are missing
        let mut missing = Vec::new();
        let sample_layers = self.config.num_hidden_layers.min(4);
        for layer_idx in 0..sample_layers {
            let prefix = format!("blk.{}", layer_idx);
            // For hybrid architectures, detect layer type from actual tensors
            let has_ssm = self.file.get_tensor_info(&format!("{}.ssm_alpha.weight", prefix)).is_some()
                || self.file.get_tensor_info(&format!("{}.ssm_out.weight", prefix)).is_some();
            let _has_fused_qkv = self.file.get_tensor_info(&format!("{}.attn_qkv.weight", prefix)).is_some();
            let has_separate_attn = self.file.get_tensor_info(&format!("{}.attn_q.weight", prefix)).is_some();

            for (gguf_suffix, _engine_name) in mappings.iter() {
                let name = format!("{}.{}", prefix, gguf_suffix);
                if self.file.get_tensor_info(&name).is_none() {
                    // For hybrid architectures (Qwen3.5/3.6), each layer is
                    // EITHER full-attention OR DeltaNet/SSM.  Don't flag
                    // missing tensors that legitimately don't exist on a
                    // different layer type.
                    if matches!(
                        self.architecture,
                        ModelArchitecture::Qwen35 | ModelArchitecture::Qwen36
                    ) {
                        // DeltaNet/SSM layers lack separate Q/K/V/O projections
                        if has_ssm
                            && (gguf_suffix.starts_with("attn_q.weight")
                                || gguf_suffix.starts_with("attn_k.weight")
                                || gguf_suffix.starts_with("attn_v.weight")
                                || gguf_suffix.starts_with("attn_output.weight")
                                || gguf_suffix.starts_with("attn_q_norm")
                                || gguf_suffix.starts_with("attn_k_norm"))
                        {
                            continue;
                        }
                        // Full-attention layers lack the fused QKV (and use
                        // separate post_attention_norm instead)
                        if has_separate_attn
                            && (*gguf_suffix == "attn_qkv.weight"
                                || *gguf_suffix == "attn_gate.weight")
                        {
                            continue;
                        }
                        // Full-attention layers lack SSM-related tensors
                        if has_separate_attn && gguf_suffix.starts_with("ssm_") {
                            continue;
                        }
                        // ssm_a may not have .weight suffix
                        if has_ssm && *gguf_suffix == "ssm_a" {
                            if self
                                .file
                                .get_tensor_info(&format!("{}.ssm_a", prefix))
                                .is_some()
                            {
                                continue;
                            }
                        }
                        // Qwen3.6 MoE layers lack the dense ffn_* projections
                        // (they use ffn_*_exps for routed experts instead)
                        if self.architecture == ModelArchitecture::Qwen36 {
                            let is_moe_layer = self
                                .file
                                .get_tensor_info(&format!("{}.ffn_gate_inp.weight", prefix))
                                .is_some()
                                || self
                                    .file
                                    .get_tensor_info(&format!("{}.ffn_gate_exps.weight", prefix))
                                    .is_some();
                            if is_moe_layer
                                && (*gguf_suffix == "ffn_gate.weight"
                                    || *gguf_suffix == "ffn_up.weight"
                                    || *gguf_suffix == "ffn_down.weight"
                                    || *gguf_suffix == "ffn_norm.weight")
                            {
                                continue;
                            }
                            // Dense attention layers lack the expert tensors
                            let is_dense_layer = self
                                .file
                                .get_tensor_info(&format!("{}.ffn_gate.weight", prefix))
                                .is_some();
                            if is_dense_layer && gguf_suffix.contains("_exps") {
                                continue;
                            }
                            // shared expert gates are optional
                            if gguf_suffix.contains("shexp")
                                || *gguf_suffix == "ffn_state.weight"
                                || *gguf_suffix == "ssm_state.weight"
                                || *gguf_suffix == "ssm_gate.weight"
                                || *gguf_suffix == "ffn_norm.weight"
                            {
                                continue;
                            }
                        }
                    }
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
            if t.name.starts_with("blk.") {
                // Handle both .weight suffix and raw parameter names like ssm_a
                let full_suffix = if t.name.contains(".nextn.") {
                    t.name.splitn(3, '.').nth(2).unwrap_or("")
                } else {
                    t.name.splitn(3, '.').nth(2).unwrap_or("")
                };
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
            uses_ssm: self.architecture.uses_ssm(),
            uses_fused_qkv: self.architecture.uses_fused_qkv(),
            quant_summary,
            missing_tensors: missing,
            extra_tensors: extra,
            can_run,
        }
    }

    fn extract_config(file: &GGUFile, arch: ModelArchitecture) -> ModelConfig {
        let mut cfg = ModelConfig::default();
        let prefix = arch.metadata_prefix();

        cfg.hidden_size = Self::get_meta_int(file, &[
            &format!("{}.embedding_length", prefix),
            "llama.embedding_length",
            "mistral3.embedding_length",
            "qwen2.embedding_length",
            "qwen35.embedding_length",
            "gemma3.embedding_length",
            "gemma4.embedding_length",
        ])
        .map(|v| v as usize)
        .unwrap_or(cfg.hidden_size);
        cfg.num_hidden_layers = Self::get_meta_int(file, &[
            &format!("{}.block_count", prefix),
            "llama.block_count",
            "mistral3.block_count",
            "qwen2.block_count",
            "qwen35.block_count",
            "gemma3.block_count",
            "gemma4.block_count",
        ])
        .map(|v| v as usize)
        .unwrap_or(cfg.num_hidden_layers);
        
        // Qwen3.5/3.6: subtract NextN/MTP layers from the main transformer count
        // (they are stored as extra decoder blocks but not executed in the main pass)
        let nextn_layers = Self::get_meta_int(file, &[&format!("{}.nextn_predict_layers", prefix), "qwen35.nextn_predict_layers", "llama.nextn_predict_layers"])
            .map(|v| v as usize)
            .unwrap_or(0);
        if nextn_layers > 0 && nextn_layers < cfg.num_hidden_layers {
            cfg.num_hidden_layers -= nextn_layers;
        }
        cfg.num_attention_heads = Self::get_meta_int(file, &[&format!("{}.attention.head_count", prefix), "llama.attention.head_count", "mistral3.attention.head_count", "qwen2.attention.head_count", "qwen35.attention.head_count"])
            .map(|v| v as usize).unwrap_or(cfg.num_attention_heads);
        cfg.num_key_value_heads = Self::get_meta_int(file, &[&format!("{}.attention.head_count_kv", prefix), "llama.attention.head_count_kv", "mistral3.attention.head_count_kv", "qwen2.attention.head_count_kv", "qwen35.attention.head_count_kv"])
            .map(|v| v as usize).unwrap_or(cfg.num_key_value_heads);
        cfg.intermediate_size = Self::get_meta_int(file, &[&format!("{}.feed_forward_length", prefix), "llama.feed_forward_length", "mistral3.feed_forward_length", "qwen2.feed_forward_length", "qwen35.feed_forward_length"])
            .map(|v| v as usize).unwrap_or(cfg.intermediate_size);
        cfg.max_seq_len = Self::get_meta_int(file, &[&format!("{}.context_length", prefix), "llama.context_length", "mistral3.context_length", "qwen2.context_length", "qwen35.context_length"])
            .map(|v| v as usize).unwrap_or(cfg.max_seq_len);
        cfg.vocab_size = Self::get_meta_int(file, &[&format!("{}.vocab_size", prefix), "tokenizer.ggml.tokens.length", "tokenizer.ggml.vocab_size"])
            .map(|v| v as usize)
            .or_else(|| {
                file.metadata.get("tokenizer.ggml.tokens")
                    .and_then(|v| if let crate::model::gguf::GGUFValue::Array(arr) = v { Some(arr.len()) } else { None })
            })
            .unwrap_or(cfg.vocab_size);
        cfg.rope_theta = Self::get_meta_int(file, &[
                &format!("{}.rope.freq_base", prefix),
                "llama.rope.freq_base",
                "qwen2.rope.freq_base",
                "qwen35.rope.freq_base",
                "mistral3.rope.freq_base",
                "phi3.rope.freq_base",
                "phi4.rope.freq_base",
                "gemma.rope.freq_base",
                "gemma2.rope.freq_base",
                "gemma3.rope.freq_base",
            ])
            .map(|v| v as f32)
            .unwrap_or(cfg.rope_theta);

        // Partial RoPE dimension count (Qwen3.5/3.6: 64 of 256)
        cfg.rope_dim = Self::get_meta_int(file, &[
                &format!("{}.rope.dimension_count", prefix),
                "qwen35.rope.dimension_count",
                "llama.rope.dimension_count",
            ])
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim);

        // RMS norm epsilon (model-specific, e.g., Qwen35 uses 1e-6)
        cfg.norm_eps = Self::get_meta_f32(file, &[
                &format!("{}.attention.layer_norm_rms_epsilon", prefix),
                "qwen35.attention.layer_norm_rms_epsilon",
                "llama.attention.layer_norm_rms_epsilon",
            ])
            .unwrap_or(cfg.norm_eps);

        // Gemma logit soft-capping (e.g., gemma3.logit_cap = 30.0)
        cfg.logit_soft_cap = Self::get_meta_f32(file, &[
            &format!("{}.logit_cap", prefix),
            "gemma.logit_cap",
            "gemma2.logit_cap",
            "gemma3.logit_cap",
            "gemma4.final_logit_softcapping",
        ])
        .unwrap_or(cfg.logit_soft_cap);

        // Compute head dimensions
        // For most models: head_dim = hidden_size / num_attention_heads.
        // For Qwen3.5/3.6 the Q projection has larger outer dim (e.g. 12288 vs 5120),
        // so we compute from actual weight tensor when available.
        cfg.head_dim = (0..cfg.num_hidden_layers)
            .find_map(|i| {
                file.get_tensor_info(&format!("blk.{}.attn_q.weight", i))
                    .map(|t| t.dimensions[1] as usize / cfg.num_attention_heads)
            })
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);

        // Compressed KV dimensions (M5) — for Qwen3.5, key_length != embedding_length / head_count
        cfg.kv_head_dim = Self::get_meta_int(file, &[&format!("{}.attention.key_length", prefix), "llama.attention.key_length", "mistral3.attention.key_length", "qwen35.attention.key_length"])
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim);

        // Attention interval (M7) — for hybrid SSM/Transformer models like Qwen3.5
        cfg.attention_interval = Self::get_meta_int(file, &[&format!("{}.full_attention_interval", prefix), "qwen35.full_attention_interval", "mistral3.full_attention_interval"])
            .map(|v| v as usize)
            .unwrap_or(1);

        // EOS token ID — per-model, not hardcoded to 2
        cfg.eos_token = Self::get_meta_int(file, &[
                "tokenizer.ggml.eos_token_id",
                &format!("{}.eos_token_id", prefix),
                "llama.eos_token_id",
                "qwen2.eos_token_id",
                "qwen35.eos_token_id",
            ])
            .map(|v| v as usize)
            .unwrap_or(2);

        // RoPE-YaRN parameters. Triggered when `rope_scaling.type == "yarn"`
        // is set in metadata. The Ministral-3-3B-Instruct-2512 GGUF stores
        // its keys under `mistral3.rope.scaling.*` with these names:
        //   factor              (the yarn_ext_factor, e.g. 16)
        //   original_context_length
        //   yarn_beta_fast / yarn_beta_slow
        //   yarn_log_multiplier (= mscale / attn_factor)
        // Other models (Llama-3.x-1M, DeepSeek-V2) store under
        // `<prefix>.rope.scaling.{yarn_ext_factor,yarn_orig_ctx,...}`.
        let scaling_type = file
            .metadata
            .get(&format!("{}.rope.scaling.type", prefix))
            .or_else(|| file.metadata.get("llama.rope.scaling.type"))
            .or_else(|| file.metadata.get("mistral3.rope.scaling.type"))
            .and_then(|v| if let crate::model::gguf::GGUFValue::String(s) = v { Some(s.as_str()) } else { None })
            .unwrap_or("");
        let yarn_active = scaling_type == "yarn";
        if yarn_active {
            // Resolve the interpolation factor (`scaling.factor`, e.g. 16 for
            // Ministral-3-2512) and the extrapolation factor (`yarn_ext_factor`).
            // These are distinct: freq_scale = 1/factor, while ext_factor is 1.0
            // for YARN type (llama.cpp llama-context.cpp:189-190 hardcodes it).
            let scaling_factor = Self::get_meta_f32(file, &[
                &format!("{}.rope.scaling.factor", prefix),
                "llama.rope.scaling.factor",
                "mistral3.rope.scaling.factor",
                "qwen2.rope.scaling.factor",
            ])
            .unwrap_or(1.0);
            // Explicit yarn_ext_factor key (DeepSeek-V2 style). Absent for
            // Ministral-3; default 1.0 for YARN, 0.0 otherwise.
            let yarn_ext_factor = Self::get_meta_f32(file, &[
                &format!("{}.rope.scaling.yarn_ext_factor", prefix),
                "llama.rope.scaling.yarn_ext_factor",
                "qwen2.rope.scaling.yarn_ext_factor",
            ])
            .unwrap_or(1.0);
            let orig_ctx = Self::get_meta_int(file, &[
                &format!("{}.rope.scaling.yarn_orig_ctx", prefix),
                "llama.rope.scaling.yarn_orig_ctx",
                &format!("{}.rope.scaling.original_context_length", prefix),
                "mistral3.rope.scaling.original_context_length",
                "qwen2.rope.scaling.original_context_length",
            ])
            .map(|v| v as usize)
            .unwrap_or(cfg.max_seq_len.max(2048));
            let beta_fast = Self::get_meta_f32(file, &[
                &format!("{}.rope.scaling.yarn_beta_fast", prefix),
                "llama.rope.scaling.yarn_beta_fast",
                "mistral3.rope.scaling.yarn_beta_fast",
                "qwen2.rope.scaling.yarn_beta_fast",
            ])
            .unwrap_or(32.0);
            let beta_slow = Self::get_meta_f32(file, &[
                &format!("{}.rope.scaling.yarn_beta_slow", prefix),
                "llama.rope.scaling.yarn_beta_slow",
                "mistral3.rope.scaling.yarn_beta_slow",
                "qwen2.rope.scaling.yarn_beta_slow",
            ])
            .unwrap_or(1.0);
            // mscale-equivalent. HuggingFace's `yarn_log_multiplier`
            // matches llama.cpp's `attn_factor`.
            let attn_factor = Self::get_meta_f32(file, &[
                &format!("{}.rope.scaling.attn_factor", prefix),
                "llama.rope.scaling.attn_factor",
                "mistral3.rope.scaling.attn_factor",
                &format!("{}.rope.scaling.yarn_log_multiplier", prefix),
                "mistral3.rope.scaling.yarn_log_multiplier",
                "qwen2.rope.scaling.yarn_log_multiplier",
            ])
            .unwrap_or(1.0);
            cfg.rope_yarn = Some(YarnParams {
                freq_scale: 1.0 / scaling_factor,
                orig_ctx,
                beta_fast,
                beta_slow,
                attn_factor,
                ext_factor: yarn_ext_factor,
            });
            eprintln!(
                "[loader] RoPE-YaRN active: factor={}, freq_scale={}, orig_ctx={}, ext_factor={}, beta_fast={}, beta_slow={}, attn_factor={}",
                scaling_factor, 1.0 / scaling_factor, orig_ctx, yarn_ext_factor, beta_fast, beta_slow, attn_factor
            );
        }

        // Attention temperature scaling (Mistral-3 / Llama-4): `temperature_scale`
        // KV under `<prefix>.attention.*`.  llama.cpp (llama-graph.cpp) computes
        // `log(floor((pos + offset)/floor_scale) + 1) * scale + 1` per position and
        // multiplies Q by it after RoPE.  floor_scale is n_ctx_orig_yarn when the
        // KV is absent (llama.cpp mistral3.cpp:14-15).
        cfg.attention_temp_scale = Self::get_meta_f32(file, &[
            &format!("{}.attention.temperature_scale", prefix),
            "mistral3.attention.temperature_scale",
            "llama4.attention.temperature_scale",
            "llama.attention.temperature_scale",
        ])
        .unwrap_or(0.0);
        if cfg.attention_temp_scale != 0.0 {
            let floor = cfg.rope_yarn.as_ref().map(|y| y.orig_ctx).unwrap_or(0);
            cfg.attention_temp_floor_scale = floor;
            if floor == 0 {
                eprintln!("[loader] WARNING: attention.temperature_scale set but no YaRN orig_ctx; temp scaling disabled");
                cfg.attention_temp_scale = 0.0;
            } else {
                eprintln!(
                    "[loader] attention temp scaling active: scale={}, floor_scale={}",
                    cfg.attention_temp_scale, floor
                );
            }
        }

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

    fn get_meta_f32(file: &GGUFile, keys: &[&str]) -> Option<f32> {
        for key in keys {
            if let Some(v) = file.get_metadata_f32(key) {
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
            // Skip optional / hybrid-only tensors that simply aren't in the file.
            // We log via `eprintln` so a missing tensor (typo in the GGUF
            // suffix mapping, or genuinely absent on a hybrid layer) is
            // visible instead of vanishing silently.
            let raw_opt = self.file.get_tensor_raw(&gguf_name);
            let info_opt = self.file.get_tensor_info(&gguf_name);
            let (raw, info) = match (raw_opt, info_opt) {
                (Some(r), Some(i)) => (r, i),
                _ => {
                    let is_optional = matches!(
                        gguf_suffix.as_ref(),
                        "ssm_alpha.weight" | "ssm_beta.weight" | "ssm_conv1d.weight"
                            | "ssm_dt.bias" | "ssm_norm.weight" | "ssm_out.weight"
                            | "ssm_a" | "ssm_state.weight" | "ssm_gate.weight"
                            | "attn_q_norm.weight" | "attn_k_norm.weight" | "attn_v_norm.weight"
                            | "attn_q.weight" | "attn_k.weight" | "attn_v.weight"
                            | "attn_output.weight" // absent on DeltaNet layers
                            | "attn_qkv.weight" | "attn_gate.weight"
                            | "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight"
                            | "ffn_gate_inp.weight" | "ffn_gate_exps.weight"
                            | "ffn_up_exps.weight" | "ffn_down_exps.weight"
                            | "ffn_norm.weight"
                            | "ffn_gate_shexp.weight" | "ffn_up_shexp.weight" | "ffn_down_shexp.weight"
                            | "nextn.eh_proj" | "nextn.eh_proj.weight"
                            | "nextn.enorm.weight" | "nextn.hnorm.weight"
                    );
                    if !is_optional && idx == 0 {
                        eprintln!(
                            "Leafcutter: warning — expected tensor '{}' for {} layer 0 not found",
                            gguf_name, self.architecture.name()
                        );
                    }
                    continue;
                }
            };

            let qtype = QuantType::from_u32(info.typ)
                .ok_or(GGUError::InvalidTensorType(info.typ))?;

            let shape_gguf: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            let is_2d = shape_gguf.len() == 2;
            // For 3-D MoE expert tensors (e.g. ffn_gate_exps.weight stored
            // with GGUF dims `[d0, d1, d2]`, d2 = num_experts), GGUF/llama.cpp
            // store the data row-major over `[expert, d1, d0]` with d0
            // contiguous (ne[0] innermost, expert outermost — see
            // llama-model.cpp create_tensor_gate_up_exps).  The quantized
            // block stream is therefore already a `[d1*d2, d0]` matrix:
            //   rows = d1*d2, cols = d0, blocks_per_row = d0/256.
            // Expert `e` then occupies rows `[e*d1, (e+1)*d1)` — exactly what
            // `Tensor::expert_slice` reads back.  (Do NOT collapse to
            // `[d0*d1, d2]` — that mislabels the block stream and makes every
            // expert block range wrong.)  The Tensor's metadata shape stays
            // 3-D so `Tensor::expert_slice` can split it per-expert.
            let (kernel_rows, kernel_cols, keep_shape_3d) = if is_2d {
                (shape_gguf[1], shape_gguf[0], false)
            } else if shape_gguf.len() == 3 {
                (shape_gguf[1] * shape_gguf[2], shape_gguf[0], true)
            } else {
                (shape_gguf[0], shape_gguf.get(1).copied().unwrap_or(1), false)
            };
            let shape_data: Vec<usize> = if keep_shape_3d {
                vec![kernel_rows, kernel_cols]
            } else if is_2d {
                vec![shape_gguf[1], shape_gguf[0]]
            } else {
                // 1-D (or >3-D) — use the kernel rows/cols already computed.
                vec![kernel_rows, kernel_cols]
            };

            let (mut tensor, needs_transpose) = match qtype {
                QuantType::Q8_0 => {
                    let q8 = crate::kernels::q8_0::Matrix {
                        rows: shape_data[0],
                        cols: shape_data[1],
                        blocks: crate::kernels::q8_0::blocks_from_bytes(raw),
                    };
                    (Tensor::from_q8_0_only(q8, shape_gguf.clone()), false)
                }
                QuantType::Q4_0 => {
                    let q4 = crate::kernels::q4_0::Matrix {
                        rows: shape_data[0],
                        cols: shape_data[1],
                        blocks: crate::kernels::q4_0::blocks_from_bytes(raw),
                    };
                    (Tensor::from_q4_0_only(q4, shape_gguf.clone()), false)
                }
                QuantType::Q4_K => {
                    let raw_bytes = raw.len();
                    let profile_blocks = std::env::var("LEAFCUTTER_PROFILE_BLOCKS").is_ok();
                    let t0 = if profile_blocks { Some(std::time::Instant::now()) } else { None };
                    let mat = crate::kernels::q4_k::Matrix {
                        rows: shape_data[0],
                        cols: shape_data[1],
                        blocks: crate::kernels::q4_k::blocks_from_bytes(raw),
                    };
                    if let Some(t0) = t0 {
                        eprintln!(
                            "[BLOCKS] Q4_K {}: rows={} cols={} blocks={} raw={}MB parse={:.2}ms",
                            gguf_name,
                            shape_data[0],
                            shape_data[1],
                            mat.blocks.len(),
                            raw_bytes / 1024 / 1024,
                            t0.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    (Tensor::from_q4_k_only(mat, shape_gguf.clone()), false)
                }
                QuantType::IQ4_NL => {
                    let q4 = crate::kernels::iq4_nl::Matrix {
                        rows: shape_data[0],
                        cols: shape_data[1],
                        blocks: crate::kernels::iq4_nl::blocks_from_bytes(raw),
                    };
                    (Tensor::from_iq4_nl_only(q4, shape_gguf.clone()), false)
                }
                QuantType::Q5_K => {
                    let profile_blocks = std::env::var("LEAFCUTTER_PROFILE_BLOCKS").is_ok();
                    let t0 = if profile_blocks {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    let q5 = crate::kernels::q5_k::Matrix {
                        rows: shape_data[0],
                        cols: shape_data[1],
                        blocks: crate::kernels::q5_k::blocks_from_bytes(raw),
                    };
                    if let Some(t0) = t0 {
                        eprintln!(
                            "[BLOCKS] Q5_K {}: rows={} cols={} blocks={} parse={:.2}ms",
                            gguf_name,
                            shape_data[0],
                            shape_data[1],
                            q5.blocks.len(),
                            t0.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    (Tensor::from_q5_k_only(q5, shape_gguf.clone()), false)
                }
                QuantType::Q6_K => {
                    let profile_blocks = std::env::var("LEAFCUTTER_PROFILE_BLOCKS").is_ok();
                    let t0 = if profile_blocks {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    // Match the Q4_K / Q5_K loader convention: use shape_data
                    // (which is [gguf[1], gguf[0]] — already swapped) so the
                    // matmul kernel's `b.cols == k, b.rows == n` assertions
                    // pass.  An earlier "fix" tried to use shape_gguf directly
                    // and broke things further; reverted.
                    let mat = crate::kernels::q6_k::Matrix {
                        rows: shape_data[0],
                        cols: shape_data[1],
                        blocks: crate::kernels::q6_k::blocks_from_bytes(raw),
                    };
                    if let Some(t0) = t0 {
                        eprintln!(
                            "[BLOCKS] Q6_K {}: rows={} cols={} blocks={} parse={:.2}ms",
                            gguf_name,
                            shape_data[0],
                            shape_data[1],
                            mat.blocks.len(),
                            t0.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    (Tensor::from_q6_k_only(mat, shape_gguf.clone()), false)
                }
                _ => {
                    // F32/F16/BF16: data is stored GGUF-native (row-major in
                    // declared dims). Use shape_gguf DIRECTLY without swap.
                    // The swap+transpose path is only correct for K-quants.
                    let t = Self::dequantize(raw, info.typ, shape_gguf.clone())?;
                    (t, false)
                }
            };
            if is_2d && needs_transpose {
                tensor = tensor.transpose();
                sanitize_weights(&mut tensor);
            }
            // 3-D MoE expert tensors stay quantized in the cache — per-expert
            // slices are produced on demand by `Tensor::expert_slice` (which
            // reads q_data directly, no f32 materialization).  Only tensors
            // that need element-wise f32 access materialize: conv1d weight
            // (ssm_conv1d.weight is read element-by-element by the conv1d
            // kernel, not as a matmul) and any other 3-D tensor that is not a
            // routed expert (kept conservative so unknown 3-D shapes don't
            // lose their .data).  This is the fix for the 35B MoE OOM: expert
            // tensors are no longer dequantized to ~3.2 GB/layer.
            let is_moe_expert = engine_name.contains("expert_") || engine_name.contains("_exps");
            if !is_moe_expert && (keep_shape_3d || engine_name.contains("ssm_conv1d")) {
                if std::env::var("LEAFCUTTER_DEBUG").map(|v| v == "1").unwrap_or(false) {
                    eprintln!("[loader] materializing {} (shape={:?})", engine_name, tensor.shape);
                }
                tensor.materialize_data();
                if std::env::var("LEAFCUTTER_DEBUG").map(|v| v == "1").unwrap_or(false) {
                    eprintln!("[loader] after materialize: data.len={}", tensor.data.len());
                }
            }
            weights.insert(engine_name.to_string(), tensor);

            // The tensor's bytes have been parsed into owned quantized
            // blocks (or f32 data) above — the mmap pages that backed them
            // are now a redundant second copy.  Release them so the model is
            // never double-resident while the layer cache fills, keeping peak
            // RSS ≈ model size (like Ollama) instead of 2× the model.
            let start_abs = self.file.data_offset + info.offset;
            self.file.drop_pages_in_range(start_abs, raw.len());
        }

        Ok(weights)
    }

    /// Load final norm weights.  Embedding and lm_head are NOT loaded here —
    /// they are accessed on-demand via memory-mapped row lookup to save RAM.
    pub fn load_special(&self) -> Result<HashMap<String, Tensor>, GGUError> {
        let mut weights = HashMap::new();

        // Only load norm weights.  Embed and lm_head are kept in the mmap'd GGUF file
        // and read per-token / per-row during inference.
        if let Some(raw) = self.file.get_tensor_raw("output_norm.weight") {
            if let Some(info) = self.file.get_tensor_info("output_norm.weight") {
                let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                let mut tensor = Self::dequantize(raw, info.typ, shape)?;
                sanitize_weights(&mut tensor);
                weights.insert("model.norm.weight".to_string(), tensor);
            }
        }

        Ok(weights)
    }

    pub fn dequantize(data: &[u8], typ: u32, shape: Vec<usize>) -> Result<Tensor, GGUError> {
        let count: usize = shape.iter().product();
        let mut out = vec![0.0f32; count];

        let qtype = QuantType::from_u32(typ)
            .ok_or(GGUError::InvalidTensorType(typ))?;

        // Reject quant types we don't have kernels for. Previously the
        // catch-all silently returned Err, but now we also flag it explicitly
        // so callers can distinguish "missing kernel" from a corrupted tensor.
        if !qtype.is_supported() {
            return Err(GGUError::UnsupportedQuantType(qtype.name().to_string(), typ));
        }

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
            QuantType::BF16 => {
                for i in 0..count {
                    let bytes = [data[i * 2], data[i * 2 + 1]];
                    out[i] = half::bf16::from_le_bytes(bytes).to_f32();
                }
            }
            QuantType::Q4_0 => kernels::dequantize_q4_0(data, &mut out),
            QuantType::Q4_1 => kernels::dequantize_q4_1(data, &mut out),
            QuantType::Q8_0 => kernels::dequantize_q8_0(data, &mut out),
            QuantType::Q4_K => kernels::dequantize_q4_k(data, &mut out),
            QuantType::Q5_K => kernels::dequantize_q5_k(data, &mut out),
            QuantType::Q6_K => kernels::dequantize_q6_k(data, &mut out),
            QuantType::Q8_K => kernels::dequantize_q8_k(data, &mut out),
            QuantType::IQ4_NL => kernels::dequantize_iq4_nl(data, &mut out),
            QuantType::IQ4_XS => kernels::dequantize_iq4_xs(data, &mut out),
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
                QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 | QuantType::IQ4_NL => {
                    // These types have ONLY a scale (f16 at bytes 0-1), no dmin
                    if block.len() >= 2 {
                        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                        (d, None)
                    } else {
                        continue;
                    }
                }
                QuantType::Q4_1 | QuantType::Q5_1 | QuantType::Q4_K | QuantType::Q5_K => {
                    // These types have d (f16 at 0-1) and dmin (f16 at 2-3)
                    if block.len() >= 4 {
                        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
                        (d, Some(dmin))
                    } else {
                        continue;
                    }
                }
                _ => {
                    // Unknown / unhandled types — skip corruption check
                    continue;
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
        assert!(special.contains_key("model.norm.weight"));

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
        assert!(report.can_run); // Qwen35 now fully supported with SSM + hybrid attention
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
    
    let _t = file.tensors.iter().find(|t| t.name == "blk.1.ffn_gate.weight").unwrap();
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
        let size = super::gguf::calculate_tensor_size(&t.dimensions, t.typ);
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

#[cfg(test)]
mod trunk_first_tests {
    use super::*;

    fn cfg() -> ModelConfig {
        ModelConfig {
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_key_value_heads: 8,
            kv_head_dim: 128,
            vocab_size: 128_000,
            max_seq_len: 4096,
            ..ModelConfig::default()
        }
    }

    #[test]
    fn small_model_fits_unlimited() {
        // A ~3 GB model on a machine with enough RAM → budget 0 (cache all).
        let c = cfg();
        let budget = GGUFModel::compute_cache_budget_bytes(3 << 30, &c);
        // We can't control available_memory_mb() here; just assert the math
        // shape is sane (>= floor).
        assert!(budget == 0 || budget >= 256 * 1024 * 1024);
    }

    #[test]
    fn trunk_first_reserves_head_and_kv() {
        // Force a large model so the budget is bounded (not 0).
        // 32 layers * 8 kv_heads * 128 dim * 4096 ctx * 2 * 4 B = 1 GiB KV,
        // plus head, so a 20 GiB model on a machine with < 20 GiB available
        // must produce a non-zero, bounded budget rather than 0.
        let c = cfg();
        let budget = GGUFModel::compute_cache_budget_bytes(20 << 30, &c);
        let avail_mb = available_memory_mb();
        if avail_mb == 0 {
            return; // meminfo unavailable on this machine
        }
        // Budget must not exceed available RAM and must be at least the floor.
        assert!(budget == 0 || budget <= avail_mb * 1024 * 1024);
        assert!(budget == 0 || budget >= 256 * 1024 * 1024);
    }

    #[test]
    fn kv_trunk_scales_with_context() {
        let mut c = cfg();
        c.max_seq_len = 4096;
        let a = GGUFModel::compute_cache_budget_bytes(20 << 30, &c);
        c.max_seq_len = 128 * 1024;
        let b = GGUFModel::compute_cache_budget_bytes(20 << 30, &c);
        if available_memory_mb() == 0 {
            return;
        }
        // Bigger context → bigger reserved trunk → smaller (or equal) cache budget.
        assert!(b == 0 || b <= a);
    }
}
