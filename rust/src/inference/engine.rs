//! Unified inference engine with native Qwen3.5 hybrid support
//!
//! Implements hybrid Transformer-Mamba forward pass:
//!   - Most layers: SSM (Mamba-style state space)
//!   - Every Nth layer (attention_interval): standard attention
//!   - Final layer: optional speculative decoding heads
//!
//! For standard architectures (Llama, Qwen2, Mistral), all layers use attention.

use crate::model::arch::ModelArchitecture;
use crate::model::loader::{GGUFModel, ModelConfig};
use crate::model::tensor::Tensor;
use crate::cache::{KVCache, ssm_state::SSMStateCache, deltanet_state::DeltaNetStateCache};
use crate::inference::attention::{attention_forward, AttentionParams};
use crate::inference::deltanet::{deltanet_forward, DeltaNetParams};
use crate::inference::mla::{mla_forward, MlaParams};
use crate::inference::moe::MoeConfig;
use crate::inference::sampler::sample_top_p;
use crate::inference::ssm::{ssm_forward, SSMConfig};
use crate::inference::speculative::SpeculativeHead;
use crate::tokenizer::GgufTokenizer;
use rayon::prelude::*;
use std::collections::HashMap;
#[cfg(feature = "llama-ffi")]
use crate::llama_ffi::{LlamaModel, LlamaContext};
#[cfg(feature = "llama-ffi")]
use std::path::Path;

pub struct Engine {
    pub model: GGUFModel,
    pub config: ModelConfig,
    pub kv_cache: KVCache,
    pub special_weights: HashMap<String, Tensor>,
    pub attn_params: AttentionParams,
    pub ssm_config: SSMConfig,
    pub deltanet_params: DeltaNetParams,
    /// Multi-Latent Attention params (DeepSeek-2 / GLM-DSA).
    pub mla_params: crate::inference::mla::MlaParams,
    /// MoE FFN params (DeepSeek-2 / Qwen-MoE / future).
    pub moe_params: crate::inference::moe::MoeConfig,
    /// Per-layer Gemma-3/4 routing (alternating G/S, per-layer head_count_kv).
    /// Empty for non-Gemma architectures.
    pub gemma_layouts: Vec<crate::inference::gemma::GemmaLayerParams>,
    /// Gemma (1+w) RMSNorm epsilon; 1e-6 by default, models override it.
    pub gemma_norm_eps: f32,
    /// Gemma logit soft-cap (final projection limiter). 0 = disabled.
    /// Negative = use Gemma 4 default (-30.0 → soft-cap at 30).
    /// Field on ModelConfig already exists; this is a per-engine cache.
    pub gemma_logit_softcap: f32,
    /// Whether the embedding pipeline applied `hidden ×= sqrt(n_embd)`,
    /// requiring `1/sqrt(n_embd)` revert in the lm_head path.
    pub embed_uses_sqrt_scale: bool,
    pub speculative_head: Option<SpeculativeHead>,
    /// Whether lm_head is tied to token embeddings (no separate output.weight tensor).
    lm_head_tied: bool,
    /// Cached quantized lm_head weights (output.weight or token_embd.weight),
    /// kept in Q6_K block form instead of a ~3.8 GB f32 dequant cache.
    /// Populated at Engine::load time. Shape: [vocab_size, hidden_size].
    cached_lm_head: Option<crate::kernels::q6_k::Matrix>,
    /// GGUF vocab tokenizer (lazy-initialized from model metadata).
    /// SSM state cache: persistent hidden state for Mamba-style layers.
    pub ssm_cache: SSMStateCache,
    /// DeltaNet state cache: persistent matrix state for DeltaNet layers.
    pub deltanet_cache: DeltaNetStateCache,
    /// Path to the GGUF model file — used to lazily (re)build the tokenizer.
    gguf_path: String,
    /// Cached tokenizer; rebuilt only when None.  Avoids re-extracting the
    /// vocab from the mmap on every `generate_text` call (which used to
    /// happen per-token, ~50 KB of work per generation step).
    cached_tokenizer: std::sync::Mutex<Option<GgufTokenizer>>,
    /// Cached lm_head projection buffer size; avoids per-token resize on
    /// thread-local buffers in `lm_head_projection`.
    cached_lm_head_size: std::sync::atomic::AtomicUsize,
    /// Whether the mmap pages have been released after the layer cache warmed.
    /// All 32 layer weight sets are then served from the RAM layer cache, so
    /// the file-backed pages are a redundant second copy (~model size).
    pages_dropped: bool,
    /// Current sequence position offset for RoPE. Tracks total tokens processed
    /// across forward calls within a generation session.
    pub seq_offset: usize,
    // Embedding lookup is on-demand via mmap — see embed_lookup_mmap()
    #[cfg(feature = "llama-ffi")]
    ffi_model: Option<LlamaModel>,
    #[cfg(feature = "llama-ffi")]
    ffi_context: Option<LlamaContext>,
}

impl Engine {
    /// Build the MoE config from the model's GGUF metadata.
    ///
    /// Reads `*.expert_count`, `*.expert_used_count`, `*.expert_feed_forward_length`,
    /// `*.expert_shared_feed_forward_length`, and gating function hints
    /// (`*.expert_gating_func`, `*.norm_topk_prob`, `*.routed_scaling_factor`).
    ///
    /// Falls back to `MoeConfig::default()` when metadata is missing, which
    /// works for DeepSeek-3 / Kimi-K2.6 style MoE.
    fn build_moe_config(
        model: &GGUFModel,
        config: &ModelConfig,
    ) -> MoeConfig {
        let get_int = |keys: &[&str]| -> usize {
            for key in keys {
                if let Some(v) = model.file.get_metadata_int(key) {
                    return v as usize;
                }
            }
            0
        };
        let get_f32 = |keys: &[&str], default: f32| -> f32 {
            for key in keys {
                if let Some(v) = model.file.get_metadata_f32(key) {
                    return v;
                }
            }
            default
        };

        let arch_prefix = match model.architecture {
            ModelArchitecture::Qwen36 => "qwen35moe",
            ModelArchitecture::Qwen35 => "qwen35",
            _ => "llama", // DeepSeek-2 uses "llama.expert_count"
        };

        let prefix = format!("{}.", arch_prefix);
        let num_experts = get_int(&[
            &format!("{}expert_count", prefix),
            "llama.expert_count",
        ]);
        let num_experts_used = get_int(&[
            &format!("{}expert_used_count", prefix),
            "llama.expert_used_count",
        ]);
        let expert_ffn = get_int(&[
            &format!("{}expert_feed_forward_length", prefix),
            "llama.expert_feed_forward_length",
        ]);

        if num_experts == 0 || num_experts_used == 0 || expert_ffn == 0 {
            // No MoE metadata → likely a dense-only model.  Return defaults
            // so dispatch still works (dispatcher won't actually use MoE).
            return MoeConfig {
                num_experts: 0,
                num_experts_used: 0,
                expert_ffn: config.intermediate_size,
                gating_func: 1,
                norm_topk_prob: false,
                routed_scaling_factor: 1.0,
                norm_eps: config.norm_eps,
            };
        }

        let gating_func = get_int(&[
            &format!("{}expert_gating_func", prefix),
            "llama.expert_gating_func",
        ]) as u32;
        let norm_topk_prob = get_int(&[
            &format!("{}norm_topk_prob", prefix),
            "llama.norm_topk_prob",
        ]) != 0;
        let routed_scaling_factor = get_f32(
            &[
                &format!("{}routed_scaling_factor", prefix),
                "llama.routed_scaling_factor",
            ],
            1.0,
        );

        MoeConfig {
            num_experts,
            num_experts_used,
            expert_ffn,
            gating_func: if gating_func == 2 { 2 } else { 1 },
            norm_topk_prob,
            routed_scaling_factor,
            norm_eps: config.norm_eps,
        }
    }

    #[cfg(feature = "llama-ffi")]
    fn load_ffi(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model = LlamaModel::load(Path::new(path), 0)
            .map_err(|e| format!("Failed to load model via FFI: {}", e))?;
        let context = LlamaContext::new(&model, 4096, 4)
            .map_err(|e| format!("Failed to create FFI context: {}", e))?;

        // Still load GGUFModel for metadata (config, tokenizer info, etc.)
        let gguf_model = GGUFModel::load(path)?;
        let config = gguf_model.config.clone();

        Ok(Self {
            model: gguf_model,
            config,
            kv_cache: KVCache::new(0),
            special_weights: HashMap::new(),
            attn_params: AttentionParams::default(),
            ssm_config: SSMConfig::default(),
            deltanet_params: DeltaNetParams::default(),
            mla_params: crate::inference::mla::MlaParams::default(),
            moe_params: crate::inference::moe::MoeConfig::default(),
            gemma_layouts: Vec::new(),
            gemma_norm_eps: 1e-6,
            gemma_logit_softcap: 0.0,
            embed_uses_sqrt_scale: false,
            speculative_head: None,
            lm_head_tied: false,
            ssm_cache: SSMStateCache::new(),
            deltanet_cache: DeltaNetStateCache::new(),
            gguf_path: path.to_string(),
            cached_tokenizer: std::sync::Mutex::new(None),
            cached_lm_head: None,
            cached_lm_head_size: std::sync::atomic::AtomicUsize::new(0),
            pages_dropped: true,
            seq_offset: 0,
            #[cfg(feature = "llama-ffi")]
            ffi_model: Some(model),
            #[cfg(feature = "llama-ffi")]
            ffi_context: Some(context),
        })
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // ── Load GGUF for architecture detection ──────────────────────
        let model = GGUFModel::load(path)?;
        let arch = model.architecture;

        // ── Native path ──────────────────────────────────────────────
        // Run pre-flight capability report
        let report = model.capability_report();
        if !report.can_run {
            eprintln!("\n{}", report.print());

            // ── AUTO-FALLBACK: unsupported quants → try FFI ──────────
            if crate::llama_ffi::is_available() {
                eprintln!("  Native path blocked. Trying llama.cpp FFI fallback...");
                #[cfg(feature = "llama-ffi")]
                return Self::load_ffi(path);
                #[cfg(not(feature = "llama-ffi"))]
                return Err("Model cannot run natively. Build with --features llama-ffi for fallback support.".into());
            } else {
                return Err(format!(
                    "Model cannot run natively (unsupported quant types). \
                     Build with --features llama-ffi for auto-fallback support. \
                     Details: architecture={} unsupported_quant={} missing_tensors={}",
                    report.architecture.name(),
                    report.quant_summary.unsupported.len(),
                    report.missing_tensors.len()
                ).into());
            }
        }

        if std::env::var("LEAFCUTTER_DEBUG").map(|v| v == "1").unwrap_or(false) {
            eprintln!("  Using native backend for {}", arch.name());
        }

        let mut config = model.config.clone();
        // Verify hidden_size against actual tensor dimensions — metadata may lie
        // (e.g. Ministral-3B: metadata says 4096, token_embd.weight is 3072)
        if let Some(info) = model.file.get_tensor_info("token_embd.weight") {
            let actual_hidden = info.dimensions[0] as usize;
            if actual_hidden != config.hidden_size && actual_hidden > 0 {
                eprintln!("  Correcting hidden_size: metadata={} → actual={}",
                    config.hidden_size, actual_hidden);
                config.hidden_size = actual_hidden;
            }
        }
        // Verify num_hidden_layers against actual tensor presence — metadata may lie
        // (e.g. Ministral-3B: metadata says 32, but only 26 layers exist)
        let actual_layers = (0..config.num_hidden_layers)
            .filter(|&i| {
                model.file.get_tensor_info(&format!("blk.{}.attn_norm.weight", i)).is_some()
                    || model.file.get_tensor_info(&format!("blk.{}.ffn_norm.weight", i)).is_some()
            })
            .count();
        if actual_layers != config.num_hidden_layers && actual_layers > 0 {
            eprintln!("  Correcting num_hidden_layers: metadata={} → actual={}",
                config.num_hidden_layers, actual_layers);
            config.num_hidden_layers = actual_layers;
        }
        let mut special_weights = model.load_special()?;
        let kv_cache = KVCache::new(config.num_hidden_layers);

        // Build attention params from actual weight shapes (metadata may lie)
        let mut attn_params = Self::infer_attention_params(&model, &config);
        attn_params.use_fused_qkv = report.uses_fused_qkv;
        attn_params.use_gate = report.uses_ssm;

        // Build DeltaNet params for hybrid architectures
        let deltanet_params = if report.uses_ssm {
            Self::infer_deltanet_params(&model, &config)
        } else {
            DeltaNetParams::default()
        };

        // Build MoE config from model metadata (must happen before
        // model/config are moved into the Engine struct).
        let moe_params = Self::build_moe_config(&model, &config);

        // Build SSM config for hybrid architectures from model metadata
        let ssm_config = if report.uses_ssm {
            let get_meta = |keys: &[&str]| -> Option<i64> {
                for key in keys {
                    if let Some(v) = model.file.get_metadata_int(key) {
                        return Some(v);
                    }
                }
                None
            };
            // Read actual state_size from ssm_a tensor (metadata may lie)
            let actual_state_size = (0..config.num_hidden_layers)
                .find_map(|i| {
                    model.file.get_tensor_info(&format!("blk.{}.ssm_a", i))
                        .map(|t| t.dimensions.iter().map(|&d| d as usize).product())
                })
                .unwrap_or(128);
            SSMConfig {
                state_size: actual_state_size,
                inner_size: get_meta(&["qwen35.ssm.inner_size", "ssm.inner_size"]).map(|v| v as usize).unwrap_or(config.intermediate_size),
                time_step_rank: get_meta(&["qwen35.ssm.time_step_rank", "ssm.time_step_rank"]).map(|v| v as usize).unwrap_or(32),
                conv_kernel: get_meta(&["qwen35.ssm.conv_kernel", "ssm.conv_kernel"]).map(|v| v as usize).unwrap_or(4),
                group_count: get_meta(&["qwen35.ssm.group_count", "ssm.group_count"]).map(|v| v as usize).unwrap_or(16),
            }
        } else {
            SSMConfig::default()
        };

        // Load speculative decoding head if present
        let speculative_head = if report.architecture == crate::model::arch::ModelArchitecture::Qwen35 {
            let mut draft_weights = HashMap::new();
            for tensor in &model.file.tensors {
                if tensor.name.starts_with("nextn") {
                    if let Some(raw) = model.file.get_tensor_raw(&tensor.name) {
                        if let Some(info) = model.file.get_tensor_info(&tensor.name) {
                            let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).rev().collect();
                            if let Ok(t) = crate::model::loader::GGUFModel::dequantize(raw, info.typ, shape) {
                                draft_weights.insert(tensor.name.clone(), t);
                            }
                        }
                    }
                }
            }
            SpeculativeHead::from_weights(&draft_weights, 4)
        } else {
            None
        };

        // Don't keep embed in RAM — use mmap per-row lookup instead.
        special_weights.remove("model.embed_tokens.weight");

        // LM head: cache the Q6_K blocks once (~0.8 GB for Ornith) to avoid
        // re-dequantizing from mmap every token (~372ms → ~2ms).  Only for
        // models that fit in RAM: a huge output.weight (e.g. 70B ≈ 8.8 GB
        // resident) would blow the "large model on small hardware" budget, so
        // it falls back to the per-row mmap path instead.  Override with
        // `LEAFCUTTER_CACHE_HEAD=0`.
        let lm_head_tied = !model.file.get_tensor_info("output.weight").is_some();
        let lm_head_tensor_name = if lm_head_tied { "token_embd.weight" } else { "output.weight" };
        // Default ON for models that fully fit in RAM (cache budget == 0):
        // holding output.weight as a resident Q6_K cache makes lm_head a fast
        // quantized GEMV instead of per-row mmap re-dequantization (~310ms →
        // ~44ms for Ornith's 248K vocab), and the model fits anyway so the
        // extra ~0.8 GB RSS is within budget.  For "large model on small
        // hardware" it falls back to the per-row mmap path instead.  Override
        // with `LEAFCUTTER_CACHE_HEAD=0` (or `=1` to force on).
        let cache_head = match std::env::var("LEAFCUTTER_CACHE_HEAD") {
            Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
            Err(_) => model.layer_cache_budget_bytes() == 0,
        };
        let cached_lm_head = if cache_head && model.model_fits_available_ram() {
            load_lm_head_cache(&model, lm_head_tensor_name)
        } else {
            None
        };
        special_weights.remove(lm_head_tensor_name);

        // Embedding lookup is done on-demand via mmap per-row dequantization.
        // Never pre-dequantize the full embedding table — it would use 1-4 GB of RAM.

        // ── Gemma-specific metadata, captured before `model` is moved. ──
        let is_gemma = matches!(report.architecture, ModelArchitecture::Gemma);
        let gemma_norm_eps = if is_gemma {
            let v = model
                .file
                .get_metadata_f32("gemma4.attention.layer_norm_rms_epsilon")
                .unwrap_or(0.0);
            if v != 0.0 { v } else { 1e-6 }
        } else { 1e-6 };
        let gemma_logit_softcap = if is_gemma {
            model
                .file
                .get_metadata_f32("gemma4.final_logit_softcapping")
                .filter(|&v| v > 0.0)
                .unwrap_or(0.0)
        } else { 0.0 };
        let gemma_layouts = if is_gemma {
            infer_gemma_layouts(&model, &report.architecture)
        } else { Vec::new() };

        Ok(Self {
            model,
            config,
            kv_cache,
            special_weights,
            attn_params,
            ssm_config,
            deltanet_params,
            mla_params: crate::inference::mla::MlaParams::default(),
            moe_params,
            gemma_layouts,
            gemma_norm_eps,
            gemma_logit_softcap,
            embed_uses_sqrt_scale: is_gemma,
            speculative_head,
            lm_head_tied,
            cached_lm_head,
            cached_lm_head_size: std::sync::atomic::AtomicUsize::new(0),
            ssm_cache: SSMStateCache::new(),
            deltanet_cache: DeltaNetStateCache::new(),
            gguf_path: path.to_string(),
            cached_tokenizer: std::sync::Mutex::new(None),
            pages_dropped: false,
            seq_offset: 0,
            #[cfg(feature = "llama-ffi")]
            ffi_model: None,
            #[cfg(feature = "llama-ffi")]
            ffi_context: None,
        })
    }

    // -------------------------------------------------------------------------
    // DeltaNet parameter inference from actual weight shapes
    // -------------------------------------------------------------------------
    fn infer_deltanet_params(model: &GGUFModel, config: &ModelConfig) -> DeltaNetParams {
        let num_layers = config.num_hidden_layers;
        for layer_idx in 0..num_layers.min(4) {
            let prefix = format!("blk.{}", layer_idx);
            let has_qkv = model.file.get_tensor_info(&format!("{}.{}", prefix, "attn_qkv.weight")).is_some();
            if !has_qkv { continue; }

            if let Some(info) = model.file.get_tensor_info(&format!("{}.{}", prefix, "attn_qkv.weight")) {
                let dims: Vec<Vec<usize>> = info.dimensions.iter().map(|&d| vec![d as usize]).collect();
                // 2-D GGUF layout: dim[0] is hidden_in (n_embd), dim[1] is conv_dim (= 2*K + V after the qkv projection gap).
                let hidden_in = *info.dimensions.first().unwrap_or(&0) as usize;
                let conv_dim = *info.dimensions.get(1).unwrap_or(&0) as usize;
                let _ = dims; // keep linter happy on existing field reads

                // num_v_heads = ssm_dt_rank (matches the shape of `ssm_dt.bias` AND ssm_a,
                // and the second dim of `ssm_alpha.weight` = [n_embd, n_v_heads]).
                // Reference: llama.cpp qwen35.cpp:61  `n_v_heads = hparams.ssm_dt_rank`
                let num_v_heads = model.file.get_tensor_info(&format!("{}.{}", prefix, "ssm_dt.bias"))
                    .or_else(|| model.file.get_tensor_info(&format!("{}.{}", prefix, "ssm_a")))
                    .map(|t| t.dimensions.iter().map(|&d| d as usize).product::<usize>())
                    .unwrap_or(32);

                // num_qk_heads = ssm_n_group (one / (groups)) heads-per-K-head.
                // Reference: llama.cpp qwen35.cpp:60 `n_k_heads = hparams.ssm_n_group`.
                // We can derive it from the FIRST dim of `ssm_alpha.weight`'s *transposed* GGUF layout too,
                // but the metadata is cleaner. Fall back to `ssm.group_count` then 16.
                let num_qk_heads: usize = {
                    // shape of ssm_alpha.weight = [n_embd, n_v_heads] — already tells us V-heads.
                    // For K-heads, we rely on metadata; ssm.group_count is the canonical key.
                    if let Some(meta) = model.file.get_metadata_int("qwen35.ssm.group_count") {
                        let v = meta as usize;
                        if v > 0 { v } else { 16 }
                    } else if let Some(meta) = model.file.get_metadata_int("ssm.group_count") {
                        let v = meta as usize;
                        if v > 0 { v } else { 16 }
                    } else {
                        16
                    }
                };

                // head_v_dim from ssm_out[0] / num_v_heads, OR from ssm_norm.weight shape
                let head_v_dim = if let Some(out_info) = model.file.get_tensor_info(&format!("{}.{}", prefix, "ssm_out.weight")) {
                    let in_v = *out_info.dimensions.first().unwrap_or(&0) as usize;
                    (in_v / num_v_heads.max(1)).max(1)
                } else if let Some(norm_info) = model.file.get_tensor_info(&format!("{}.{}", prefix, "ssm_norm.weight")) {
                    *norm_info.dimensions.first().unwrap_or(&128) as usize
                } else { 128 };

                // head_k_dim  derived from invariant: conv_dim = 2*K + V = 2 * (num_qk_heads * head_k_dim) + (num_v_heads * head_v_dim)
                let head_k_dim = if num_qk_heads > 0 && conv_dim >= num_v_heads * head_v_dim {
                    let k_total = (conv_dim - num_v_heads * head_v_dim) / 2;
                    (k_total / num_qk_heads).max(1)
                } else {
                    head_v_dim
                };

                // Cross-check: ensure 2*qk_h*head_k + v_h*head_v == conv_dim (sanity)
                if num_qk_heads * head_k_dim * 2 + num_v_heads * head_v_dim != conv_dim {
                    eprintln!(
                        "  DeltaNet WARN: dim mismatch on layer {}: 2*{}*{} + {}*{} = {} != conv_dim {}",
                        layer_idx, num_qk_heads, head_k_dim, num_v_heads, head_v_dim,
                        2 * num_qk_heads * head_k_dim + num_v_heads * head_v_dim, conv_dim
                    );
                }

                let conv_kernel = if let Some(conv_info) = model.file.get_tensor_info(&format!("{}.{}", prefix, "ssm_conv1d.weight")) {
                    *conv_info.dimensions.first().unwrap_or(&4) as usize
                } else { 4 };

                if std::env::var("LEAFCUTTER_DEBUG").map(|v| v == "1").unwrap_or(false) {
                    eprintln!(
                        "  DeltaNet: qk_heads={}, v_heads={}, head_k={}, head_v={}, conv_dim={}, conv_k={}, hidden_in={}",
                        num_qk_heads, num_v_heads, head_k_dim, head_v_dim, conv_dim, conv_kernel, hidden_in
                    );
                }

                return DeltaNetParams {
                    num_qk_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    conv_dim,
                    conv_kernel,
                    state_size: head_v_dim,
                    norm_eps: config.norm_eps,
                };
            }
        }
        eprintln!("  Warning: Could not infer DeltaNet params, using defaults");
        DeltaNetParams::default()
    }
    // -------------------------------------------------------------------------
    // Attention parameter inference from actual weight shapes
    // -------------------------------------------------------------------------
    fn infer_attention_params(model: &GGUFModel, config: &ModelConfig) -> AttentionParams {
        // mistral3 (and similar NORM-rope arches) rotate consecutive pairs (2d, 2d+1);
        // classic arches use NEOX-style half-dim pairs (d, d+rope_dim/2).
        let rope_pair_norm = match model.file.metadata.get("general.architecture") {
            Some(crate::model::gguf::GGUFValue::String(s)) => s == "mistral3",
            _ => false,
        };
        let num_layers = config.num_hidden_layers;
        let mut q_out_dim = 0usize;
        let mut kv_out_dim = 0usize;
        let mut o_in_dim = 0usize;
        for layer_idx in 0..num_layers.min(8) {
            if let Some(info) = model.file.get_tensor_info(&format!("blk.{}.attn_q.weight", layer_idx)) {
                q_out_dim = info.dimensions[1] as usize;
                if let Some(k_info) = model.file.get_tensor_info(&format!("blk.{}.attn_k.weight", layer_idx)) {
                    kv_out_dim = k_info.dimensions[1] as usize;
                }
                if let Some(o_info) = model.file.get_tensor_info(&format!("blk.{}.attn_output.weight", layer_idx)) {
                    // GGUF stores 2-D weights as [in_dim, out_dim]
                    o_in_dim = o_info.dimensions[0] as usize;
                }
                break;
            }
        }

        let window_size = model.file.get_metadata_int("llama.attention.sliding_window")
            .or_else(|| model.file.get_metadata_int("mistral.attention.sliding_window"))
            .or_else(|| model.file.get_metadata_int("qwen35.attention.sliding_window"))
            .map(|v| v as usize)
            .unwrap_or(0);

        if q_out_dim == 0 {
            return AttentionParams {
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                kv_head_dim: config.kv_head_dim,
                rope_theta: config.rope_theta,
                rope_dim: config.rope_dim,
                use_fused_qkv: false,
                use_gate: false,
                window_size,
                yarn: config.rope_yarn.clone(),
                temp_scale: config.attention_temp_scale,
                temp_floor_scale: config.attention_temp_floor_scale,
                rope_pair_norm,
            };
        }

        let num_kv_heads = model.file.get_metadata_int("qwen35.attention.head_count_kv")
            .or_else(|| model.file.get_metadata_int("attention.head_count_kv"))
            .or_else(|| model.file.get_metadata_int("llama.attention.head_count_kv"))
            .map(|v| v as usize)
            .unwrap_or(config.num_key_value_heads);

        let kv_head_dim = if num_kv_heads > 0 { kv_out_dim / num_kv_heads } else { config.kv_head_dim };
        // Derive num_heads from O_proj's expected input rather than Q's total dims.
        // Q may have extra dimensions (e.g. 12288 = 48 × 256) but O_proj expects
        // only num_heads × kv_head_dim input dims (e.g. 6144 = 24 × 256).
        let num_heads = if kv_head_dim > 0 && o_in_dim > 0 { o_in_dim / kv_head_dim } else { config.num_attention_heads };
        let head_dim = if num_heads > 0 { q_out_dim / num_heads } else { config.head_dim };

        let rope_theta = model.file.get_metadata_int("qwen35.rope.freq_base")
            .or_else(|| model.file.get_metadata_int("llama.rope.freq_base"))
            .map(|v| v as f32)
            .unwrap_or(config.rope_theta);
        if std::env::var("LEAFCUTTER_ROPE_DEBUG").is_ok() {
            eprintln!("[rope] theta={} (cfg default={})", rope_theta, config.rope_theta);
        }

        AttentionParams {
            num_heads,
            num_kv_heads,
            head_dim,
            kv_head_dim,
            rope_theta,
            rope_dim: config.rope_dim,
            use_fused_qkv: false,
            use_gate: false,
            window_size,
            yarn: config.rope_yarn.clone(),
            temp_scale: config.attention_temp_scale,
            temp_floor_scale: config.attention_temp_floor_scale,
            rope_pair_norm,
        }
    }

    /// Public generate dispatch — uses FFI for Qwen, native for others.
    pub fn generate(&mut self, tokens: &[usize], max_tokens: usize, temperature: f32, top_p: f32) -> Vec<usize> {
        #[cfg(feature = "llama-ffi")]
        if let Some(model) = &self.ffi_model {
            // Recreate context to ensure fresh KV cache (test_generation calls forward before generate)
            let mut ctx = LlamaContext::new(model, 4096, 4)
                .expect("Failed to recreate FFI context");
            let tokens_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let eos = model.eos_token();
            let generated = ctx.generate(&tokens_i32, max_tokens, temperature, eos);
            self.ffi_context = Some(ctx);
            return generated.into_iter().map(|t| t as usize).collect();
        }
        self.generate_native(tokens, max_tokens, temperature, top_p)
    }

    pub fn generate_native(&mut self, tokens: &[usize], max_tokens: usize, temperature: f32, top_p: f32) -> Vec<usize> {
        self.kv_cache.clear();
        self.ssm_cache.clear();
        self.deltanet_cache.clear();
        self.seq_offset = 0;

        // Prefill
        let logits = match self.forward_native(tokens) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Forward pass failed: {}", e);
                return vec![];
            }
        };
        self.seq_offset = tokens.len();
        let mut next_token = sample_top_p(&logits, temperature, top_p);
        let mut generated = vec![next_token];

        if next_token == self.config.eos_token {
            return generated;
        }

        // Decode loop
        for _ in 0..max_tokens - 1 {
            let mut logits = match self.forward_native(&[next_token]) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Forward pass failed: {}", e);
                    break;
                }
            };
            self.seq_offset += 1;

            next_token = sample_top_p(&logits, temperature, top_p);
            generated.push(next_token);

            if next_token == self.config.eos_token {
                break;
            }
        }

        generated
    }

    /// Streaming variant of `generate_native`.  Invokes the callback for each
    /// sampled token (id + already-decoded surface string).  Generation
    /// halts when the callback returns `false`, EOS is sampled, or
    /// `max_tokens` is reached.  All engine optimizations carry over:
    /// anti-doom loop suppression, async layer prefetch, etc.
    ///
    /// Used by the `leaf` chat REPL so tokens print as they're produced
    /// rather than batched at the end.
    pub fn generate_streaming_with<F>(
        &mut self,
        tokens: &[usize],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        mut on_token: F,
    ) -> Vec<usize>
    where
        F: FnMut(usize, &str) -> bool,
    {
        // No stop_tokens parameter → use the engine's default eos_token.
        self.generate_streaming_with_stops(
            tokens,
            max_tokens,
            temperature,
            top_p,
            &[],
            &mut on_token,
        )
    }

    /// Streaming generation with explicit stop tokens.
    ///
    /// When `stop_tokens` is non-empty, generation stops only when the
    /// model emits one of those tokens.  When empty, falls back to the
    /// engine's `config.eos_token` (the GGUF metadata's EOS id).
    ///
    /// This is important for reasoning models like Ornith / Qwen3.5 where
    /// `<|im_end|>` (the GGUF EOS) appears BOTH after the thinking block
    /// AND after the final response — stopping on the first EOS would
    /// truncate the answer.  The caller passes the profile's stop_tokens
    /// (which may be empty to disable early stopping entirely).
    pub fn generate_streaming_with_stops<F>(
        &mut self,
        tokens: &[usize],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        stop_tokens: &[usize],
        mut on_token: F,
    ) -> Vec<usize>
    where
        F: FnMut(usize, &str) -> bool,
    {
        self.kv_cache.clear();
        self.ssm_cache.clear();
        self.deltanet_cache.clear();
        self.seq_offset = 0;

        // Prefill
        let logits = match self.forward_native(tokens) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Forward pass failed: {}", e);
                return vec![];
            }
        };
        self.seq_offset = tokens.len();
        let mut next_token = sample_top_p(&logits, temperature, top_p);
        let mut generated = vec![next_token];

        // Streaming UTF-8 byte buffer: joins multi-byte chars split across
        // byte-level tokens so emoji etc. aren't emitted as lossy `�`.
        let mut pending_bytes: Vec<u8> = Vec::new();

        // First token — call the callback before recording into anti-doom
        // so the detector sees the very first token in its history.
        if !self.emit_stream_token(next_token, &mut pending_bytes, &mut on_token) {
            return generated;
        }
        // Stop-token check: use the caller's stop_tokens if provided,
        // otherwise fall back to the engine's config.eos_token.
        let is_stop = if stop_tokens.is_empty() {
            next_token == self.config.eos_token
        } else {
            stop_tokens.contains(&next_token)
        };
        if is_stop {
            return generated;
        }

        for _ in 0..max_tokens - 1 {
            let mut logits = match self.forward_native(&[next_token]) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Forward pass failed: {}", e);
                    break;
                }
            };
            self.seq_offset += 1;

            next_token = sample_top_p(&logits, temperature, top_p);
            generated.push(next_token);

            if !self.emit_stream_token(next_token, &mut pending_bytes, &mut on_token) {
                break;
            }
            // Stop-token check: same as above — use caller's stop_tokens
            // when provided, else fall back to config.eos_token.
            let is_stop = if stop_tokens.is_empty() {
                next_token == self.config.eos_token
            } else {
                stop_tokens.contains(&next_token)
            };
            if is_stop {
                break;
            }
        }

        generated
    }

    /// Token-emit helper: decode one id to text, hand to the callback, and
    /// return whether generation should continue (callback decides).
    /// Token decoding swallows decode errors by returning empty string
    /// so a missing decode path never breaks streaming output.
    ///
    /// Uses a byte buffer so multi-byte UTF-8 chars (e.g. emoji) that split
    /// across byte-level tokens are joined before being handed to the
    /// callback, instead of each partial fragment being lossy-decoded to `�`.
    fn emit_stream_token<F>(&self, token_id: usize, pending: &mut Vec<u8>, cb: &mut F) -> bool
    where
        F: FnMut(usize, &str) -> bool,
    {
        if let Some(tok) = self.tokenizer_from_model() {
            pending.extend_from_slice(&tok.decode_bytes(&[token_id]));
        } else {
            // No native tokenizer — fall back to lossy per-token decode.
            let surface = self.decode(&[token_id]);
            return cb(token_id, &surface);
        }
        emit_complete_utf8(token_id, pending, cb)
    }

    /// Tokenize text using the model's native tokenizer (FFI path only).
    pub fn tokenize(&self, text: &str, add_special: bool) -> Vec<usize> {
        #[cfg(feature = "llama-ffi")]
        if let Some(ctx) = &self.ffi_context {
            return ctx.tokenize(text, add_special, true).into_iter().map(|t| t as usize).collect();
        }
        // Native fallback — use the GGUF tokenizer built from metadata.
        if let Some(tok) = self.tokenizer_from_model() {
            return tok.encode(text, add_special);
        }
        Vec::new()
    }

    /// Decode tokens to text using the model's native tokenizer (FFI path only).
    pub fn decode(&self, tokens: &[usize]) -> String {
        #[cfg(feature = "llama-ffi")]
        if let Some(ctx) = &self.ffi_context {
            return tokens.iter().map(|&t| ctx.token_to_piece(t as i32)).collect();
        }
        // Native fallback — use the GGUF tokenizer built from metadata.
        if let Some(tok) = self.tokenizer_from_model() {
            return tok.decode(tokens);
        }
        String::new()
    }

    /// Returns true if this engine uses the llama.cpp FFI backend.
    pub fn is_ffi(&self) -> bool {
        #[cfg(feature = "llama-ffi")]
        return self.ffi_context.is_some();
        #[cfg(not(feature = "llama-ffi"))]
        false
    }

    /// Public forward dispatch — uses FFI for Qwen, native for others.
    /// On the FFI path, llama.cpp errors are propagated rather than panicking
    /// (see audit fix for finding #7).
    pub fn forward(&mut self, tokens: &[usize]) -> Vec<f32> {
        #[cfg(feature = "llama-ffi")]
        if let Some(ctx) = &mut self.ffi_context {
            let tokens_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            return match ctx.forward(&tokens_i32) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("⚠️  FFI forward failed: {}", e);
                    vec![]
                }
            };
        }
        self.forward_native(tokens).unwrap_or_else(|e| {
            // TODO(audit-2026-07, finding #5): propagate Result to callers.
            // For now, log and continue with empty so the chat loop survives.
            eprintln!("Forward pass failed: {}", e);
            vec![]
        })
    }

    /// Hybrid forward pass supporting both standard transformers and SSM/Transformer hybrids.
    pub fn forward_native(&mut self, tokens: &[usize]) -> Result<Vec<f32>, String> {
        let seq_len = tokens.len();

        // Embedding lookup via mmap (avoids loading full embed matrix into RAM)
        let mut hidden = self.embed_lookup_mmap(tokens)?;

        // Transformer / hybrid layers — stream one layer at a time
        let is_gemma = self.gemma_layouts.len() == self.config.num_hidden_layers;

        // Gemma 3/4: scale token embeddings by sqrt(hidden_size) before the
        // first layer (matches llama.cpp's `inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd))`
        // in models/gemma4.cpp:14 and HuggingFace's Gemma3ForCausalLM).
        //
        // On 2026-06-29 we observed logits saturating the 30.0 soft-cap with
        // every TEMP=0 prompt, even ones with clean deterministic answers
        // (top-1 prediction distinguishes between visually-identical
        // Unicode variants of colons).  With per-layer dumps we found that
        // the FINAL HIDDEN STATE magnitude is L2 ≈ 770, which — multiplied
        // through the lm_head dot product with an embed-row of L2 ≈ 9.5 —
        // produces numerical magnitudes in the tens.  Reference Gemma 4
        // produces logits in the ±5 range, not 5–30.  The sqrt(n_embd)
        // embedding pre-scale appears to NOT be correctly compensated by
        // the model.norm + lm_head pipeline in this implementation, so we
        // Gemma 3/4: scale token embeddings by sqrt(hidden_size) before the
        // first layer (matches llama.cpp's `inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd))`
        // in models/gemma4.cpp:14 and HuggingFace's Gemma3ForCausalLM).
        if is_gemma {
            let scale = (self.config.hidden_size as f32).sqrt();
            for v in hidden.data.iter_mut() {
                *v *= scale;
            }
            if std::env::var("LEAFCUTTER_DEBUG_NORMS").is_ok() {
                let l2 = hidden.data.iter().map(|&v| v * v).sum::<f32>().sqrt();
                let max = hidden
                    .data
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let min = hidden.data.iter().cloned().fold(f32::INFINITY, f32::min);
                eprintln!(
                    "[NORM] emb_scaled               n={:>6}  l2={:>10.3}  min={:>12.4}  max={:>12.4}",
                    hidden.data.len(), l2, min, max
                );
            }
        }
        // Phase 2: per-layer async prefetch via std::thread::scope.
        //
        // load_layer() dequantizes Q4_K/Q6_K weights from mmap; on the 3B
        // model that's ~12 ms per layer / ~310 ms per pass / ~40% of wall.
        // We spawn `load_layer(layer_idx+1)` on a worker thread while the
        // main thread runs layer `layer_idx`'s matmul, so the next layer is
        // ready when we ask for it.
        //
        // Borrow mechanics:
        //   - `model_ref = &self.model` is a SHARED borrow of `self.model`.
        //   - In the scope body we still mutably touch `self.kv_cache`, etc.
        //   - These compile because Rust allows disjoint-field borrows:
        //     `self.model` and `self.kv_cache` are different fields, so the
        //     worker can hold `&self.model` while main borrows `&mut
        //     self.kv_cache` simultaneously.
        let model_ref = &self.model;
        let num_layers = self.config.num_hidden_layers;
        // Layer weights are served from a persistent cache (`get_layer`).
        // The first call per layer parses + dequantizes from the mmap; every
        // later token is a cache hit (cheap Arc clone). The prefetch worker
        // overlaps the FIRST token's layer loads with matmul; afterwards all
        // layers are already resident so prefetch is instant.
        //
        // Borrow mechanics:
        //   - `model_ref = &self.model` is a SHARED borrow of `self.model`.
        //   - In the scope body we still mutably touch `self.kv_cache`, etc.
        //   - These compile because Rust allows disjoint-field borrows:
        //     `self.model` and `self.kv_cache` are different fields, so the
        //     worker can hold `&self.model` while main borrows `&mut
        //     self.kv_cache` simultaneously.
        std::thread::scope(|scope| -> Result<(), String> {
            // Initial layer 0 (sync) + prefetch kick-off for layer 1.
            let mut layer_weights: std::sync::Arc<HashMap<String, Tensor>> = model_ref
                .get_layer(0)
                .map_err(|e| format!("layer 0 load: {}", e))?;
            // Prefetch loads the next layer's weights into RAM while computing
            // the current one — faster decode, but holds 2 layers resident.
            // Default OFF for an Ollama-like 1× footprint (Leafcutter is a
            // stateless tool).  Opt in with `LEAFCUTTER_PREFETCH=1`; the
            // available-RAM guard still protects tight hosts from `=1`.
            // Override: LEAFCUTTER_PREFETCH=1 to force on, =0 to force off.
            let use_prefetch = match std::env::var("LEAFCUTTER_PREFETCH").ok().as_deref() {
                Some("0") | Some("false") => false,
                Some("1") | Some("true") => true,
                _ => {
                    let total_ram = crate::detect::probe_hardware().ram_total_mb;
                    let model_mb = (self.model.file.file_size_bytes() / (1024 * 1024)) as u64;
                    total_ram >= model_mb
                }
            };
            let mut prefetch: Option<std::thread::ScopedJoinHandle<'_, Result<std::sync::Arc<HashMap<String, Tensor>>, String>>> =
                if use_prefetch && num_layers > 1 {
                    Some(scope.spawn(move || {
                        model_ref.get_layer(1).map_err(|e| format!("layer 1 load: {}", e))
                    }))
                } else { None };

            for layer_idx in 0..num_layers {
                if std::env::var("LEAFCUTTER_DEBUG_LAYERS").is_ok() {
                    let l2 = hidden.data.iter().map(|&v| v * v).sum::<f32>().sqrt();
                    let max = hidden.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let min = hidden.data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let nan_count = hidden.data.iter().filter(|&&v| v.is_nan()).count();
                    eprintln!(
                        "[LAYER {layer_idx:>2}] pre  n={:>6}  l2={:>10.3}  min={:>12.4}  max={:>12.4}  nan={}",
                        hidden.data.len(), l2, min, max, nan_count
                    );
                }
                if is_gemma {
                    // Gemma mutates layer weights in place (fuses QKV by
                    // materializing f32), so it gets its own owned map —
                    // never the shared Arc cache. Non-cached per token.
                    let mut gemma_weights: HashMap<String, Tensor> = model_ref
                        .load_layer(layer_idx)
                        .map_err(|e| format!("layer {} load: {}", layer_idx, e))?;
                    let cfg = &self.gemma_layouts[layer_idx];
                    let new_hidden = crate::inference::gemma::gemma_layer_forward(
                        &hidden,
                        &mut gemma_weights,
                        cfg,
                        &self.attn_params,
                        &mut self.kv_cache,
                        layer_idx,
                        self.seq_offset,
                        self.gemma_norm_eps,
                    )?;
                    hidden = new_hidden;
                } else {
                    // Detect layer type from actual tensor contents (most robust)
                    let has_standard_attn = layer_weights.contains_key("self_attn.q_proj.weight")
                        || layer_weights.contains_key("attn_q.weight");
                    let has_deltanet = layer_weights.contains_key("ssm_alpha.weight")
                        || layer_weights.contains_key("self_attn.qkv_proj.weight");
                    let has_ssm = layer_weights.contains_key("ssm_out.weight")
                        && !has_deltanet;
                    let has_mla = layer_weights.contains_key("attn_q_a.weight")
                        && layer_weights.contains_key("attn_kv_a_mqa.weight")
                        && layer_weights.contains_key("attn_q_b.weight")
                        && layer_weights.contains_key("attn_k_b.weight")
                        && layer_weights.contains_key("attn_v_b.weight");
                    let has_moe = layer_weights.contains_key("ffn_gate_inp.weight")
                        || layer_weights.contains_key("mlp.expert_gate.weight");
                    let has_shared_expert = layer_weights.contains_key("ffn_gate_shexp.weight")
                        || layer_weights.contains_key("mlp.shared_expert_gate.weight");

                    let pre_norm_weight = layer_weights
                        .get("input_layernorm.weight")
                        .or_else(|| layer_weights.get("attn_norm.weight"))
                        .ok_or_else(|| format!("layer {}: missing pre-norm (input_layernorm/attn_norm)", layer_idx))?;
                    let _t_pre = std::time::Instant::now();
                    let normed = hidden.rms_norm(pre_norm_weight, self.config.norm_eps);
                    if std::env::var("LEAFCUTTER_PROFILE").is_ok() {
                        eprintln!("[PROFILE] pre_norm               {:>8.2}ms", _t_pre.elapsed().as_secs_f32() * 1000.0);
                    }

                    if has_standard_attn {
                        let _t_attn = std::time::Instant::now();
                        let attn_out = attention_forward(&normed, &layer_weights, &self.attn_params, &mut self.kv_cache, layer_idx, self.seq_offset);
                        if std::env::var("LEAFCUTTER_PROFILE").is_ok() {
                            eprintln!("[PROFILE] attention_forward      {:>8.2}ms", _t_attn.elapsed().as_secs_f32() * 1000.0);
                        }
                        hidden = hidden.add(&attn_out);
                    } else if has_deltanet {
                        let _t_delta = std::time::Instant::now();
                        let deltanet_out = deltanet_forward(&normed, &layer_weights, &self.deltanet_params, &mut self.deltanet_cache, layer_idx);
                        if std::env::var("LEAFCUTTER_PROFILE").is_ok() {
                            eprintln!("[PROFILE] deltanet_forward      {:>8.2}ms", _t_delta.elapsed().as_secs_f32() * 1000.0);
                        }
                        hidden = hidden.add(&deltanet_out);
                    } else if has_ssm {
                        let _t_ssm = std::time::Instant::now();
                        let ssm_out = ssm_forward(&normed, &layer_weights, &self.ssm_config, &mut self.ssm_cache, layer_idx);
                        if std::env::var("LEAFCUTTER_PROFILE").is_ok() {
                            eprintln!("[PROFILE] ssm_forward            {:>8.2}ms", _t_ssm.elapsed().as_secs_f32() * 1000.0);
                        }
                        hidden = hidden.add(&ssm_out);
                    } else if has_mla {
                        let _t_mla = std::time::Instant::now();
                        let mla_out = mla_forward(&normed, &layer_weights, &self.mla_params, &mut self.kv_cache, layer_idx, self.seq_offset);
                        if std::env::var("LEAFCUTTER_PROFILE").is_ok() {
                            eprintln!("[PROFILE] mla_forward            {:>8.2}ms", _t_mla.elapsed().as_secs_f32() * 1000.0);
                        }
                        hidden = hidden.add(&mla_out);
                    }

                    let post_norm_weight = layer_weights
                        .get("post_attention_layernorm.weight")
                        .or_else(|| layer_weights.get("post_attention_norm.weight"))
                        .or_else(|| layer_weights.get("ffn_norm.weight"))
                        .ok_or_else(|| format!("layer {}: missing post-norm (post_attention_layernorm/_norm/ffn_norm)", layer_idx))?;
                    let _t_post = std::time::Instant::now();
                    let normed = hidden.rms_norm(post_norm_weight, self.config.norm_eps);
                    if std::env::var("LEAFCUTTER_PROFILE").is_ok() {
                        eprintln!("[PROFILE] post_norm              {:>8.2}ms", _t_post.elapsed().as_secs_f32() * 1000.0);
                    }
                    let _t_ffn = std::time::Instant::now();
                    let ffn_out = if has_moe {
                        Self::ffn_moe_forward(&normed, &layer_weights, &self.moe_params, has_shared_expert)?
                    } else {
                        Self::ffn_forward(&normed, &layer_weights)?
                    };
                    if std::env::var("LEAFCUTTER_PROFILE").is_ok() {
                        eprintln!("[PROFILE] ffn_forward            {:>8.2}ms", _t_ffn.elapsed().as_secs_f32() * 1000.0);
                    }
                    hidden = hidden.add(&ffn_out);
                }

                // ── Common tail: swap to prefetched next (if any).
                // Keep the Arc alive so the cache retains the weights; the
                // next iteration reuses them via `get_layer`. The old Arc
                // reference is released when we reassign below.

                // MADV_DONTNEED on the whole mmap is NOT applied per layer:
                // evicting pages every layer would force disk re-reads while
                // the cache is still cold. Instead the pages are dropped ONCE
                // after the first forward pass (see end of this fn), when every
                // layer is already resident in the RAM layer cache.

                if let Some(h) = prefetch.take() {
                    layer_weights = h.join()
                        .map_err(|_| "worker panicked".to_string())
                        .and_then(|r| r)?;

                    // Schedule the layer after next (2 ahead of the once
                    // we're about to enter).  Iteration N will use this.
                    let next_layer_idx = layer_idx + 2;
                    if next_layer_idx < num_layers {
                        prefetch = Some(scope.spawn(move || {
                            model_ref.get_layer(next_layer_idx)
                                .map_err(|e| format!("layer {} load: {}", next_layer_idx, e))
                        }));
                    }
                } else if layer_idx + 1 < num_layers {
                    // No prefetch available: load synchronously (env var NOT set
                    // path).  The `else if` matches single-layer models.
                    layer_weights = model_ref
                        .get_layer(layer_idx + 1)
                        .map_err(|e| format!("layer {} load: {}", layer_idx + 1, e))?;
                }
            }
            Ok(())
        })?;

        // ── One-shot mmap page release ──────────────────────────────────
        // After the first forward pass every layer's weights live in the RAM
        // layer cache, so the file-backed mmap pages are a redundant second
        // copy. Release them once (MADV_DONTNEED) to free ~model-size RSS.
        // The embed table + lm_head rows still fault in on demand (~KB/token).
        //
        // Only safe when the FULL model is resident in the layer cache.  For
        // huge models that exceed the cache budget, evicted layers stream from
        // the mmap every token, so we must keep the kernel page cache warm.
        // Skipped when the layer cache is disabled (LEAFCUTTER_NO_CACHE=1) —
        // in that mode weights are re-read from the mmap every token.
        let cache_enabled = std::env::var("LEAFCUTTER_NO_CACHE").map(|v| v != "1").unwrap_or(true);
        if !self.pages_dropped && cache_enabled && self.model.all_layers_cached() {
            self.model.file.drop_pages_from_cache();
            self.pages_dropped = true;
        }

        // Final norm        // Final norm        // Final norm
        let final_norm = self
            .special_weights
            .get("model.norm.weight")
            .ok_or_else(|| "missing model.norm.weight (final norm)".to_string())?;
        hidden = hidden.rms_norm(final_norm, self.config.norm_eps);
        if std::env::var("LEAFCUTTER_DEBUG_NORMS").is_ok() {
            let l2 = hidden.data.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let max = hidden.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = hidden.data.iter().cloned().fold(f32::INFINITY, f32::min);
            eprintln!("[NORM] final_norm                n={:>6}  l2={:>10.3}  min={:>12.4}  max={:>12.4}",
                hidden.data.len(), l2, min, max);
        }

        // LM head — computed via outer-product over rows from mmap (no full matrix in RAM)
        let mut logits = if self.lm_head_tied {
            self.lm_head_tied_forward(&hidden, seq_len)
        } else {
            self.lm_head_separate_forward(&hidden, seq_len)
        };

        // Gemma logit soft-capping: prevents extreme logits
        let cap = self.config.logit_soft_cap;
        if cap > 0.0 {
            for logit in logits.iter_mut() {
                *logit = cap * (*logit / cap).tanh();
            }
        }

        Ok(logits)
    }

    /// Forward pass with per-layer RSS debugging.
    pub fn forward_debug(&mut self, tokens: &[usize]) -> Vec<f32> {
        fn read_rss_kb() -> usize {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if let Some(Ok(v)) = parts.get(1).map(|s| s.parse::<usize>()) {
                            return v;
                        }
                    }
                }
            }
            0
        }

        let seq_len = tokens.len();
        let mut hidden = match self.embed_lookup_mmap(tokens) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("embed_lookup_mmap failed: {}", e);
                return vec![];
            }
        };
        println!("   [embed] RSS: {} MB", read_rss_kb() / 1024);

        for layer_idx in 0..self.config.num_hidden_layers {
            let layer_weights = self.model.load_layer(layer_idx)
                .expect("Failed to load layer");
            let rss_after_load = read_rss_kb();

            let has_standard_attn = layer_weights.contains_key("self_attn.q_proj.weight")
                || layer_weights.contains_key("attn_q.weight");
            let has_deltanet = layer_weights.contains_key("ssm_alpha.weight")
                || layer_weights.contains_key("self_attn.qkv_proj.weight");
            let has_ssm = layer_weights.contains_key("ssm_out.weight")
                && !has_deltanet;

            let pre_norm_weight = layer_weights.get("input_layernorm.weight")
                .or_else(|| layer_weights.get("attn_norm.weight"))
                .expect("Missing pre-norm");
            let normed = hidden.rms_norm(pre_norm_weight, self.config.norm_eps);

            if has_standard_attn {
                let attn_out = attention_forward(&normed, &layer_weights, &self.attn_params, &mut self.kv_cache, layer_idx, self.seq_offset);
                hidden = hidden.add(&attn_out);
            } else if has_deltanet || has_ssm {
                let ssm_out = ssm_forward(&normed, &layer_weights, &self.ssm_config, &mut self.ssm_cache, layer_idx);
                hidden = hidden.add(&ssm_out);
            }

            let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
                .or_else(|| layer_weights.get("ffn_norm.weight"))
                .expect("Missing post-norm");
            let normed = hidden.rms_norm(post_norm_weight, self.config.norm_eps);
            // forward_debug is a diagnostic-only path — keep the old panic on
            // missing FFN weights to surface inconsistencies loudly.
            let ffn_out = Self::ffn_forward(&normed, &layer_weights)
                .expect("forward_debug: ffn_forward failed");
            hidden = hidden.add(&ffn_out);

            let rss_after_layer = read_rss_kb();
            if layer_idx < 3 || layer_idx == self.config.num_hidden_layers - 1 {
                println!("   [layer {:>2}] after load: {:>5} MB | after compute: {:>5} MB | delta: {:>4} MB",
                    layer_idx, rss_after_load / 1024, rss_after_layer / 1024,
                    (rss_after_layer as i64 - rss_after_load as i64) / 1024);
            }
        }

        let final_norm = self.special_weights.get("model.norm.weight")
            .expect("Missing final norm");
        hidden = hidden.rms_norm(final_norm, self.config.norm_eps);
        println!("   [final norm] RSS: {} MB", read_rss_kb() / 1024);

        let mut logits = if self.lm_head_tied {
            self.lm_head_tied_forward(&hidden, seq_len)
        } else {
            self.lm_head_separate_forward(&hidden, seq_len)
        };
        let cap = self.config.logit_soft_cap;
        if cap > 0.0 {
            for logit in logits.iter_mut() {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        println!("   [lm_head] RSS: {} MB", read_rss_kb() / 1024);
        logits
    }

    /// Legacy embed lookup from a fully-materialized tensor (for tests / compatibility).
    pub fn embed_lookup(&self, tokens: &[usize], embed: &Tensor) -> Tensor {
        let hidden_size = embed.shape[1];
        let mut data = vec![0.0f32; tokens.len() * hidden_size];
        for (i, &token) in tokens.iter().enumerate() {
            if token < embed.shape[0] {
                data[i * hidden_size..(i + 1) * hidden_size]
                    .copy_from_slice(&embed.data[token * hidden_size..(token + 1) * hidden_size]);
            }
        }
        Tensor::from_vec(data, vec![tokens.len(), hidden_size])
    }

    /// Lookup embeddings via on-demand mmap per-row dequantization.
    /// Each token reads exactly one row from the quantized embedding table —
    /// ~3 KB for Q4_K instead of 1.5 GB for the full f32 matrix.
    pub fn embed_lookup_mmap(&self, tokens: &[usize]) -> Result<Tensor, String> {
        let hidden_size = self.config.hidden_size;

        // Determine the actual embedding table dimensions from the file,
        // not from config.vocab_size (which can be 0 when the tokenizer
        // metadata is missing).  Bounds-checking against 0 would let
        // every token through and produce OOB reads on `embed.data`.
        let embed_info = self.model.file.get_tensor_info("token_embd.weight")
            .ok_or_else(|| "Missing token_embd.weight tensor".to_string())?;
        let embed_dim: usize = embed_info.dimensions.first().copied().unwrap_or(hidden_size as u64) as usize;
        let embed_rows: usize = embed_info.dimensions.get(1).copied().unwrap_or(0) as usize;
        if embed_rows == 0 {
            return Err("token_embd.weight has 0 rows".to_string());
        }

        let mut data = vec![0.0f32; tokens.len() * hidden_size];
        for (i, &token) in tokens.iter().enumerate() {
            if token >= embed_rows {
                return Err(format!(
                    "Token ID {} is out of bounds (embed table has {} rows, config.vocab_size={})",
                    token, embed_rows, self.config.vocab_size
                ));
            }
            let row = self.model.file.get_tensor_row_f32("token_embd.weight", token)
                .ok_or_else(|| format!("Failed to read embedding row {}", token))?;
            // Embedding dim may differ from hidden_size (e.g. Ministral-3B: 3072 vs 4096)
            let copy_len = row.len().min(hidden_size).min(embed_dim);
            data[i * hidden_size..i * hidden_size + copy_len].copy_from_slice(&row[..copy_len]);
            // Remaining dims stay zero (padding)
        }
        Ok(Tensor::from_vec(data, vec![tokens.len(), hidden_size]))
    }

    /// Compute logits when lm_head is tied to embeddings.
    /// Parallelized over the vocabulary using rayon.
    fn lm_head_tied_forward(&self, hidden: &Tensor, seq_len: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let hidden_last = &hidden.data[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        // For Gemma, the embedding pre-scale applied by `forward_native`
        // (`hidden *= sqrt(n_embd)`) carries all the way to the lm_head,
        // inflating logits by `sqrt(n_embd)` (≈ 62× on a 3840-wide model).
        // Reverting it here brings logits back into the reference's
        // [−5, +5] range and lets the softcap stop masking the
        // distribution.
        let inv = if self.embed_uses_sqrt_scale {
            1.0 / (hidden_size as f32).sqrt()
        } else {
            1.0
        };
        let logits = if inv != 1.0 {
            let mut scaled = hidden_last.to_vec();
            for v in scaled.iter_mut() {
                *v *= inv;
            }
            self.lm_head_projection(&scaled, "token_embd.weight", hidden_size, vocab_size)
        } else {
            self.lm_head_projection(hidden_last, "token_embd.weight", hidden_size, vocab_size)
        };
        logits
    }

    /// Compute logits when lm_head is a separate tensor (output.weight).
    /// The raw GGUF data is [vocab, hidden] (before transpose), so each row
    /// corresponds to one vocab token's weights.  We compute dot(hidden, row_j)
    /// for each token j — same pattern as tied embeddings.
    fn lm_head_separate_forward(&self, hidden: &Tensor, seq_len: usize) -> Vec<f32> {
        let profile = std::env::var("LEAFCUTTER_PROFILE").is_ok();
        let _t0 = if profile { Some(std::time::Instant::now()) } else { None };
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let hidden_last = &hidden.data[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        let inv = if self.embed_uses_sqrt_scale {
            1.0 / (hidden_size as f32).sqrt()
        } else {
            1.0
        };
        let logits = if inv != 1.0 {
            let mut scaled = hidden_last.to_vec();
            for v in scaled.iter_mut() {
                *v *= inv;
            }
            self.lm_head_projection(&scaled, "output.weight", hidden_size, vocab_size)
        } else {
            self.lm_head_projection(hidden_last, "output.weight", hidden_size, vocab_size)
        };
        if let Some(t0) = _t0 {
            eprintln!("[PROFILE] lm_head_separate_forward: {:.3} ms (vocab={})",
                t0.elapsed().as_secs_f64() * 1000.0, vocab_size);
        }
        logits
    }

    /// LM head projection: dot(hidden_last, embed_row) for each token.
    /// Uses cached dequantized weights when available (Engine::load time).
    /// Falls back to row-by-row mmap dequant for models loaded without cache.
    fn lm_head_projection(&self, hidden_last: &[f32], tensor_name: &str, hidden_size: usize, vocab_size: usize) -> Vec<f32> {
        // Use cached lm_head Q6_K blocks if available (loaded at Engine::load
        // time).  The GEMM dequantizes blocks on the fly, avoiding both the
        // ~3.8 GB f32 cache and the per-row mmap path.
        if let Some(ref mat) = self.cached_lm_head {
            let mut logits = vec![0.0f32; vocab_size];
            crate::kernels::q6_k_gemm::q6_k_matmul_transposed_b(
                hidden_last,
                mat,
                &mut logits,
                1,
                hidden_size,
                vocab_size,
            );
            return logits;
        }

        // Fallback: row-by-row mmap dequant (original slow path).
        use rayon::prelude::*;
        let file = &self.model.file;
        (0..vocab_size).into_par_iter().map(|token_id| {
            thread_local! {
                static BUF: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
                static CAP: std::cell::Cell<usize> = std::cell::Cell::new(0);
            }
            BUF.with(|buf| {
                let mut buf = buf.borrow_mut();
                let cap = CAP.with(|c| c.get());
                if buf.capacity() < hidden_size {
                    buf.resize(hidden_size, 0.0);
                    CAP.with(|c| c.set(buf.capacity()));
                } else if cap != hidden_size {
                    buf.resize(hidden_size, 0.0);
                    CAP.with(|c| c.set(buf.capacity()));
                }
                if file.get_tensor_row_f32_into(tensor_name, token_id, &mut buf).is_none() {
                    0.0
                } else {
                    crate::kernels::simd::simd_dot_product(hidden_last, &buf[..hidden_size])
                }
            })
        }).collect()
    }

    /// Format a chat prompt based on the model's special tokens.
    /// Auto-detects Llama-3, Qwen2.5/Qwen3 (ChatML `<|im_start|>`), or
    /// raw-text by default for Ornith-family models.
    pub fn format_chat_prompt(&self, system: &str, user: &str) -> String {
        let tok = self.tokenizer_from_model();
        let has_llama3 = tok.as_ref()
            .map(|t| t.token_to_id.contains_key("<|start_header_id|>"))
            .unwrap_or(false);
        let has_qwen = tok.as_ref()
            .map(|t| t.token_to_id.contains_key("<|im_start|>"))
            .unwrap_or(false);

        if has_llama3 {
            format!(
                "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\
                 <|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n",
                system, user
            )
        } else if has_qwen {
            format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                system, user
            )
        } else {
            // Plain text fallback (used for Ornith 1.0 9B which ships
            // a custom non-ChatML tokenizer).  Pass through verbatim.
            user.to_string()
        }
    }

    /// Build a GgufTokenizer from the model's embedded vocab metadata.
    /// Lazily build the GGUF vocab tokenizer, but cache the result for
    /// subsequent calls.  Vocabulary extraction walks the entire token
    /// array (~50–500 KB of HashMap construction), so doing it per
    /// `generate_text` invocation (which previously happened on every
    /// `tokenize` call) was a measurable hot-path cost.
    pub fn tokenizer_from_model(&self) -> Option<GgufTokenizer> {
        if let Ok(guard) = self.cached_tokenizer.lock() {
            if let Some(tok) = guard.as_ref() {
                return Some(tok.clone());
            }
        }
        let built = self.build_tokenizer_from_model()?;
        if let Ok(mut guard) = self.cached_tokenizer.lock() {
            *guard = Some(built.clone());
        }
        Some(built)
    }

    fn build_tokenizer_from_model(&self) -> Option<GgufTokenizer> {
        let file = &self.model.file;
        let tokens = file.metadata.get("tokenizer.ggml.tokens")?;
        let vocab: Vec<String> = if let crate::model::gguf::GGUFValue::Array(arr) = tokens {
            arr.iter()
                .filter_map(|v| {
                    if let crate::model::gguf::GGUFValue::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            return None;
        };
        if vocab.is_empty() {
            return None;
        }

        // Read BPE merge rules if available
        let merges: Vec<String> = match file.metadata.get("tokenizer.ggml.merges") {
            Some(crate::model::gguf::GGUFValue::Array(arr)) => arr
                .iter()
                .filter_map(|v| {
                    if let crate::model::gguf::GGUFValue::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        };

        Some(GgufTokenizer::from_vocab_and_merges(vocab, merges))
    }
    /// Generate text from a string prompt using the embedded GGUF tokenizer.
    pub fn generate_text(&mut self, prompt: &str, max_tokens: usize, temperature: f32, top_p: f32) -> Option<String> {
        let tok = self.tokenizer_from_model()?;
        let token_ids = tok.encode(prompt, true);
        let generated_ids = self.generate(&token_ids, max_tokens, temperature, top_p);

        // Skip BOS and stop at EOS
        let clean_ids: Vec<usize> = generated_ids.iter()
            .skip_while(|&&id| Some(id) == tok.bos_id())
            .take_while(|&&id| Some(id) != tok.eos_id())
            .copied()
            .collect();

        Some(tok.decode(&clean_ids))
    }

    pub fn ffn_forward(x: &Tensor, weights: &HashMap<String, Tensor>) -> Result<Tensor, String> {
        let gate = weights
            .get("mlp.gate_proj.weight")
            .ok_or_else(|| "ffn_forward: missing mlp.gate_proj.weight".to_string())?;
        let up = weights
            .get("mlp.up_proj.weight")
            .ok_or_else(|| "ffn_forward: missing mlp.up_proj.weight".to_string())?;
        let down = weights
            .get("mlp.down_proj.weight")
            .ok_or_else(|| "ffn_forward: missing mlp.down_proj.weight".to_string())?;

        let _t_gate = std::time::Instant::now();
        let gate_proj = x.matmul(gate);
        let t_gate = _t_gate.elapsed().as_secs_f32() * 1000.0;
        let _t_up = std::time::Instant::now();
        let up_proj = x.matmul(up);
        let t_up = _t_up.elapsed().as_secs_f32() * 1000.0;
        let _t_silu = std::time::Instant::now();
        let activated = gate_proj.silu();
        let t_silu = _t_silu.elapsed().as_secs_f32() * 1000.0;
        let _t_fused = std::time::Instant::now();
        let mut fused = vec![0.0f32; activated.size()];
        crate::kernels::simd::simd_vec_mul(&activated.data, &up_proj.data, &mut fused);
        let fused_tensor = Tensor::from_vec(fused, activated.shape.clone());
        let t_fused = _t_fused.elapsed().as_secs_f32() * 1000.0;
        let _t_down = std::time::Instant::now();
        let out = fused_tensor.matmul(down);
        let t_down = _t_down.elapsed().as_secs_f32() * 1000.0;
        if std::env::var("LEAFCUTTER_PROFILE").is_ok() {
            eprintln!("[PROFILE]   ffn gate={:.3} up={:.3} silu={:.3} fusedmul={:.3} down={:.3}",
                t_gate, t_up, t_silu, t_fused, t_down);
        }
        Ok(out)
    }

    /// MoE FFN forward (DeepSeek-2 / GLM-DSA).  Per-token, top-k routing
    /// over routed experts + shared expert branch.  Driven by
    /// `crate::inference::moe::moe_forward_one_token`.
    ///
    /// Routed `*_exps.weight` tensors stay resident as 3-D QUANTIZED
    /// tensors (`[d0, d1, num_experts]`); each active expert is sliced out
    /// on demand inside the MoE module via `Tensor::expert_slice` (no f32
    /// materialization of the full 3-D tensor).  Shared expert weights are
    /// 2-D and pass through unchanged.
    pub fn ffn_moe_forward(
        x: &Tensor,
        weights: &HashMap<String, Tensor>,
        cfg: &crate::inference::moe::MoeConfig,
        has_shared_expert: bool,
    ) -> Result<Tensor, String> {
        let _ = has_shared_expert; // reserved for later exp_probs_b wiring
        Ok(crate::inference::moe::moe_forward(x, weights, cfg))
    }

    /// Get engine info for diagnostics
    pub fn info(&self) -> EngineInfo {
        EngineInfo {
            architecture: self.model.architecture.name().to_string(),
            total_layers: self.config.num_hidden_layers,
            hidden_size: self.config.hidden_size,
            kv_cache_tokens: self.kv_cache.total_seq_len(),
            kv_cache_bytes: self.kv_cache.memory_bytes(),
            use_ssm: self.model.architecture.uses_ssm(),
            use_fused_qkv: self.attn_params.use_fused_qkv,
            use_compressed_kv: self.config.kv_head_dim != self.config.head_dim,
            use_speculative: self.speculative_head.is_some(),
        }
    }
}

#[derive(Debug)]
pub struct EngineInfo {
    pub architecture: String,
    pub total_layers: usize,
    pub hidden_size: usize,
    pub kv_cache_tokens: usize,
    pub kv_cache_bytes: usize,
    pub use_ssm: bool,
    pub use_fused_qkv: bool,
    pub use_compressed_kv: bool,
    pub use_speculative: bool,
}

// -------------------------------------------------------------------------
// Gemma per-layer layouts (Gemma 3/4 family only)
// -------------------------------------------------------------------------
/// Drain complete UTF-8 sequences from a streaming byte buffer, handing each
/// decoded character (or run of them) to the callback.  Incomplete trailing
/// bytes stay buffered for the next token.  Returns false if the callback
/// asked generation to stop.
fn emit_complete_utf8<F>(token_id: usize, pending: &mut Vec<u8>, cb: &mut F) -> bool
where
    F: FnMut(usize, &str) -> bool,
{
    let mut complete_len = 0usize;
    let mut i = 0usize;
    while i < pending.len() {
        let b = pending[i];
        let len = if b < 0x80 {
            1
        } else if b >= 0xC2 && b <= 0xDF {
            2
        } else if b >= 0xE0 && b <= 0xEF {
            3
        } else if b >= 0xF0 && b <= 0xF4 {
            4
        } else {
            // Invalid leading byte; treat as complete (lossy anyway).
            1
        };
        if i + len > pending.len() {
            break;
        }
        i += len;
        complete_len = i;
    }

    if complete_len == 0 {
        return true;
    }
    let text = String::from_utf8_lossy(&pending[..complete_len]).to_string();
    pending.drain(..complete_len);
    cb(token_id, &text)
}

fn infer_gemma_layouts(
    model: &GGUFModel,
    arch: &ModelArchitecture,
) -> Vec<crate::inference::gemma::GemmaLayerParams> {
    use crate::inference::gemma::GemmaLayerParams;
    use crate::model::gguf::GGUFValue;
    if !matches!(arch, ModelArchitecture::Gemma) {
        return Vec::new();
    }
    let num_layers = match arch.metadata_prefix() {
        p => model
            .file
            .get_metadata_int(&format!("{}.block_count", p))
            .or_else(|| model.file.get_metadata_int(&format!("{}.block_count", p.replace("gemma", "gemma4"))))
            .or_else(|| model.file.get_metadata_int(&format!("{}.block_count", p.replace("gemma", "gemma3"))))
            .or_else(|| model.file.get_metadata_int(&format!("{}.block_count", p.replace("gemma", "gemma2"))))
            .or_else(|| model.file.get_metadata_int("num_hidden_layers"))
            .unwrap_or(0) as usize,
    };
    if num_layers == 0 {
        return Vec::new();
    }
    // Default head_dims.  Gemma 4 sets both, with sliding layers using the
    // smaller of the two.
    let key_length = model
        .file
        .get_metadata_int("gemma4.attention.key_length")
        .unwrap_or(256) as usize;
    let value_length = model
        .file
        .get_metadata_int("gemma4.attention.value_length")
        .unwrap_or(256) as usize;
    let key_length_swa = model
        .file
        .get_metadata_int("gemma4.attention.key_length_swa")
        .unwrap_or(key_length as i64) as usize;
    let value_length_swa = model
        .file
        .get_metadata_int("gemma4.attention.value_length_swa")
        .unwrap_or(value_length as i64) as usize;
    let rope_theta_global = model
        .file
        .get_metadata_f32("gemma4.rope.freq_base")
        .unwrap_or(1_000_000.0);
    let rope_theta_swa = model
        .file
        .get_metadata_f32("gemma4.rope.freq_base_swa")
        .unwrap_or(10_000.0);
    // Per-layer head_count_kv (i32 array).
    let head_count_kv: Vec<i64> = match model.file.metadata.get("gemma4.attention.head_count_kv") {
        Some(GGUFValue::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                GGUFValue::I32(i) => *i as i64,
                GGUFValue::I64(i) => *i,
                _ => 0,
            })
            .collect(),
        _ => vec![8_i64; num_layers],
    };
    // Per-layer sliding-window pattern (bool array).
    // Gemma 3/4 default: 5 global then 1 sliding, repeating.
    let mut default_pattern: Vec<bool> = (0..num_layers).map(|i| i % 6 == 5).collect();
    let swa_pattern: Vec<bool> = match model
        .file
        .metadata
        .get("gemma4.attention.sliding_window_pattern")
    {
        Some(GGUFValue::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                GGUFValue::Bool(b) => *b,
                _ => false,
            })
            .collect(),
        _ => default_pattern.clone(),
    };
    let _ = default_pattern;
    let head_count_kv = head_count_kv;

    // Per-layer Q/K projection output dimensions, derived from actual tensor
    // shapes.  Gemma 4 metadata `gemma4.attention.key_length{,swa}` is wrong
    // (it reports the RoPE dim, not the head_dim of K/V projections), so we
    // build the per-layer params by introspecting the GGUF tensors instead.
    let mut q_proj_out: Vec<usize> = vec![4096; num_layers];
    let mut k_proj_out: Vec<usize> = vec![2048; num_layers];
    for i in 0..num_layers {
        if let Some(info) = model
            .file
            .get_tensor_info(&format!("blk.{}.attn_q.weight", i))
        {
            q_proj_out[i] = info.dimensions[1] as usize;
        }
        if let Some(info) = model
            .file
            .get_tensor_info(&format!("blk.{}.attn_k.weight", i))
        {
            k_proj_out[i] = info.dimensions[1] as usize;
        }
    }

    (0..num_layers)
        .map(|i| {
            let kv = head_count_kv.get(i).copied().unwrap_or(8) as usize;
            let is_swa = swa_pattern.get(i).copied().unwrap_or(false);
            // Gemma 4 has 16 Q-heads always.  Each Q-head contains a Q
            // vector; sliding layers also fold V into the *second half*
            // of the same Q-head, so the total Q_proj_out = num_heads * head_dim
            // for global layers and  num_heads * 2 * head_dim for sliding.
            let num_heads: usize = 16;
            let q_per_head = (if is_swa { 2 } else { 1 }) * num_heads;
            let q_total = *q_proj_out.get(i).unwrap_or(&4096);
            let k_total = *k_proj_out.get(i).unwrap_or(&2048);
            let q_head_dim = if q_per_head > 0 {
                q_total / q_per_head
            } else {
                256
            };
            let k_head_dim = if kv > 0 { k_total / kv } else { q_head_dim };
            GemmaLayerParams {
                num_kv_heads: kv,
                q_head_dim,
                k_head_dim,
                v_head_dim: q_head_dim,
                is_global: !is_swa,
                rope_theta: if is_swa {
                    rope_theta_swa
                } else {
                    rope_theta_global
                },
            }
        })
        .collect()
}

// -------------------------------------------------------------------------
// LM head cache
// -------------------------------------------------------------------------

/// Load the lm_head tensor (`output.weight`, or `token_embd.weight` when
/// tied) and keep it in its native Q6_K block form, once, at load time.
///
/// The Q6_K GEMM dequantizes blocks on the fly, so this needs only the raw
/// block bytes (~0.8 GB for a 248K×4096 vocab) instead of a ~3.8 GB f32
/// dequant cache, while remaining ~2× faster than the per-token mmap
/// row-by-row path.  Matches llama.cpp's lm_head layout.
///
/// Returns `None` if the tensor is missing or its quant type is not Q6_K
/// (caller then falls back to the mmap row-by-row path).
fn load_lm_head_cache(model: &GGUFModel, tensor_name: &str) -> Option<crate::kernels::q6_k::Matrix> {
    let info = model.file.get_tensor_info(tensor_name)?;
    if info.typ != crate::model::quant::QuantType::Q6_K.code() {
        return None;
    }
    let raw = model.file.get_tensor_raw(tensor_name)?;
    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    // GGUF stores a matrix as [inner_dim, outer_dim]; for lm_head that's
    // [hidden_size, vocab_size].  Q6_K blocks are row-major over the outer
    // (vocab) dim, exactly the layout q6_k_matmul_transposed_b expects
    // (b.rows = vocab, b.cols = hidden).
    if shape.len() != 2 {
        return None;
    }
    let hidden_size = shape[0];
    let vocab_size = shape[1];
    let matrix = crate::kernels::q6_k::Matrix {
        rows: vocab_size,
        cols: hidden_size,
        blocks: crate::kernels::q6_k::blocks_from_bytes(raw),
    };
    if std::env::var("LEAFCUTTER_DEBUG").map(|v| v == "1").unwrap_or(false) {
        let bytes = matrix.blocks.len() * crate::kernels::q6_k::Block::BYTES;
        eprintln!(
            "[lm_head] cached '{}' as Q6_K ({}x{}, {:.2} MiB)",
            tensor_name,
            matrix.rows,
            matrix.cols,
            bytes as f64 / 1024.0 / 1024.0
        );
    }
    Some(matrix)
}
