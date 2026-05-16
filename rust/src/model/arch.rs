//! Model architecture auto-detection and layer-mapping dispatch.
//!
//! Different model families (Llama, Qwen, Mistral, etc.) store weights
//! under different GGUF key prefixes and use slightly different layer
//! structures.  This module centralises the detection and mapping so
//! adding a new architecture is a single enum variant + config.

use super::gguf::{GGUFile, GGUFValue};

/// Supported / detected model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    Llama,
    Qwen2,
    Qwen35,
    Mistral,
    Phi,
    Gemma,
    Unknown,
}

impl ModelArchitecture {
    /// Detect architecture from GGUF metadata.
    pub fn detect(file: &GGUFile) -> Self {
        if let Some(GGUFValue::String(arch)) = file.metadata.get("general.architecture") {
            match arch.as_str() {
                "llama"   => ModelArchitecture::Llama,
                "qwen2"   => ModelArchitecture::Qwen2,
                "qwen35"  => ModelArchitecture::Qwen35,
                "mistral" => ModelArchitecture::Mistral,
                "phi"     => ModelArchitecture::Phi,
                "gemma"   => ModelArchitecture::Gemma,
                _         => ModelArchitecture::Unknown,
            }
        } else {
            ModelArchitecture::Unknown
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            ModelArchitecture::Llama   => "Llama",
            ModelArchitecture::Qwen2   => "Qwen2",
            ModelArchitecture::Qwen35  => "Qwen3.5",
            ModelArchitecture::Mistral => "Mistral",
            ModelArchitecture::Phi     => "Phi",
            ModelArchitecture::Gemma   => "Gemma",
            ModelArchitecture::Unknown => "Unknown",
        }
    }

    /// Metadata key prefix used for hyper-parameters.
    pub fn metadata_prefix(self) -> &'static str {
        match self {
            ModelArchitecture::Llama   => "llama",
            ModelArchitecture::Qwen2   => "qwen2",
            ModelArchitecture::Qwen35  => "qwen35",
            ModelArchitecture::Mistral => "llama", // Mistral uses llama.* keys
            ModelArchitecture::Phi     => "phi",
            ModelArchitecture::Gemma   => "gemma",
            ModelArchitecture::Unknown => "llama", // best-effort fallback
        }
    }

    /// Whether the full inference stack supports this architecture.
    ///
    /// "Supported" here means: we have the correct layer weight mappings
    /// and attention/FFN structure implemented.
    pub fn is_supported(self) -> bool {
        matches!(self,
            ModelArchitecture::Llama |
            ModelArchitecture::Qwen2 |
            ModelArchitecture::Mistral
        )
    }

    /// Standard transformer layer weight mappings for this architecture.
    ///
    /// Returns `(gguf_suffix, engine_name)` pairs.  The loader will look
    /// for `blk.N.{gguf_suffix}` in the GGUF and map it to the engine
    /// key `{engine_name}`.
    pub fn layer_mappings(self) -> &'static [(&'static str, &'static str)] {
        // Default standard transformer mapping
        &[
            ("attn_q.weight",     "self_attn.q_proj.weight"),
            ("attn_k.weight",     "self_attn.k_proj.weight"),
            ("attn_v.weight",     "self_attn.v_proj.weight"),
            ("attn_output.weight","self_attn.o_proj.weight"),
            ("ffn_gate.weight",   "mlp.gate_proj.weight"),
            ("ffn_up.weight",     "mlp.up_proj.weight"),
            ("ffn_down.weight",   "mlp.down_proj.weight"),
            ("attn_norm.weight",  "input_layernorm.weight"),
            ("ffn_norm.weight",   "post_attention_layernorm.weight"),
        ]
    }

    /// Extra weight suffixes that may appear in this architecture but
    /// are not part of the standard mapping.  Used for capability reports.
    pub fn known_extra_suffixes(self) -> &'static [&'static str] {
        match self {
            ModelArchitecture::Qwen35 => &[
                "attn_gate.weight",
                "attn_qkv.weight",
                "post_attention_norm.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "attn_q.weight",
                "attn_v.weight",
            ],
            ModelArchitecture::Llama => &[
                "attn_norm.bias",
                "ffn_norm.bias",
            ],
            _ => &[],
        }
    }

    /// Expected RMS norm epsilon (or LayerNorm epsilon for non-RMS).
    pub fn default_norm_eps(self) -> f32 {
        1e-5
    }
}

/// Pre-flight capability report for a GGUF model.
#[derive(Debug)]
pub struct CapabilityReport {
    pub architecture: ModelArchitecture,
    pub arch_supported: bool,
    pub quant_summary: super::quant::QuantSummary,
    pub missing_tensors: Vec<String>,
    pub extra_tensors: Vec<String>,
    pub can_run: bool,
}

impl CapabilityReport {
    pub fn print(&self) -> String {
        let mut lines = vec![
            "╔═══════════════════════════════════════════════════════════════╗".to_string(),
            "║          Leafcutter Model Capability Report                   ║".to_string(),
            "╚═══════════════════════════════════════════════════════════════╝".to_string(),
            format!("  Architecture : {} (supported: {})",
                self.architecture.name(),
                if self.arch_supported { "YES ✅" } else { "NO ❌" }
            ),
        ];

        lines.push("\n  Quantization:".to_string());
        lines.push(self.quant_summary.report());

        if !self.missing_tensors.is_empty() {
            lines.push(format!("\n  Missing required tensors ({}):", self.missing_tensors.len()));
            for t in &self.missing_tensors {
                lines.push(format!("    - {}", t));
            }
        }

        if !self.extra_tensors.is_empty() {
            lines.push(format!("\n  Extra / unrecognised tensors ({}):", self.extra_tensors.len()));
            for t in &self.extra_tensors {
                lines.push(format!("    - {}", t));
            }
        }

        lines.push(format!("\n  ➤ Can run: {}",
            if self.can_run { "YES ✅" } else { "NO ❌" }
        ));

        lines.join("\n")
    }
}
