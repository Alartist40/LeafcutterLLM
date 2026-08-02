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
    Qwen36,
    Mistral,
    Phi,
    Gemma,
    Yi,
    Nemotron,
    Falcon,
    /// DeepSeek-2/V3 family (Kimi K2.6, DeepSeek-V3, etc.).
    /// MLA attention + routed MoE + optional shared expert.
    DeepSeek2,
    /// GLM-DSA / Z.AI DeepSeek-Sparse-Attention models.
    /// Extends DeepSeek-2 with a sparse-attention indexer + MTP heads.
    GlmDsa,
    Unknown,
}

impl ModelArchitecture {
    /// Detect architecture from GGUF metadata.
    pub fn detect(file: &GGUFile) -> Self {
        if let Some(GGUFValue::String(arch)) = file.metadata.get("general.architecture") {
            match arch.as_str() {
                "llama"    => ModelArchitecture::Llama,
                "qwen2" | "qwen3" => ModelArchitecture::Qwen2,
                "qwen35"   => ModelArchitecture::Qwen35,
                "qwen35moe" | "qwen36" | "ornith" => ModelArchitecture::Qwen36,
                "mistral"  => ModelArchitecture::Mistral,
                "mistral3" => ModelArchitecture::Mistral,
                "phi" | "phi3" | "phi4" => ModelArchitecture::Phi,
                "gemma" | "gemma2" | "gemma3" | "gemma4" => ModelArchitecture::Gemma,
                "yi" => ModelArchitecture::Yi,
                "nemotron" | "nvidia_nemotron" => ModelArchitecture::Nemotron,
                "falcon" | "falcon3" | "falcon2" => ModelArchitecture::Falcon,
                "deepseek2" | "deepseek" => ModelArchitecture::DeepSeek2,
                "glm-dsa" => ModelArchitecture::GlmDsa,
                _ => ModelArchitecture::Unknown,
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
            ModelArchitecture::Qwen36  => "Qwen3.6",
            ModelArchitecture::Mistral => "Mistral",
            ModelArchitecture::Phi     => "Phi",
            ModelArchitecture::Gemma   => "Gemma",
            ModelArchitecture::Yi      => "Yi",
            ModelArchitecture::Nemotron => "Nemotron",
            ModelArchitecture::Falcon => "Falcon",
            ModelArchitecture::DeepSeek2 => "DeepSeek-2",
            ModelArchitecture::GlmDsa => "GLM-DSA",
            ModelArchitecture::Unknown => "Unknown",
        }
    }

    /// Metadata key prefix used for hyper-parameters.
    pub fn metadata_prefix(self) -> &'static str {
        match self {
            ModelArchitecture::Llama   => "llama",
            ModelArchitecture::Qwen2   => "qwen2",
            ModelArchitecture::Qwen35  => "qwen35",
            ModelArchitecture::Qwen36  => "qwen35moe",
            ModelArchitecture::Mistral => "llama", // Mistral uses llama.* keys
            ModelArchitecture::Phi     => "phi",
            ModelArchitecture::Gemma   => "gemma",
            ModelArchitecture::Yi       => "llama", // Yi uses llama.* keys in GGUF
            ModelArchitecture::Nemotron => "llama", // Nemotron uses llama.* keys
            ModelArchitecture::Falcon => "falcon", // Falcon uses falcon.* keys
            ModelArchitecture::DeepSeek2 => "deepseek2",
            ModelArchitecture::GlmDsa => "glm-dsa",
            ModelArchitecture::Unknown => "llama", // best-effort fallback
        }
    }
    /// Whether the full inference stack supports this architecture.
    ///
    /// "Supported" here means: we have the correct layer weight mappings
    /// and attention/FFN structure implemented.  Gemma-3/4 is supported
    /// for text models (12B It is the verification target).
    pub fn is_supported(self) -> bool {
        matches!(
            self,
            ModelArchitecture::Llama
                | ModelArchitecture::Qwen2
                | ModelArchitecture::Qwen35
                | ModelArchitecture::Qwen36
                | ModelArchitecture::Mistral
                | ModelArchitecture::Phi
                | ModelArchitecture::Gemma
                | ModelArchitecture::Yi
                | ModelArchitecture::Nemotron
        )
    }

    /// Whether this architecture uses SSM (State Space Model) layers.
    pub fn uses_ssm(self) -> bool {
        matches!(self, ModelArchitecture::Qwen35 | ModelArchitecture::Qwen36)
    }

    /// Whether this architecture uses fused QKV projections.
    pub fn uses_fused_qkv(self) -> bool {
        matches!(self, ModelArchitecture::Qwen35 | ModelArchitecture::Qwen36 | ModelArchitecture::Phi)
    }

    /// Standard transformer layer weight mappings for this architecture.
    ///
    /// Returns `(gguf_suffix, engine_name)` pairs.  The loader will look
    /// for `blk.N.{gguf_suffix}` in the GGUF and map it to the engine
    /// key `{engine_name}`.
    pub fn layer_mappings(self) -> &'static [(&'static str, &'static str)] {
        match self {
            ModelArchitecture::Qwen35 => &[
                // Attention (full attention layers)
                ("attn_q.weight",     "self_attn.q_proj.weight"),
                ("attn_k.weight",     "self_attn.k_proj.weight"),
                ("attn_v.weight",     "self_attn.v_proj.weight"),
                ("attn_output.weight","self_attn.o_proj.weight"),
                // Q/K per-head RMSNorm (Qwen3.5-style)
                ("attn_q_norm.weight", "attn_q_norm.weight"),
                ("attn_k_norm.weight", "attn_k_norm.weight"),
                // Fused attention (SSM layers use fused QKV)
                ("attn_qkv.weight",   "self_attn.qkv_proj.weight"),
                ("attn_gate.weight",  "self_attn.gate_proj.weight"),
                // FFN
                ("ffn_gate.weight",   "mlp.gate_proj.weight"),
                ("ffn_up.weight",     "mlp.up_proj.weight"),
                ("ffn_down.weight",   "mlp.down_proj.weight"),
                // Norms
                ("attn_norm.weight",  "input_layernorm.weight"),
                ("post_attention_norm.weight", "post_attention_layernorm.weight"),
                // SSM
                ("ssm_alpha.weight",  "ssm_alpha.weight"),
                ("ssm_beta.weight",   "ssm_beta.weight"),
                ("ssm_conv1d.weight", "ssm_conv1d.weight"),
                ("ssm_dt.bias",       "ssm_dt.bias"),
                ("ssm_norm.weight",   "ssm_norm.weight"),
                ("ssm_out.weight",    "ssm_out.weight"),
                ("ssm_a",             "ssm_a"),
            ],
            ModelArchitecture::Qwen36 => &[
                // Gated DeltaNet (linear attention with state matrix)
                ("attn_q.weight",          "self_attn.q_proj.weight"),
                ("attn_k.weight",          "self_attn.k_proj.weight"),
                ("attn_v.weight",          "self_attn.v_proj.weight"),
                ("attn_output.weight",     "self_attn.o_proj.weight"),
                // Fused QKV for hybrid layers
                ("attn_qkv.weight",        "self_attn.qkv_proj.weight"),
                ("attn_gate.weight",       "self_attn.gate_proj.weight"),
                // MoE routing
                ("ffn_gate_inp.weight",    "mlp.gate.weight"),
                ("ffn_gate_exps.weight",   "mlp.expert_gate.weight"),
                ("ffn_down_exps.weight",   "mlp.expert_down.weight"),
                ("ffn_up_exps.weight",     "mlp.expert_up.weight"),
                ("ffn_norm.weight",        "mlp.norm.weight"),
                // Shared expert (Qwen-MoE style)
                ("ffn_gate_shexp.weight",  "mlp.shared_expert_gate.weight"),
                ("ffn_up_shexp.weight",    "mlp.shared_expert_up.weight"),
                ("ffn_down_shexp.weight",  "mlp.shared_expert_down.weight"),
                // Norms
                ("attn_norm.weight",       "input_layernorm.weight"),
                ("post_attention_norm.weight", "post_attention_layernorm.weight"),
                // DeltaNet state / recurrent weights
                ("ssm_alpha.weight",       "ssm_alpha.weight"),
                ("ssm_beta.weight",        "ssm_beta.weight"),
                ("ssm_conv1d.weight",      "ssm_conv1d.weight"),
                ("ssm_dt.bias",            "ssm_dt.bias"),
                ("ssm_norm.weight",        "ssm_norm.weight"),
                ("ssm_out.weight",         "ssm_out.weight"),
                ("ssm_a",                  "ssm_a"),
                ("ssm_state.weight",       "ssm_state.weight"),
                ("ssm_gate.weight",        "ssm_gate.weight"),
            ],
            ModelArchitecture::Qwen2 => &[
                // Attention — Qwen2 uses separate Q/K/V projections WITH
                // per-projection biases (unlike Llama, which has none).
                ("attn_q.weight",  "self_attn.q_proj.weight"),
                ("attn_k.weight",  "self_attn.k_proj.weight"),
                ("attn_v.weight",  "self_attn.v_proj.weight"),
                ("attn_output.weight", "self_attn.o_proj.weight"),
                ("attn_q.bias",    "self_attn.q_proj.bias"),
                ("attn_k.bias",    "self_attn.k_proj.bias"),
                ("attn_v.bias",    "self_attn.v_proj.bias"),
                // FFN
                ("ffn_gate.weight", "mlp.gate_proj.weight"),
                ("ffn_up.weight",   "mlp.up_proj.weight"),
                ("ffn_down.weight", "mlp.down_proj.weight"),
                // Norms
                ("attn_norm.weight", "input_layernorm.weight"),
                ("ffn_norm.weight",  "post_attention_layernorm.weight"),
            ],
            ModelArchitecture::Gemma => &[
                // Attention — Gemma's G layers use separate Q/K/V; S layers fuse Q+V
                ("attn_q.weight", "self_attn.q_proj.weight"),
                ("attn_k.weight", "self_attn.k_proj.weight"),
                ("attn_v.weight", "self_attn.v_proj.weight"),
                ("attn_output.weight", "self_attn.o_proj.weight"),
                // Per-head RMSNorm on Q and K (Gemma 3+)
                ("attn_q_norm.weight", "attn_q_norm.weight"),
                ("attn_k_norm.weight", "attn_k_norm.weight"),
                // FFN — Gemma uses GeGLU with separate gate and up projections
                ("ffn_gate.weight", "mlp.gate_proj.weight"),
                ("ffn_up.weight", "mlp.up_proj.weight"),
                ("ffn_down.weight", "mlp.down_proj.weight"),
                // Norms
                //   Gemma has 4 RMSNorm tensors per layer:
                //   * attn_norm         — applied before attention
                //   * post_attention_norm — applied to attention output (pre-residual-add)
                //   * ffn_norm          — applied before FFN
                //   * post_ffw_norm     — applied to FFN output (pre-residual-add)
                // We map the *input* norms to the engine's input_layernorm /
                // post_attention_layernorm slots so existing engine dispatch
                // works; per-layer post_norms are read directly via the
                // Gemma-specific forward path.
                ("attn_norm.weight", "input_layernorm.weight"),
                ("ffn_norm.weight", "post_attention_layernorm.weight"),
                // Gemma 4 has two extra per-layer RMSNorms applied AFTER
                // attention and AFTER FFN, before the residual add.
                ("post_attention_norm.weight", "post_attention_norm.weight"),
                ("post_ffw_norm.weight", "post_ffw_norm.weight"),
                // Layer-output scale (scalar, Gemma 3+)
                ("layer_output_scale.weight", "layer_output_scale.weight"),
            ],
            _ => &[
                ("attn_q.weight", "self_attn.q_proj.weight"),
                ("attn_k.weight", "self_attn.k_proj.weight"),
                ("attn_v.weight", "self_attn.v_proj.weight"),
                ("attn_output.weight", "self_attn.o_proj.weight"),
                ("ffn_gate.weight", "mlp.gate_proj.weight"),
                ("ffn_up.weight", "mlp.up_proj.weight"),
                ("ffn_down.weight", "mlp.down_proj.weight"),
                ("attn_norm.weight", "input_layernorm.weight"),
                ("ffn_norm.weight", "post_attention_layernorm.weight"),
            ],
        }
    }

    /// Extra weight suffixes that may appear in this architecture but
    /// are not part of the standard mapping.  Used for capability reports.
    pub fn known_extra_suffixes(self) -> &'static [&'static str] {
        match self {
            ModelArchitecture::Qwen35 => &[
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "nextn.eh_proj.weight",
                "nextn.enorm.weight",
                "nextn.hnorm.weight",
                "nextn.shared_head_norm.weight",
            ],
            ModelArchitecture::Qwen36 => &[
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "attn_v_norm.weight",
                "nextn.eh_proj.weight",
                "nextn.enorm.weight",
                "nextn.hnorm.weight",
                "nextn.shared_head_norm.weight",
                "moe_gate.weight",
                "moe_norm.weight",
                "expert_gate.weight",
                // Shared MoE experts (Qwen3.6 architecture)
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
                "ffn_down_shexp.weight",
                // NextN / MTP layer projections
                "nextn.embed_tokens.weight",
                "nextn.shared_head.weight",
                "nextn.enorm.weight",
                "nextn.hnorm.weight",
            ],
            ModelArchitecture::Llama => &["attn_norm.bias", "ffn_norm.bias"],
            ModelArchitecture::Gemma => &[
                // Per-head RMSNorm on Q / K (Gemma 2+)
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                // Gemma 4-norm layout: post-attention and post-ffw norms
                "post_attention_norm.weight",
                "post_ffw_norm.weight",
                // Per-layer scalar residual scale
                "layer_output_scale.weight",
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
    pub uses_ssm: bool,
    pub uses_fused_qkv: bool,
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

        lines.push(format!("\n  SSM layers  : {}", if self.uses_ssm { "YES ✅" } else { "NO" }));
        lines.push(format!("  Fused QKV   : {}", if self.uses_fused_qkv { "YES ✅" } else { "NO" }));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write_minimal_gguf(path: &str, arch: &str) {
        use std::io::Write;
        let mut buf: Vec<u8> = Vec::new();
        // Magic
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        // Version
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count
        buf.extend_from_slice(&0u64.to_le_bytes());
        // Metadata count
        buf.extend_from_slice(&1u64.to_le_bytes());
        // Key: "general.architecture"
        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        // Value type: String = 8
        buf.extend_from_slice(&8u32.to_le_bytes());
        // Value
        let val = arch.as_bytes();
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);
        std::fs::write(path, buf).unwrap();
    }

    fn mock_gguf_with_arch(arch: &str) -> GGUFile {
        let tmp = std::env::temp_dir().join(format!("test_arch_{}.gguf", arch.replace('.', "_")));
        write_minimal_gguf(tmp.to_str().unwrap(), arch);
        GGUFile::open(&tmp).expect("Failed to open mock GGUF")
    }

    #[test]
    fn test_detect_llama() {
        let f = mock_gguf_with_arch("llama");
        assert_eq!(ModelArchitecture::detect(&f), ModelArchitecture::Llama);
    }

    #[test]
    fn test_detect_mistral3() {
        let f = mock_gguf_with_arch("mistral3");
        assert_eq!(ModelArchitecture::detect(&f), ModelArchitecture::Mistral);
    }

    #[test]
    fn test_detect_gemma3() {
        let f = mock_gguf_with_arch("gemma3");
        assert_eq!(ModelArchitecture::detect(&f), ModelArchitecture::Gemma);
    }

    #[test]
    fn test_detect_phi4() {
        let f = mock_gguf_with_arch("phi4");
        assert_eq!(ModelArchitecture::detect(&f), ModelArchitecture::Phi);
    }

    #[test]
    fn test_detect_yi() {
        let f = mock_gguf_with_arch("yi");
        assert_eq!(ModelArchitecture::detect(&f), ModelArchitecture::Yi);
    }

    #[test]
    fn test_detect_nemotron() {
        let f = mock_gguf_with_arch("nemotron");
        assert_eq!(ModelArchitecture::detect(&f), ModelArchitecture::Nemotron);
    }

    #[test]
    fn test_detect_falcon() {
        let f = mock_gguf_with_arch("falcon");
        assert_eq!(ModelArchitecture::detect(&f), ModelArchitecture::Falcon);
    }

    #[test]
    fn test_detect_qwen3() {
        let f = mock_gguf_with_arch("qwen3");
        assert_eq!(ModelArchitecture::detect(&f), ModelArchitecture::Qwen2);
    }

    #[test]
    fn test_yi_is_supported() {
        assert!(ModelArchitecture::Yi.is_supported());
    }

    #[test]
    fn test_nemotron_is_supported() {
        assert!(ModelArchitecture::Nemotron.is_supported());
    }

    #[test]
    fn test_falcon_not_supported() {
        assert!(!ModelArchitecture::Falcon.is_supported());
    }

    #[test]
    fn test_detect_deepseek2() {
        let f = mock_gguf_with_arch("deepseek2");
        assert_eq!(ModelArchitecture::detect(&f), ModelArchitecture::DeepSeek2);
    }

    #[test]
    fn test_detect_glm_dsa() {
        let f = mock_gguf_with_arch("glm-dsa");
        assert_eq!(ModelArchitecture::detect(&f), ModelArchitecture::GlmDsa);
    }

    #[test]
    fn test_deepseek2_meta_prefix_and_name() {
        assert_eq!(ModelArchitecture::DeepSeek2.metadata_prefix(), "deepseek2");
        assert_eq!(ModelArchitecture::DeepSeek2.name(), "DeepSeek-2");
        assert_eq!(ModelArchitecture::GlmDsa.metadata_prefix(), "glm-dsa");
        assert_eq!(ModelArchitecture::GlmDsa.name(), "GLM-DSA");
    }
}
