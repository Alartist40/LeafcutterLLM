//! GGUF quantization type registry and dispatch
//!
//! Auto-detects tensor quantization formats and routes to the correct
//! dequantization kernel. Unknown types are reported clearly rather than
//! failing with an opaque error code.

use std::fmt;

/// Every quantization type defined in the GGUF / ggml specification.
///
/// Type numbers are taken from llama.cpp `ggml_type` enum (stable since
/// GGUF v3).  Gaps (4, 5, 11) are deprecated / removed types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    F32,      // 0
    F16,      // 1
    Q4_0,     // 2
    Q4_1,     // 3
    Q5_0,     // 6
    Q5_1,     // 7
    Q8_0,     // 8
    Q8_1,     // 9
    Q2_K,     // 10
    Q3_K,     // 11 (deprecated)
    Q4_K,     // 12
    Q5_K,     // 13
    Q6_K,     // 14
    Q8_K,     // 15
    IQ2_XXS,  // 16
    IQ2_XS,   // 17
    IQ3_XXS,  // 18
    IQ3_S,    // 19
    IQ4_NL,   // 20
    IQ4_XS,   // 21
    IQ4_K,    // 22
    IQ5_0,    // 23
    IQ5_NL,   // 24
    IQ5_K,    // 25
    BF16,     // 30 (newer ggml)
}

impl QuantType {
    /// Raw GGUF type code.
    pub fn code(self) -> u32 {
        match self {
            QuantType::F32     => 0,
            QuantType::F16     => 1,
            QuantType::Q4_0    => 2,
            QuantType::Q4_1    => 3,
            QuantType::Q5_0    => 6,
            QuantType::Q5_1    => 7,
            QuantType::Q8_0    => 8,
            QuantType::Q8_1    => 9,
            QuantType::Q2_K    => 10,
            QuantType::Q3_K    => 11,
            QuantType::Q4_K    => 12,
            QuantType::Q5_K    => 13,
            QuantType::Q6_K    => 14,
            QuantType::Q8_K    => 15,
            QuantType::IQ2_XXS => 16,
            QuantType::IQ2_XS  => 17,
            QuantType::IQ3_XXS => 18,
            QuantType::IQ3_S   => 19,
            QuantType::IQ4_NL  => 20,
            QuantType::IQ4_XS  => 21,
            QuantType::IQ4_K   => 22,
            QuantType::IQ5_0   => 23,
            QuantType::IQ5_NL  => 24,
            QuantType::IQ5_K   => 25,
            QuantType::BF16    => 30,
        }
    }

    /// Parse a raw GGUF type code.
    pub fn from_u32(code: u32) -> Option<Self> {
        match code {
            0  => Some(QuantType::F32),
            1  => Some(QuantType::F16),
            2  => Some(QuantType::Q4_0),
            3  => Some(QuantType::Q4_1),
            6  => Some(QuantType::Q5_0),
            7  => Some(QuantType::Q5_1),
            8  => Some(QuantType::Q8_0),
            9  => Some(QuantType::Q8_1),
            10 => Some(QuantType::Q2_K),
            11 => Some(QuantType::Q3_K),
            12 => Some(QuantType::Q4_K),
            13 => Some(QuantType::Q5_K),
            14 => Some(QuantType::Q6_K),
            15 => Some(QuantType::Q8_K),
            16 => Some(QuantType::IQ2_XXS),
            17 => Some(QuantType::IQ2_XS),
            18 => Some(QuantType::IQ3_XXS),
            19 => Some(QuantType::IQ3_S),
            20 => Some(QuantType::IQ4_NL),
            21 => Some(QuantType::IQ4_XS),
            22 => Some(QuantType::IQ4_K),
            23 => Some(QuantType::IQ5_0),
            24 => Some(QuantType::IQ5_NL),
            25 => Some(QuantType::IQ5_K),
            30 => Some(QuantType::BF16),
            _  => None,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            QuantType::F32     => "F32",
            QuantType::F16     => "F16",
            QuantType::Q4_0    => "Q4_0",
            QuantType::Q4_1    => "Q4_1",
            QuantType::Q5_0    => "Q5_0",
            QuantType::Q5_1    => "Q5_1",
            QuantType::Q8_0    => "Q8_0",
            QuantType::Q8_1    => "Q8_1",
            QuantType::Q2_K    => "Q2_K",
            QuantType::Q3_K    => "Q3_K",
            QuantType::Q4_K    => "Q4_K",
            QuantType::Q5_K    => "Q5_K",
            QuantType::Q6_K    => "Q6_K",
            QuantType::Q8_K    => "Q8_K",
            QuantType::IQ2_XXS => "IQ2_XXS",
            QuantType::IQ2_XS  => "IQ2_XS",
            QuantType::IQ3_XXS => "IQ3_XXS",
            QuantType::IQ3_S   => "IQ3_S",
            QuantType::IQ4_NL  => "IQ4_NL",
            QuantType::IQ4_XS  => "IQ4_XS",
            QuantType::IQ4_K   => "IQ4_K",
            QuantType::IQ5_0   => "IQ5_0",
            QuantType::IQ5_NL  => "IQ5_NL",
            QuantType::IQ5_K   => "IQ5_K",
            QuantType::BF16    => "BF16",
        }
    }

    /// Bits per weight (approximate).
    pub fn bits_per_weight(self) -> f32 {
        match self {
            QuantType::F32     => 32.0,
            QuantType::F16     => 16.0,
            QuantType::BF16    => 16.0,
            QuantType::Q4_0    => 4.5,
            QuantType::Q4_1    => 5.0,
            QuantType::Q5_0    => 5.5,
            QuantType::Q5_1    => 6.0,
            QuantType::Q8_0    => 8.5,
            QuantType::Q8_1    => 9.0,
            QuantType::Q2_K    => 2.625,
            QuantType::Q3_K    => 3.4375,
            QuantType::Q4_K    => 4.5,
            QuantType::Q5_K    => 5.5,
            QuantType::Q6_K    => 6.5625,
            QuantType::Q8_K    => 8.5,
            QuantType::IQ2_XXS => 2.0625,
            QuantType::IQ2_XS  => 2.3125,
            QuantType::IQ3_XXS => 3.0625,
            QuantType::IQ3_S   => 3.4375,
            QuantType::IQ4_NL  => 4.5,
            QuantType::IQ4_XS  => 4.25,
            QuantType::IQ4_K   => 4.25,
            QuantType::IQ5_0   => 5.0,
            QuantType::IQ5_NL  => 5.0,
            QuantType::IQ5_K   => 5.5,
        }
    }

    /// Number of elements per block.
    pub fn block_size(self) -> usize {
        match self {
            QuantType::F32 | QuantType::F16 | QuantType::BF16 => 1,
            QuantType::Q4_0 | QuantType::Q4_1
            | QuantType::Q5_0 | QuantType::Q5_1
            | QuantType::Q8_0 | QuantType::Q8_1
            | QuantType::IQ4_NL => 32,
            QuantType::Q2_K | QuantType::Q3_K | QuantType::Q4_K
            | QuantType::Q5_K | QuantType::Q6_K | QuantType::Q8_K
            | QuantType::IQ2_XXS | QuantType::IQ2_XS | QuantType::IQ3_XXS
            | QuantType::IQ3_S | QuantType::IQ4_XS | QuantType::IQ4_K
            | QuantType::IQ5_0 | QuantType::IQ5_NL | QuantType::IQ5_K => 256,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            QuantType::F32     => 4,
            QuantType::F16     => 2,
            QuantType::BF16    => 2,
            QuantType::Q4_0    => 18,   // 2 + 16
            QuantType::Q4_1    => 20,   // 2 + 2 + 16
            QuantType::Q5_0    => 22,   // 2 + 4 + 16
            QuantType::Q5_1    => 24,   // 2 + 2 + 4 + 16
            QuantType::Q8_0    => 34,   // 2 + 32
            QuantType::Q8_1    => 36,   // 4 + 4 + 32
            QuantType::Q2_K    => {
                // 2 + 2 + 256/16 + 256/4 = 2 + 2 + 16 + 64 = 84
                2 + 2 + 256/16 + 256/4
            }
            QuantType::Q3_K    => {
                // 2 + 256/4 + 256/8 + 12 = 2 + 64 + 32 + 12 = 110
                2 + 256/4 + 256/8 + 12
            }
            QuantType::Q4_K    => 144,  // 2 + 2 + 128 + 12
            QuantType::Q5_K    => 176,  // 2 + 2 + 128 + 32 + 12
            QuantType::Q6_K    => 210,  // 2 + 128 + 64 + 16
            QuantType::Q8_K    => 292,  // 4 + 256 + 32
            QuantType::IQ2_XXS => 66,   // 2 + 64
            QuantType::IQ2_XS  => 74,   // 2 + 64 + 8
            QuantType::IQ3_XXS => 98,   // 2 + 64 + 32
            QuantType::IQ3_S   => {
                // 2 + 64 + 32 + 8 + 4 = 110
                2 + 256/4 + 256/8 + 256/32 + 4
            }
            QuantType::IQ4_NL  => 18,   // 2 + 16
            QuantType::IQ4_XS  => {
                // sizeof(ggml_half) + sizeof(uint16_t) + 256/64 + 256/2
                // = 2 + 2 + 4 + 128 = 136
                2 + 2 + 4 + 128
            }
            QuantType::IQ4_K   => {
                // Similar to IQ4_XS but with K-scale
                2 + 2 + 128 + 4
            }
            QuantType::IQ5_0   => {
                // 2 + 160 = 162 (approx)
                2 + 160
            }
            QuantType::IQ5_NL  => {
                // 2 + 160 = 162
                2 + 160
            }
            QuantType::IQ5_K   => {
                // 2 + 2 + 160 + 12 = 176
                2 + 2 + 160 + 12
            }
        }
    }

    /// Whether Leafcutter has a dequantization kernel for this type.
    pub fn is_supported(self) -> bool {
        matches!(self,
            QuantType::F32 |
            QuantType::F16 |
            QuantType::Q4_0 |
            QuantType::Q8_0 |
            QuantType::Q4_K |
            QuantType::Q5_K |
            QuantType::Q6_K |
            QuantType::IQ4_NL
        )
    }

    /// Calculate the raw byte size for a tensor with `count` elements.
    pub fn tensor_size(self, count: usize) -> usize {
        let bs = self.block_size();
        let bb = self.block_bytes();
        let nblocks = (count + bs - 1) / bs;
        nblocks * bb
    }
}

impl fmt::Display for QuantType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Summarizes the quantization types used in a model.
#[derive(Debug, Default)]
pub struct QuantSummary {
    pub types: std::collections::HashMap<QuantType, usize>, // type -> tensor count
    pub total_tensors: usize,
    pub unsupported: Vec<QuantType>,
}

impl QuantSummary {
    pub fn is_fully_supported(&self) -> bool {
        self.unsupported.is_empty()
    }

    pub fn report(&self) -> String {
        let mut lines = vec![
            format!("Quantization Type Report ({} tensors):", self.total_tensors),
            "  Type       | Count | Supported | Bits/W".to_string(),
            "  -----------|-------|-----------|-------".to_string(),
        ];
        let mut types: Vec<_> = self.types.iter().collect();
        types.sort_by_key(|(t, _)| t.code());
        for (typ, count) in types {
            lines.push(format!(
                "  {:10} | {:5} | {:9} | {:.2}",
                typ.name(),
                count,
                if typ.is_supported() { "YES" } else { "NO" },
                typ.bits_per_weight()
            ));
        }
        if !self.unsupported.is_empty() {
            lines.push("\n  Unsupported types (blocking load):".to_string());
            for t in &self.unsupported {
                lines.push(format!("    - {}", t.name()));
            }
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iq4nl_block_size() {
        // Verified against Qwen3.5-9B file offsets
        assert_eq!(QuantType::IQ4_NL.block_size(), 32);
        assert_eq!(QuantType::IQ4_NL.block_bytes(), 18);
        // 4096*4096 = 16_777_216 elements
        // blocks = 524_288, size = 9_437_184
        assert_eq!(QuantType::IQ4_NL.tensor_size(16_777_216), 9_437_184);
    }

    #[test]
    fn test_q4k_block_size() {
        assert_eq!(QuantType::Q4_K.block_size(), 256);
        assert_eq!(QuantType::Q4_K.block_bytes(), 144);
    }

    #[test]
    fn test_f32_block_size() {
        assert_eq!(QuantType::F32.block_size(), 1);
        assert_eq!(QuantType::F32.block_bytes(), 4);
        assert_eq!(QuantType::F32.tensor_size(100), 400);
    }
}
