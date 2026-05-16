//! ShardWriter — splits a GGUF model into per-layer shard files
//!
//! This is a **one-time preprocessing step**.  After splitting, the
//! original GGUF can be kept or archived; inference uses the shards.

use crate::model::loader::{GGUFModel, ModelConfig};
use crate::model::tensor::Tensor;
use super::format::{ShardHeader, ShardTensorMeta, QuantFormat, align_up, DATA_ALIGN};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

pub struct ShardWriter {
    pub config: ModelConfig,
    pub output_dir: String,
    pub quant_format: QuantFormat,
}

impl ShardWriter {
    pub fn new(config: ModelConfig, output_dir: &str) -> Self {
        Self::with_quant(config, output_dir, QuantFormat::F32)
    }

    pub fn with_quant(config: ModelConfig, output_dir: &str, quant_format: QuantFormat) -> Self {
        std::fs::create_dir_all(output_dir).ok();
        Self {
            config,
            output_dir: output_dir.to_string(),
            quant_format,
        }
    }

    /// Compute the on-disk data size for a tensor given the quantization format.
    fn tensor_data_size(&self, element_count: usize) -> u64 {
        match self.quant_format {
            QuantFormat::F32 => (element_count * 4) as u64,
            QuantFormat::Q8_0 => {
                assert_eq!(element_count % 32, 0, "Q8_0 requires element count to be multiple of 32");
                ((element_count / 32) * 34) as u64
            }
            QuantFormat::Q4_0 => {
                assert_eq!(element_count % 32, 0, "Q4_0 requires element count to be multiple of 32");
                ((element_count / 32) * 18) as u64
            }
        }
    }

    /// Write tensor data in the configured quantization format.
    fn write_tensor_data<W: Write>(&self, writer: &mut W, tensor: &Tensor) -> std::io::Result<()> {
        match self.quant_format {
            QuantFormat::F32 => {
                for &val in &tensor.data {
                    writer.write_all(&val.to_le_bytes())?;
                }
            }
            QuantFormat::Q8_0 => {
                let q8_bytes = crate::kernels::q8_0::quantize_f32_to_q8_0(&tensor.data);
                writer.write_all(&q8_bytes)?;
            }
            QuantFormat::Q4_0 => {
                let q4_bytes = crate::kernels::q4_0::quantize_f32_to_q4_0(&tensor.data);
                writer.write_all(&q4_bytes)?;
            }
        }
        Ok(())
    }

    /// Write a single layer's weights to a `.shard` file.
    pub fn write_layer_shard(
        &self,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> std::io::Result<(String, u64)> {
        let filename = format!("layer_{:03}.shard", layer_idx);
        let path = Path::new(&self.output_dir).join(&filename);
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        // Collect metadata and compute layout
        let tensor_count = weights.len() as u32;
        let mut metas: Vec<(String, &Tensor, ShardTensorMeta)> = Vec::new();

        // Header size
        let mut current_offset = ShardHeader::SIZE as u64;

        // First pass: compute metadata sizes
        for (name, tensor) in weights.iter() {
            let name_bytes = name.as_bytes().len() as u64;
            let meta_size = 4 + name_bytes + 4 + (tensor.shape.len() as u64) * 8 + 8 + 8;
            current_offset += meta_size;
        }

        // Align to data section boundary
        let data_start = align_up(current_offset, DATA_ALIGN);
        current_offset = data_start;

        // Second pass: assign data offsets and build metadata
        for (name, tensor) in weights.iter() {
            let data_size = self.tensor_data_size(tensor.data.len());
            let meta = ShardTensorMeta {
                name: name.clone(),
                rank: tensor.shape.len() as u32,
                dims: tensor.shape.iter().map(|&d| d as u64).collect(),
                data_offset: current_offset,
                data_size,
            };
            metas.push((name.clone(), tensor, meta));
            current_offset += data_size;
        }

        // Write header
        let header = ShardHeader::new(tensor_count, data_start, self.quant_format);
        header.write(&mut writer)?;

        // Write metadata
        for (_, _, meta) in &metas {
            meta.write(&mut writer)?;
        }

        // Pad to data_start
        let padding = data_start - writer.stream_position()?;
        if padding > 0 {
            writer.write_all(&vec![0u8; padding as usize])?;
        }

        // Write tensor data
        for (_, tensor, meta) in &metas {
            debug_assert_eq!(writer.stream_position()?, meta.data_offset);
            self.write_tensor_data(&mut writer, tensor)?;
        }

        writer.flush()?;
        let file_size = std::fs::metadata(&path)?.len();
        Ok((filename, file_size))
    }

    /// Write special weights (embed, norm, lm_head) to individual shard files.
    pub fn write_special_shard(
        &self,
        name: &str,
        weights: &HashMap<String, Tensor>,
    ) -> std::io::Result<(String, u64)> {
        let filename = format!("{}.shard", name);
        let path = Path::new(&self.output_dir).join(&filename);
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        let tensor_count = weights.len() as u32;
        let mut metas: Vec<ShardTensorMeta> = Vec::new();

        let mut current_offset = ShardHeader::SIZE as u64;
        for (tensor_name, tensor) in weights.iter() {
            let name_bytes = tensor_name.as_bytes().len() as u64;
            let meta_size = 4 + name_bytes + 4 + (tensor.shape.len() as u64) * 8 + 8 + 8;
            current_offset += meta_size;
        }

        let data_start = align_up(current_offset, DATA_ALIGN);
        current_offset = data_start;

        for (tensor_name, tensor) in weights.iter() {
            let data_size = self.tensor_data_size(tensor.data.len());
            let meta = ShardTensorMeta {
                name: tensor_name.clone(),
                rank: tensor.shape.len() as u32,
                dims: tensor.shape.iter().map(|&d| d as u64).collect(),
                data_offset: current_offset,
                data_size,
            };
            metas.push(meta);
            current_offset += data_size;
        }

        let header = ShardHeader::new(tensor_count, data_start, self.quant_format);
        header.write(&mut writer)?;

        for meta in &metas {
            meta.write(&mut writer)?;
        }

        let padding = data_start - writer.stream_position()?;
        if padding > 0 {
            writer.write_all(&vec![0u8; padding as usize])?;
        }

        for (tensor_name, tensor) in weights.iter() {
            let meta = metas.iter().find(|m| m.name == *tensor_name).unwrap();
            debug_assert_eq!(writer.stream_position()?, meta.data_offset);
            self.write_tensor_data(&mut writer, tensor)?;
        }

        writer.flush()?;
        let file_size = std::fs::metadata(&path)?.len();
        Ok((filename, file_size))
    }
}

/// Convenience: split an entire GGUF model into shards.
pub fn split_gguf_model(
    model_path: &str,
    output_dir: &str,
    quant_format: super::format::QuantFormat,
) -> Result<super::loader::Manifest, Box<dyn std::error::Error>> {
    println!("Loading GGUF model: {}", model_path);
    let model = GGUFModel::load(model_path)?;
    let config = model.config.clone();

    let writer = ShardWriter::with_quant(config.clone(), output_dir, quant_format);

    // Write layer shards
    let mut layer_files = Vec::new();
    println!("Splitting {} layers...", config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        let weights = model.load_layer(layer_idx)?;
        let (filename, size) = writer.write_layer_shard(layer_idx, &weights)?;
        layer_files.push(super::loader::LayerManifest {
            idx: layer_idx,
            file: filename,
            size,
        });
        if (layer_idx + 1) % 10 == 0 || layer_idx == config.num_hidden_layers - 1 {
            println!("  Layer {}/{} done", layer_idx + 1, config.num_hidden_layers);
        }
    }

    // Write special shards
    println!("Writing special weights...");
    let special_weights = model.load_special()?;

    // Split special weights into individual files for simplicity
    let embed_weights: HashMap<String, Tensor> = special_weights
        .iter()
        .filter(|(k, _)| k.contains("embed"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let (embed_file, embed_size) = writer.write_special_shard("embed", &embed_weights)?;

    let norm_weights: HashMap<String, Tensor> = special_weights
        .iter()
        .filter(|(k, _)| k.contains("norm"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let (norm_file, norm_size) = writer.write_special_shard("norm", &norm_weights)?;

    let lm_head_weights: HashMap<String, Tensor> = special_weights
        .iter()
        .filter(|(k, _)| k.contains("lm_head"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let (lm_head_file, lm_head_size) = writer.write_special_shard("lm_head", &lm_head_weights)?;

    let manifest = super::loader::Manifest {
        model: config_to_model_name(model_path),
        num_layers: config.num_hidden_layers,
        hidden_size: config.hidden_size,
        vocab_size: config.vocab_size,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        intermediate_size: config.intermediate_size,
        max_seq_len: config.max_seq_len,
        rope_theta: config.rope_theta,
        shard_dir: output_dir.to_string(),
        layers: layer_files,
        special: super::loader::SpecialManifest {
            embed: embed_file,
            embed_size,
            norm: norm_file,
            norm_size,
            lm_head: lm_head_file,
            lm_head_size,
        },
    };

    // Write manifest JSON
    let manifest_path = Path::new(output_dir).join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&manifest_path, manifest_json)?;
    println!("Manifest written to: {}", manifest_path.display());

    Ok(manifest)
}

fn config_to_model_name(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard::loader::{ShardLoader, Manifest, LayerManifest, SpecialManifest};
    use std::collections::HashMap;

    #[test]
    fn test_shard_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let output_dir = dir.path().to_str().unwrap();

        // Create fake layer weights
        let mut weights = HashMap::new();
        weights.insert("self_attn.q_proj.weight".to_string(), Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]
        ));
        weights.insert("mlp.gate_proj.weight".to_string(), Tensor::from_vec(
            vec![0.5, 1.5, 2.5, 3.5], vec![2, 2]
        ));
        weights.insert("input_layernorm.weight".to_string(), Tensor::from_vec(
            vec![1.0, 1.0], vec![2]
        ));

        let config = ModelConfig {
            hidden_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 2,
            max_seq_len: 128,
            vocab_size: 100,
            rope_theta: 10000.0,
        };

        let writer = ShardWriter::new(config, output_dir);
        let (filename, size) = writer.write_layer_shard(0, &weights).unwrap();
        assert!(size > 0);

        // Create a minimal manifest for the loader
        let manifest = Manifest {
            model: "test".to_string(),
            num_layers: 1,
            hidden_size: 2,
            vocab_size: 100,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 2,
            max_seq_len: 128,
            rope_theta: 10000.0,
            shard_dir: output_dir.to_string(),
            layers: vec![LayerManifest { idx: 0, file: filename, size }],
            special: SpecialManifest {
                embed: "embed.shard".to_string(),
                embed_size: 0,
                norm: "norm.shard".to_string(),
                norm_size: 0,
                lm_head: "lm_head.shard".to_string(),
                lm_head_size: 0,
            },
        };

        let loader = ShardLoader::from_manifest(manifest);
        let loaded = loader.load_layer(0).unwrap();

        assert_eq!(loaded.len(), 3);
        let q = loaded.get("self_attn.q_proj.weight").unwrap();
        assert_eq!(q.shape, vec![2, 2]);
        assert_eq!(q.data, vec![1.0, 2.0, 3.0, 4.0]);

        let gate = loaded.get("mlp.gate_proj.weight").unwrap();
        assert_eq!(gate.data, vec![0.5, 1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_q8_0_shard_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let output_dir = dir.path().to_str().unwrap();

        // Create fake layer weights (cols must be multiple of 32 for native Q8_0 matmul)
        let mut weights = HashMap::new();
        let q_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
        weights.insert("self_attn.q_proj.weight".to_string(), Tensor::from_vec(q_data.clone(), vec![2, 32]));

        let config = ModelConfig {
            hidden_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 32,
            max_seq_len: 128,
            vocab_size: 100,
            rope_theta: 10000.0,
        };

        let writer = ShardWriter::with_quant(config, output_dir, crate::shard::QuantFormat::Q8_0);
        let (filename, size) = writer.write_layer_shard(0, &weights).unwrap();
        assert!(size > 0);

        let manifest = Manifest {
            model: "test".to_string(),
            num_layers: 1,
            hidden_size: 32,
            vocab_size: 100,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 32,
            max_seq_len: 128,
            rope_theta: 10000.0,
            shard_dir: output_dir.to_string(),
            layers: vec![LayerManifest { idx: 0, file: filename, size }],
            special: SpecialManifest {
                embed: "embed.shard".to_string(),
                embed_size: 0,
                norm: "norm.shard".to_string(),
                norm_size: 0,
                lm_head: "lm_head.shard".to_string(),
                lm_head_size: 0,
            },
        };

        let loader = ShardLoader::from_manifest(manifest);
        let loaded = loader.load_layer(0).unwrap();

        assert_eq!(loaded.len(), 1);
        let q = loaded.get("self_attn.q_proj.weight").unwrap();
        assert_eq!(q.shape, vec![2, 32]);

        // Check roundtrip error is small (< 1% relative)
        for i in 0..q.data.len() {
            let err = (q.data[i] - q_data[i]).abs();
            assert!(err < 0.1, "Q8_0 roundtrip error too large at {}: got {}, expected {}, err={}",
                i, q.data[i], q_data[i], err);
        }

        // Verify the tensor carries native Q8_0 metadata for fast matmul
        assert!(q.is_quantized(), "Q8_0-loaded tensor should carry Q8_0 metadata");

        // Verify INT8 GEMM path produces the same result as f32 matmul
        // a [4, 2] @ q [2, 32] -> [4, 32]
        let a = Tensor::from_vec((0..8).map(|i| (i as f32) * 0.1).collect(), vec![4, 2]);
        let c_q8 = a.matmul(q);
        let q_f32 = Tensor::from_vec(q.data.clone(), q.shape.clone());
        let c_f32 = a.matmul(&q_f32);
        for i in 0..c_q8.data.len() {
            assert!((c_q8.data[i] - c_f32.data[i]).abs() < 1e-3,
                "INT8 GEMM mismatch at {}: q8={}, f32={}", i, c_q8.data[i], c_f32.data[i]);
        }
    }

    #[test]
    fn test_q4_0_shard_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let output_dir = dir.path().to_str().unwrap();

        let mut weights = HashMap::new();
        let q_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
        weights.insert("self_attn.q_proj.weight".to_string(), Tensor::from_vec(q_data.clone(), vec![2, 32]));

        let config = ModelConfig {
            hidden_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 32,
            max_seq_len: 128,
            vocab_size: 100,
            rope_theta: 10000.0,
        };

        let writer = ShardWriter::with_quant(config.clone(), output_dir, crate::shard::QuantFormat::Q4_0);
        let (filename, size) = writer.write_layer_shard(0, &weights).unwrap();
        assert!(size > 0);

        let manifest = Manifest {
            model: "test".to_string(),
            num_layers: 1,
            hidden_size: 32,
            vocab_size: 100,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 32,
            max_seq_len: 128,
            rope_theta: 10000.0,
            shard_dir: output_dir.to_string(),
            layers: vec![LayerManifest { idx: 0, file: filename, size }],
            special: SpecialManifest {
                embed: "embed.shard".to_string(),
                embed_size: 0,
                norm: "norm.shard".to_string(),
                norm_size: 0,
                lm_head: "lm_head.shard".to_string(),
                lm_head_size: 0,
            },
        };

        let loader = ShardLoader::from_manifest(manifest);
        let loaded = loader.load_layer(0).unwrap();

        assert_eq!(loaded.len(), 1);
        let q = loaded.get("self_attn.q_proj.weight").unwrap();
        assert_eq!(q.shape, vec![2, 32]);

        // Q4_0 has higher error tolerance (~0.5 max)
        for i in 0..q.data.len() {
            let err = (q.data[i] - q_data[i]).abs();
            assert!(err < 0.5, "Q4_0 roundtrip error too large at {}: got {}, expected {}, err={}",
                i, q.data[i], q_data[i], err);
        }

        assert!(q.is_quantized(), "Q4_0-loaded tensor should carry quantized metadata");

        // Verify Q4_0 GEMM path
        let a = Tensor::from_vec((0..8).map(|i| (i as f32) * 0.1).collect(), vec![4, 2]);
        let c_q4 = a.matmul(q);
        let q_f32 = Tensor::from_vec(q.data.clone(), q.shape.clone());
        let c_f32 = a.matmul(&q_f32);
        for i in 0..c_q4.data.len() {
            assert!((c_q4.data[i] - c_f32.data[i]).abs() < 1e-3,
                "Q4_0 GEMM mismatch at {}: q4={}, f32={}", i, c_q4.data[i], c_f32.data[i]);
        }
    }
}
