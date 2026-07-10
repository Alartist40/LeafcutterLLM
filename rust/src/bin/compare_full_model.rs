//! Full-Model Layer-by-Layer Comparison Tool (CORRECTED)
//!
//! FIXED from v1:
//!   - ssm_forward (not ssm_layer)
//!   - attention_forward (not transformer_layer)
//!   - GGUF tokenizer (not byte-level tokenization)
//!   - Tensor::rms_norm() (not manual reimplementation)
//!
//! Usage:
//!   cargo run --bin compare_full_model -- \
//!     --model /path/to/model.gguf \
//!     --prompt "The capital of France is" \
//!     --output-dir /tmp/layer_dumps

use leafcutter::model::tensor::Tensor;
use std::collections::HashMap;
use std::fs;

fn dump_tensor(name: &str, data: &[f32], output_dir: &str) {
    let path = format!("{}/{}", output_dir, name);
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    if fs::write(&path, bytes).is_ok() {
        eprintln!("  Dumped: {} ({} floats)", path, data.len());
    }
}

fn print_stats(name: &str, data: &[f32]) {
    let finite: Vec<f32> = data.iter().copied().filter(|v| v.is_finite()).collect();
    let all_nan = data.iter().all(|&v| v.is_nan());
    let min = finite.iter().copied().fold(f32::INFINITY, f32::min);
    let max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = if !finite.is_empty() { finite.iter().sum::<f32>() / finite.len() as f32 } else { f32::NAN };
    let zeros = data.iter().filter(|&&v| v == 0.0).count();
    eprintln!("  {:30} min={:12.6} max={:12.6} mean={:12.6} zeros={:6}/{} all_nan={}",
        name, min, max, mean, zeros, data.len(), all_nan);
}

/// Tokenize using GGUF vocab with greedy longest-match.
/// Falls back to byte-level if vocab not available.
fn tokenize_prompt(prompt: &str, gguf_file: &leafcutter::model::gguf::GGUFile) -> Vec<usize> {
    let vocab_tokens: Vec<String> = gguf_file.metadata.iter()
        .find(|(k, _)| *k == "tokenizer.ggml.tokens")
        .and_then(|(_, v)| {
            if let leafcutter::model::gguf::GGUFValue::Array(arr) = v {
                Some(arr.iter().filter_map(|item| {
                    if let leafcutter::model::gguf::GGUFValue::String(s) = item {
                        Some(s.clone())
                    } else { None }
                }).collect())
            } else { None }
        })
        .unwrap_or_default();

    if vocab_tokens.is_empty() {
        eprintln!("  WARNING: No tokenizer vocab in GGUF, falling back to bytes");
        return prompt.bytes().map(|b| b as usize).collect();
    }

    let token_to_id: HashMap<&str, usize> = vocab_tokens.iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();

    let mut tokens = Vec::new();
    let mut remaining = prompt;

    // BOS
    if let Some(&bos_id) = token_to_id.get("<s>") {
        tokens.push(bos_id);
    } else if let Some(&bos_id) = token_to_id.get("<|begin_of_text|>") {
        tokens.push(bos_id);
    }

    // Greedy longest-match
    while !remaining.is_empty() {
        let mut matched = None;
        for len in (1..=remaining.len().min(64)).rev() {
            if let Some(&id) = token_to_id.get(&remaining[..len]) {
                matched = Some((id, len));
                break;
            }
        }
        if let Some((id, len)) = matched {
            tokens.push(id);
            remaining = &remaining[len..];
        } else {
            tokens.push(remaining.as_bytes()[0] as usize);
            remaining = &remaining[1..];
        }
    }
    tokens
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = None;
    let mut prompt = "The capital of France is";
    let mut output_dir = "/tmp/layer_dumps".to_string();
    let mut raw_tokens: Option<Vec<usize>> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_path = args.get(i).cloned(); }
            "--prompt" => { i += 1; prompt = args.get(i).map(|s| s.as_str()).unwrap_or(prompt); }
            "--output-dir" => { i += 1; output_dir = args.get(i).unwrap_or(&output_dir).to_string(); }
            "--tokens" => {
                i += 1;
                if let Some(s) = args.get(i) {
                    raw_tokens = Some(s.split(',').filter_map(|t| t.parse().ok()).collect());
                }
            }
            _ => {}
        }
        i += 1;
    }

    let model_path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("Usage: compare_full_model --model <path> [--prompt \"text\"] [--tokens 9906] [--output-dir /tmp/dumps]");
            std::process::exit(1);
        }
    };

    fs::create_dir_all(&output_dir).unwrap_or_default();

    eprintln!("============================================================");
    eprintln!("Full-Model Layer Comparison (CORRECTED v2)");
    eprintln!("============================================================");
    eprintln!("Model:   {}", model_path);
    eprintln!("Prompt:  '{}'", prompt);
    eprintln!("Output:  {}", output_dir);
    eprintln!("============================================================");

    eprintln!("\n[1] Loading model...");
    let mut engine = match leafcutter::inference::engine::Engine::load(&model_path) {
        Ok(e) => {
            let info = e.info();
            eprintln!("    Architecture: {}", info.architecture);
            eprintln!("    Layers:       {}", info.total_layers);
            eprintln!("    Hidden size:  {}", info.hidden_size);
            eprintln!("    SSM mode:     {:?}", info.use_ssm);
            e
        }
        Err(e) => {
            eprintln!("    FAILED: {}", e);
            std::process::exit(1);
        }
    };

    let token_ids = raw_tokens.unwrap_or_else(|| tokenize_prompt(prompt, &engine.model.file));
    eprintln!("\n[2] Prompt: '{}'", prompt);
    eprintln!("    Token IDs: {:?}", token_ids);
    eprintln!("    Token count: {}", token_ids.len());

    let hidden_size = engine.config.hidden_size;
    let seq_len = token_ids.len();

    // Embedding
    eprintln!("\n[3] Embedding lookup...");
    let mut embed_data = vec![0.0f32; seq_len * hidden_size];
    for (i, &token) in token_ids.iter().enumerate() {
        let idx = token.min(engine.config.vocab_size - 1);
        let row = engine.model.file.get_tensor_row_f32("token_embd.weight", idx)
            .expect("embedding row");
        embed_data[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(&row);
    }
    let mut x = Tensor::from_vec(embed_data, vec![seq_len, hidden_size]);

    print_stats("00_embedding_output", &x.data);
    dump_tensor("00_embedding_output.bin", &x.data, &output_dir);

    // All layers
    eprintln!("\n[4] Running {} layers...", engine.config.num_hidden_layers);

    for layer_idx in 0..engine.config.num_hidden_layers {
        let weights = engine.model.load_layer(layer_idx)
            .expect(&format!("load layer {}", layer_idx));

        let has_standard_attn = weights.contains_key("self_attn.q_proj.weight")
            || weights.contains_key("attn_q.weight");
        let has_fused_qkv = weights.contains_key("attn_qkv.weight");
        let has_ssm = weights.contains_key("ssm_out.weight")
            || weights.contains_key("ssm_alpha.weight");

        // Pre-LN: save original, norm, compute, add residual to ORIGINAL
        let residual_attn = x.clone();

        // Input RMSNorm
        let x_normed = if let Some(ln_w) = weights.get("input_layernorm.weight")
            .or_else(|| weights.get("attn_norm.weight"))
        {
            x.rms_norm(ln_w, 1e-5)
        } else {
            x.clone()
        };

        // Attention or SSM
        let attn_out = if has_ssm {
            leafcutter::inference::ssm::ssm_forward(
                &x_normed, &weights, &engine.ssm_config, &mut engine.ssm_cache, layer_idx
            )
        } else if has_standard_attn || has_fused_qkv {
            leafcutter::inference::attention::attention_forward(
                &x_normed, &weights, &engine.attn_params, &mut engine.kv_cache, layer_idx, 0
            )
        } else {
            eprintln!("    Layer {}: WARNING — unknown type", layer_idx);
            eprintln!("    Available: {:?}", weights.keys().collect::<Vec<_>>());
            Tensor::from_vec(vec![0.0f32; x_normed.size()], x_normed.shape.clone())
        };
        x = residual_attn.add(&attn_out);

        // Pre-LN FFN: save original, norm, compute, add residual to ORIGINAL
        let residual_ffn = x.clone();

        // Post-attention RMSNorm
        let x_normed = if let Some(ln_w) = weights.get("post_attention_layernorm.weight")
            .or_else(|| weights.get("ffn_norm.weight"))
        {
            x.rms_norm(ln_w, 1e-5)
        } else {
            x.clone()
        };

        // FFN
        let has_ffn = weights.contains_key("mlp.gate_proj.weight")
            || weights.contains_key("ffn_gate.weight");
        if has_ffn {
            let ffn_out = leafcutter::inference::engine::Engine::ffn_forward(&x_normed, &weights).expect("ffn_forward failed");
            x = residual_ffn.add(&ffn_out);
        }

        let name = format!("layer_{:02}_output", layer_idx);
        print_stats(&name, &x.data);
        dump_tensor(&format!("{}.bin", name), &x.data, &output_dir);
    }

    // Final norm
    eprintln!("\n[5] Final layer norm...");
    if let Some(norm_w) = engine.special_weights.get("model.norm.weight")
        .or_else(|| engine.special_weights.get("output_norm.weight"))
    {
        x = x.rms_norm(norm_w, 1e-5);
    }

    print_stats("29_final_norm_output", &x.data);
    dump_tensor("29_final_norm_output.bin", &x.data, &output_dir);

    // LM head
    eprintln!("\n[6] LM head projection...");
    let hidden_last = &x.data[(seq_len - 1) * hidden_size..seq_len * hidden_size];
    let logits: Vec<f32> = if engine.special_weights.contains_key("output.weight") {
        (0..engine.config.vocab_size).map(|tid| {
            let row = engine.model.file.get_tensor_row_f32("output.weight", tid).unwrap();
            hidden_last.iter().zip(row.iter()).map(|(a, b)| a * b).sum::<f32>()
        }).collect()
    } else {
        (0..engine.config.vocab_size).map(|tid| {
            let row = engine.model.file.get_tensor_row_f32("token_embd.weight", tid).unwrap();
            hidden_last.iter().zip(row.iter()).map(|(a, b)| a * b).sum::<f32>()
        }).collect()
    };

    print_stats("30_lm_head_logits", &logits);
    dump_tensor("30_lm_head_logits.bin", &logits, &output_dir);

    let (top_token, top_logit) = logits.iter().enumerate()
        .filter(|(_, v)| !v.is_nan())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap_or((0, &f32::NEG_INFINITY));
    eprintln!("\nTop token: id={} logit={:.6}", top_token, top_logit);

    let mut top: Vec<(usize, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i, v)).collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("\nTop 10:");
    for (rank, (tok, val)) in top.iter().take(10).enumerate() {
        eprintln!("  {:2}. id={:<8} logit={:12.6}", rank + 1, tok, val);
    }

    // Generation test
    eprintln!("\n[7] Generation test (10 tokens, greedy)...");
    engine.kv_cache.clear();
    engine.seq_offset = 0;

    let prefill_logits = engine.forward(&token_ids);
    engine.seq_offset = token_ids.len();

    let mut next_token = prefill_logits.iter().enumerate()
        .filter(|(_, v)| !v.is_nan())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    eprintln!("  Prefill top: {}", next_token);

    for step in 0..2 {
        let dec_logits = engine.forward(&[next_token]);
        engine.seq_offset += 1;
        next_token = dec_logits.iter().enumerate()
            .filter(|(_, v)| !v.is_nan())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        eprintln!("  Step {:2}: token_id={:<8}", step, next_token);
    }

    eprintln!("\n============================================================");
    eprintln!("All dumps in: {}", output_dir);
    eprintln!("============================================================");
}
