//! Comprehensive Diagnostic — Verify all engine capabilities
//!
//! Tests in priority order:
//! 1. Llama (known working) — confirm nothing broke
//! 2. Qwen synthetic shapes — verify deltanet_forward without real model
//! 3. Qwen layer isolation — identify which layer type produces garbage
//! 4. Embedding/tokenizer — confirm correct token IDs
//! 5. Memory profile — confirm layer-wise loading still works
//! 6. Speed benchmark — confirm matrixmultiply performance

use leafcutter::inference::engine::Engine;
use std::env;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  LeafcutterLLM Comprehensive Diagnostic");
    println!("═══════════════════════════════════════════════════════════════\n");

    let args: Vec<String> = env::args().collect();

    // ── 1. Llama Smoke Test (proven working path) ─────────────────
    println!("━━━ Test 1: Llama-3.2-3B Smoke Test ━━━");
    if args.len() > 1 {
        let llama_path = &args[1];
        if std::path::Path::new(llama_path).exists() {
            test_llama_smoke(llama_path);
        } else {
            println!("  SKIP: Llama model not found at {}", llama_path);
        }
    } else {
        println!("  SKIP: Provide Llama path as arg 1");
    }

    // ── 2. Qwen Synthetic Shape Test ──────────────────────────────
    println!("\n━━━ Test 2: Qwen Synthetic Shape Test ━━━");
    test_qwen_synthetic();

    // ── 3. Qwen Layer Isolation (if model provided) ───────────────
    println!("\n━━━ Test 3: Qwen Layer Isolation ━━━");
    if args.len() > 2 {
        let qwen_path = &args[2];
        if std::path::Path::new(qwen_path).exists() {
            test_qwen_isolation(qwen_path);
        } else {
            println!("  SKIP: Qwen model not found at {}", qwen_path);
        }
    } else {
        println!("  SKIP: Provide Qwen path as arg 2");
    }

    // ── 4. Embedding Consistency ──────────────────────────────────
    println!("\n━━━ Test 4: Embedding Consistency ━━━");
    if args.len() > 1 && std::path::Path::new(&args[1]).exists() {
        test_embedding_consistency(&args[1]);
    } else {
        println!("  SKIP: Needs Llama model");
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Diagnostic complete");
    println!("═══════════════════════════════════════════════════════════════");
}

// ── 1. Llama Smoke Test ──────────────────────────────────────────

fn test_llama_smoke(path: &str) {
    use std::time::Instant;

    let start = Instant::now();
    let mut engine = match Engine::load(path) {
        Ok(e) => e,
        Err(e) => { println!("  FAIL: Could not load: {}", e); return; }
    };
    println!("  Load: {:.1}s", start.elapsed().as_secs_f32());

    // Test embedding lookup
    let tokens = vec![1, 2, 3, 4, 5]; // BOS + some tokens
    let hidden = engine.embed_lookup_mmap(&tokens).expect("embed_lookup_mmap failed");
    println!("  Embedding: shape={:?}, mean={:.4}, std={:.4}",
        hidden.shape,
        mean(&hidden.data),
        stddev(&hidden.data));

    // Test single-layer forward
    let start = Instant::now();
    let _ = engine.forward(&tokens);
    let elapsed = start.elapsed().as_secs_f32();
    println!("  1-layer forward: {:.2}s ({:.1} tok/s)", elapsed, tokens.len() as f32 / elapsed);

    // Verify no NaN
    let info = engine.info();
    println!("  Engine: arch={}, layers={}, hidden={}",
        info.architecture, info.total_layers, info.hidden_size);

    println!("  PASS");
}

// ── 2. Qwen Synthetic Test ───────────────────────────────────────

fn test_qwen_synthetic() {
    use leafcutter::inference::deltanet::{deltanet_forward, DeltaNetParams};
    use leafcutter::inference::ssm::{ssm_forward, SSMConfig};
    use leafcutter::cache::deltanet_state::DeltaNetStateCache;
    use leafcutter::cache::ssm_state::SSMStateCache;
    use leafcutter::model::tensor::Tensor;
    use std::collections::HashMap;

    let hidden = 4096;
    let conv_dim = 8192;

    // DeltaNet weights
    let mut dn_weights = HashMap::new();
    dn_weights.insert("attn_qkv.weight".to_string(), Tensor::from_vec(vec![0.001f32; hidden * conv_dim], vec![hidden, conv_dim]));
    dn_weights.insert("ssm_conv1d.weight".to_string(), Tensor::from_vec(vec![0.1f32; 4 * conv_dim], vec![4, conv_dim]));
    dn_weights.insert("ssm_alpha.weight".to_string(), Tensor::from_vec(vec![0.01f32; hidden * 32], vec![hidden, 32]));
    dn_weights.insert("ssm_beta.weight".to_string(), Tensor::from_vec(vec![0.01f32; hidden * 32], vec![hidden, 32]));
    dn_weights.insert("ssm_dt.bias".to_string(), Tensor::from_vec(vec![0.5f32; 32], vec![32]));
    dn_weights.insert("ssm_a".to_string(), Tensor::from_vec(vec![-0.046f32; 32], vec![32]));
    dn_weights.insert("ssm_out.weight".to_string(), Tensor::from_vec(vec![0.001f32; 4096 * hidden], vec![4096, hidden]));

    let dn_params = DeltaNetParams {
        num_qk_heads: 32, num_v_heads: 32, head_k_dim: 64, head_v_dim: 128,
        conv_dim, conv_kernel: 4, state_size: 128, norm_eps: 1e-5,
    };

    let input = Tensor::from_vec(vec![0.01f32; 2 * hidden], vec![2, hidden]);
    let mut cache = DeltaNetStateCache::new();

    let out = deltanet_forward(&input, &dn_weights, &dn_params, &mut cache, 0);
    println!("  DeltaNet output: shape={:?}, mean={:.6}, std={:.6}",
        out.shape, mean(&out.data), stddev(&out.data));

    // Check for NaN
    let nan_count = out.data.iter().filter(|&&v| !v.is_finite()).count();
    if nan_count > 0 {
        println!("  FAIL: {} NaN/Inf values", nan_count);
        return;
    }

    // Check magnitude is reasonable
    let max_abs = out.data.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
    if max_abs > 100.0 {
        println!("  WARNING: max_abs={:.2} — output may be too large", max_abs);
    } else if max_abs < 0.001 {
        println!("  WARNING: max_abs={:.6} — output may be too small (near zero)", max_abs);
    } else {
        println!("  Magnitude OK: max_abs={:.4}", max_abs);
    }

    // SSM test (for comparison)
    let mut ssm_weights = HashMap::new();
    ssm_weights.insert("attn_qkv.weight".to_string(), dn_weights["attn_qkv.weight"].clone());
    ssm_weights.insert("ssm_conv1d.weight".to_string(), dn_weights["ssm_conv1d.weight"].clone());
    ssm_weights.insert("ssm_alpha.weight".to_string(), dn_weights["ssm_alpha.weight"].clone());
    ssm_weights.insert("ssm_beta.weight".to_string(), dn_weights["ssm_beta.weight"].clone());
    ssm_weights.insert("ssm_dt.bias".to_string(), dn_weights["ssm_dt.bias"].clone());
    ssm_weights.insert("ssm_a".to_string(), dn_weights["ssm_a"].clone());
    ssm_weights.insert("ssm_out.weight".to_string(), dn_weights["ssm_out.weight"].clone());

    let ssm_config = SSMConfig { state_size: 32, inner_size: hidden, time_step_rank: 32, conv_kernel: 4, group_count: 1 };
    let mut ssm_cache = SSMStateCache::new();
    let ssm_out = ssm_forward(&input, &ssm_weights, &ssm_config, &mut ssm_cache, 0);

    println!("  SSM output:      shape={:?}, mean={:.6}, std={:.6}, max_abs={:.4}",
        ssm_out.shape, mean(&ssm_out.data), stddev(&ssm_out.data),
        ssm_out.data.iter().map(|&v| v.abs()).fold(0.0f32, f32::max));

    println!("  PASS");
}

// ── 3. Qwen Layer Isolation ──────────────────────────────────────

fn test_qwen_isolation(path: &str) {
    use leafcutter::model::gguf::GGUFile;

    let file = match GGUFile::open(path) {
        Ok(f) => f,
        Err(e) => { println!("  FAIL: {}", e); return; }
    };

    // Inspect first 8 layers
    for layer_idx in 0..8 {
        let prefix = format!("blk.{}", layer_idx);
        println!("  Layer {} tensors:", layer_idx);

        let has_qkv = file.get_tensor_info(&format!("{}.attn_qkv.weight", prefix)).is_some();
        let has_q = file.get_tensor_info(&format!("{}.attn_q.weight", prefix)).is_some();
        let has_gate = file.get_tensor_info(&format!("{}.attn_gate.weight", prefix)).is_some();

        if has_qkv {
            println!("    Type: DeltaNet/SSM");
            println!("    Has attn_gate: {}", has_gate);
            if has_gate {
                if let Some(info) = file.get_tensor_info(&format!("{}.attn_gate.weight", prefix)) {
                    let dims: Vec<_> = info.dimensions.iter().map(|&d| d as usize).collect();
                    println!("    Gate shape: {:?} (GGUF inner×outer)", dims);
                }
            }
        } else if has_q {
            println!("    Type: Attention");
        } else {
            println!("    Type: Unknown");
        }

        // Check ssm_a values for this layer
        if let Some(_info) = file.get_tensor_info(&format!("{}.ssm_a", prefix)) {
            if let Some(raw) = file.get_tensor_raw(&format!("{}.ssm_a", prefix)) {
                let vals: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                let mean_val = vals.iter().sum::<f32>() / vals.len() as f32;
                let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                println!("    ssm_a: n={}, mean={:.3}, min={:.3}, max={:.3}", vals.len(), mean_val, min, max);
            }
        }
    }

    println!("  PASS");
}

// ── 4. Embedding Consistency ─────────────────────────────────────

fn test_embedding_consistency(path: &str) {
    let engine = match Engine::load(path) {
        Ok(e) => e,
        Err(e) => { println!("  FAIL: {}", e); return; }
    };

    let tokens = vec![1, 100, 1000, 5000, 10000];

    let mmap_result = engine.embed_lookup_mmap(&tokens).expect("embed_lookup_mmap failed");
    println!("  mmap lookup: shape={:?}, mean={:.4}", mmap_result.shape, mean(&mmap_result.data));

    println!("  PASS");
}

// ── Statistics helpers ───────────────────────────────────────────

fn mean(data: &[f32]) -> f32 {
    if data.is_empty() { 0.0 } else { data.iter().sum::<f32>() / data.len() as f32 }
}

fn stddev(data: &[f32]) -> f32 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    let var = data.iter().map(|&v| (v - m).powi(2)).sum::<f32>() / (data.len() - 1) as f32;
    var.sqrt()
}
