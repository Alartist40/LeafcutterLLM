//! End-to-end integration test: tokenize → generate → detokenize
//!
//! This test loads the real Qwen2.5-3B model and runs a single forward pass.
//! It is marked #[ignore] because it takes several minutes with naive matmul.
//! Run with: cargo test --test end_to_end -- --ignored --nocapture

use leafcutter::inference::engine::Engine;
use leafcutter::tokenizer::Tokenizer;
use std::time::Instant;

#[test]
#[ignore = "Slow: runs real inference on 3B model"]
fn test_end_to_end_generation() {
    let model_path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    let tok_path = "tests/tokenizer.json";

    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping: model not found at {}", model_path);
        return;
    }
    if !std::path::Path::new(tok_path).exists() {
        eprintln!("Skipping: tokenizer not found at {}", tok_path);
        return;
    }

    println!("\n🌿 Leafcutter End-to-End Test");
    println!("   Model: {}", model_path);

    // Load tokenizer
    let tok = Tokenizer::from_file(tok_path).expect("Failed to load tokenizer");
    println!("✅ Tokenizer loaded: vocab_size={}", tok.vocab_size());

    // Load engine
    let mut engine = Engine::load(model_path).expect("Failed to load engine");
    println!("✅ Engine loaded: {} layers, hidden_size={}", engine.config.num_hidden_layers, engine.config.hidden_size);

    // Apply chat template and tokenize
    let prompt = tok.apply_chat_template("Hello");
    let tokens = tok.encode(&prompt);
    println!("📝 Prompt: '{}' → Tokens: {:?}", prompt.trim(), tokens);

    // Generate 3 tokens
    println!("\n⏳ Running inference (this may take a few minutes)...");
    let start = Instant::now();
    let generated = engine.generate(&tokens, 3, 0.7, 0.9);
    let elapsed = start.elapsed();

    println!("\n✅ Generated {} tokens in {:?}", generated.len(), elapsed);
    println!("   Token IDs: {:?}", generated);

    // Decode
    let all_tokens: Vec<usize> = tokens.iter().chain(generated.iter()).copied().collect();
    let decoded = tok.decode(&all_tokens, false);
    println!("   Decoded: '{}'", decoded);

    // Sanity checks
    assert!(!generated.is_empty(), "Should generate at least one token");
    // Note: model vocab (151,936) may exceed tokenizer.json vocab (151,665).
    // Extended tokens decode to empty strings but are valid model outputs.
    assert!(generated.iter().all(|&t| t < engine.config.vocab_size), "All tokens should be in model vocab");
}

#[test]
fn test_engine_loads_without_crashing() {
    let model_path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping: model not found");
        return;
    }

    let engine = Engine::load(model_path);
    assert!(engine.is_ok(), "Engine should load without crashing");

    let eng = engine.unwrap();
    assert_eq!(eng.config.num_hidden_layers, 36);
    assert!(eng.config.vocab_size > 0);
    println!("✅ Engine loads: {} layers, vocab={}", eng.config.num_hidden_layers, eng.config.vocab_size);
}

#[test]
#[ignore = "Slow: runs real inference on 3B model"]
fn test_simple_prompt_no_template() {
    let model_path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    let tok_path = "tests/tokenizer.json";
    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(tok_path).exists() {
        return;
    }
    let tok = Tokenizer::from_file(tok_path).unwrap();
    let mut engine = Engine::load(model_path).unwrap();
    
    let tokens = tok.encode("Hello");
    println!("No-template prompt: 'Hello' → {:?}", tokens);
    let gen = engine.generate(&tokens, 5, 0.0, 1.0);
    println!("Generated (no template, greedy): {:?}", gen);
    let decoded = tok.decode(&gen, false);
    println!("Decoded: '{}'", decoded);
}

#[test]
#[ignore = "Slow: runs real inference on 3B model"]
fn test_debug_logits() {
    let model_path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    let tok_path = "tests/tokenizer.json";
    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(tok_path).exists() {
        return;
    }
    let tok = Tokenizer::from_file(tok_path).unwrap();
    let mut engine = Engine::load(model_path).unwrap();
    
    // Simple prompt
    let tokens = tok.encode("Hello");
    engine.kv_cache.clear();
    let logits = engine.forward(&tokens);
    
    // Find top 10 logits
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    println!("Top 10 logits for 'Hello':");
    for (i, (idx, val)) in indexed.iter().take(10).enumerate() {
        println!("  {}: token={} logit={}", i, idx, val);
    }
}

#[test]
#[ignore = "Slow: runs real inference on 3B model"]
fn test_find_nan_source() {
    let model_path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    let tok_path = "tests/tokenizer.json";
    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(tok_path).exists() {
        return;
    }
    let tok = Tokenizer::from_file(tok_path).unwrap();
    let mut engine = Engine::load(model_path).unwrap();
    
    let tokens = tok.encode("Hello");
    engine.kv_cache.clear();
    
    // Embedding
    let embed = engine.special_weights.get("model.embed_tokens.weight").unwrap();
    let mut hidden = engine.embed_lookup(&tokens, embed);
    println!("After embed: nan={} inf={}", 
        hidden.data.iter().filter(|&&v| v.is_nan()).count(),
        hidden.data.iter().filter(|&&v| v.is_infinite()).count());
    
    let attn_params = leafcutter::inference::attention::AttentionParams {
        num_heads: engine.config.num_attention_heads,
        num_kv_heads: engine.config.num_key_value_heads,
        head_dim: engine.config.hidden_size / engine.config.num_attention_heads,
        rope_theta: engine.config.rope_theta,
    };
    
    for layer_idx in 0..engine.config.num_hidden_layers {
        let layer_weights = engine.model.load_layer(layer_idx).unwrap();
        
        let pre_norm_weight = layer_weights.get("input_layernorm.weight").unwrap();
        let normed = hidden.rms_norm(pre_norm_weight, 1e-5);
        if normed.data.iter().any(|&v| v.is_nan()) {
            println!("Layer {}: NaN AFTER pre-norm", layer_idx);
            break;
        }
        
        let attn_out = leafcutter::inference::attention::attention_forward(
            &normed, &layer_weights, &attn_params, &mut engine.kv_cache, layer_idx);
        if attn_out.data.iter().any(|&v| v.is_nan()) {
            println!("Layer {}: NaN AFTER attention", layer_idx);
            break;
        }
        
        hidden = hidden.add(&attn_out);
        if hidden.data.iter().any(|&v| v.is_nan()) {
            println!("Layer {}: NaN AFTER attention residual", layer_idx);
            break;
        }
        
        let post_norm_weight = layer_weights.get("post_attention_layernorm.weight").unwrap();
        let normed = hidden.rms_norm(post_norm_weight, 1e-5);
        if normed.data.iter().any(|&v| v.is_nan()) {
            println!("Layer {}: NaN AFTER post-norm", layer_idx);
            break;
        }
        
        // FFN
        let gate = layer_weights.get("mlp.gate_proj.weight").unwrap();
        let up = layer_weights.get("mlp.up_proj.weight").unwrap();
        let down = layer_weights.get("mlp.down_proj.weight").unwrap();
        let gate_proj = normed.matmul(gate);
        let up_proj = normed.matmul(up);
        let activated = gate_proj.silu();
        let mut fused = vec![0.0f32; activated.size()];
        for i in 0..activated.size() {
            fused[i] = activated.data[i] * up_proj.data[i];
        }
        let fused_tensor = leafcutter::model::tensor::Tensor::from_vec(fused, activated.shape.clone());
        let ffn_out = fused_tensor.matmul(down);
        
        if ffn_out.data.iter().any(|&v| v.is_nan()) {
            println!("Layer {}: NaN AFTER FFN", layer_idx);
            break;
        }
        
        hidden = hidden.add(&ffn_out);
        if hidden.data.iter().any(|&v| v.is_nan()) {
            println!("Layer {}: NaN AFTER FFN residual", layer_idx);
            break;
        }
        
        if layer_idx < 3 || layer_idx % 10 == 0 {
            println!("Layer {}: OK (no NaN)", layer_idx);
        }
    }
    
    let final_norm = engine.special_weights.get("model.norm.weight").unwrap();
    hidden = hidden.rms_norm(final_norm, 1e-5);
    println!("After final norm: nan={}", hidden.data.iter().filter(|&&v| v.is_nan()).count());
    
    let lm_head = engine.special_weights.get("lm_head.weight").unwrap();
    let logits = hidden.matmul(lm_head);
    println!("After lm_head: nan={}", logits.data.iter().filter(|&&v| v.is_nan()).count());
}

#[test]
#[ignore = "Slow: runs real inference on 3B model"]
fn test_debug_layer1_ffn() {
    let model_path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    let tok_path = "tests/tokenizer.json";
    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(tok_path).exists() {
        return;
    }
    let tok = Tokenizer::from_file(tok_path).unwrap();
    let mut engine = Engine::load(model_path).unwrap();
    let tokens = tok.encode("Hello");
    engine.kv_cache.clear();
    
    let embed = engine.special_weights.get("model.embed_tokens.weight").unwrap();
    let mut hidden = engine.embed_lookup(&tokens, embed);
    
    let attn_params = leafcutter::inference::attention::AttentionParams {
        num_heads: engine.config.num_attention_heads,
        num_kv_heads: engine.config.num_key_value_heads,
        head_dim: engine.config.hidden_size / engine.config.num_attention_heads,
        rope_theta: engine.config.rope_theta,
    };
    
    // Run layer 0
    let w0 = engine.model.load_layer(0).unwrap();
    let n0 = hidden.rms_norm(w0.get("input_layernorm.weight").unwrap(), 1e-5);
    let a0 = leafcutter::inference::attention::attention_forward(&n0, &w0, &attn_params, &mut engine.kv_cache, 0);
    hidden = hidden.add(&a0);
    let n0 = hidden.rms_norm(w0.get("post_attention_layernorm.weight").unwrap(), 1e-5);
    let g0 = n0.matmul(w0.get("mlp.gate_proj.weight").unwrap());
    let u0 = n0.matmul(w0.get("mlp.up_proj.weight").unwrap());
    let act0 = g0.silu();
    let mut f0 = vec![0.0f32; act0.size()];
    for i in 0..act0.size() { f0[i] = act0.data[i] * u0.data[i]; }
    let f0t = leafcutter::model::tensor::Tensor::from_vec(f0, act0.shape.clone());
    let ffn0 = f0t.matmul(w0.get("mlp.down_proj.weight").unwrap());
    hidden = hidden.add(&ffn0);
    println!("After layer 0: nan={} inf={} min={} max={}", 
        hidden.data.iter().filter(|&&v| v.is_nan()).count(),
        hidden.data.iter().filter(|&&v| v.is_infinite()).count(),
        hidden.data.iter().cloned().fold(f32::INFINITY, f32::min),
        hidden.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    // Run layer 1 step by step
    let w1 = engine.model.load_layer(1).unwrap();
    let n1 = hidden.rms_norm(w1.get("input_layernorm.weight").unwrap(), 1e-5);
    println!("Layer 1 pre-norm: nan={} inf={}", 
        n1.data.iter().filter(|&&v| v.is_nan()).count(),
        n1.data.iter().filter(|&&v| v.is_infinite()).count());
    
    let a1 = leafcutter::inference::attention::attention_forward(&n1, &w1, &attn_params, &mut engine.kv_cache, 1);
    println!("Layer 1 attn_out: nan={} inf={}", 
        a1.data.iter().filter(|&&v| v.is_nan()).count(),
        a1.data.iter().filter(|&&v| v.is_infinite()).count());
    
    hidden = hidden.add(&a1);
    println!("Layer 1 after attn residual: nan={} inf={}", 
        hidden.data.iter().filter(|&&v| v.is_nan()).count(),
        hidden.data.iter().filter(|&&v| v.is_infinite()).count());
    
    let n1 = hidden.rms_norm(w1.get("post_attention_layernorm.weight").unwrap(), 1e-5);
    println!("Layer 1 post-norm: nan={} inf={} min={} max={}", 
        n1.data.iter().filter(|&&v| v.is_nan()).count(),
        n1.data.iter().filter(|&&v| v.is_infinite()).count(),
        n1.data.iter().cloned().fold(f32::INFINITY, f32::min),
        n1.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    let g1 = n1.matmul(w1.get("mlp.gate_proj.weight").unwrap());
    println!("Layer 1 gate_proj: nan={} inf={} min={} max={}", 
        g1.data.iter().filter(|&&v| v.is_nan()).count(),
        g1.data.iter().filter(|&&v| v.is_infinite()).count(),
        g1.data.iter().cloned().fold(f32::INFINITY, f32::min),
        g1.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    let u1 = n1.matmul(w1.get("mlp.up_proj.weight").unwrap());
    println!("Layer 1 up_proj: nan={} inf={} min={} max={}", 
        u1.data.iter().filter(|&&v| v.is_nan()).count(),
        u1.data.iter().filter(|&&v| v.is_infinite()).count(),
        u1.data.iter().cloned().fold(f32::INFINITY, f32::min),
        u1.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    let act1 = g1.silu();
    println!("Layer 1 silu: nan={} inf={}", 
        act1.data.iter().filter(|&&v| v.is_nan()).count(),
        act1.data.iter().filter(|&&v| v.is_infinite()).count());
    
    let mut f1 = vec![0.0f32; act1.size()];
    for i in 0..act1.size() { f1[i] = act1.data[i] * u1.data[i]; }
    println!("Layer 1 fused (before down): nan={} inf={}", 
        f1.iter().filter(|&&v| v.is_nan()).count(),
        f1.iter().filter(|&&v| v.is_infinite()).count());
    
    let f1t = leafcutter::model::tensor::Tensor::from_vec(f1, act1.shape.clone());
    let ffn1 = f1t.matmul(w1.get("mlp.down_proj.weight").unwrap());
    println!("Layer 1 ffn_out: nan={} inf={} min={} max={}", 
        ffn1.data.iter().filter(|&&v| v.is_nan()).count(),
        ffn1.data.iter().filter(|&&v| v.is_infinite()).count(),
        ffn1.data.iter().cloned().fold(f32::INFINITY, f32::min),
        ffn1.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
}

#[test]
#[ignore = "Slow: runs real inference on 3B model"]
fn test_single_forward_no_nan() {
    let model_path = "/run/media/xander/rootfs/home/pi/the-pathfinder-eye_ai/models/qwen2.5-3b-q4.gguf";
    let tok_path = "tests/tokenizer.json";
    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(tok_path).exists() {
        return;
    }
    let tok = Tokenizer::from_file(tok_path).unwrap();
    let mut engine = Engine::load(model_path).unwrap();
    let tokens = tok.encode("Hello");
    engine.kv_cache.clear();
    
    let logits = engine.forward(&tokens);
    let nan_count = logits.iter().filter(|&&v| v.is_nan()).count();
    let inf_count = logits.iter().filter(|&&v| v.is_infinite()).count();
    
    println!("Prompt: 'Hello' ({} tokens)", tokens.len());
    println!("Logits len: {}", logits.len());
    println!("NaN count: {}/{}", nan_count, logits.len());
    println!("Inf count: {}/{}", inf_count, logits.len());
    println!("Min: {}  Max: {}", 
        logits.iter().cloned().fold(f32::INFINITY, f32::min),
        logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    assert_eq!(nan_count, 0, "Logits contain NaN values!");
    assert_eq!(inf_count, 0, "Logits contain Inf values!");
}
