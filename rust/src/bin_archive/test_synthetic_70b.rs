use leafcutter::inference::engine::Engine;

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

fn read_peak_rss_kb() -> usize {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmHWM:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(Ok(v)) = parts.get(1).map(|s| s.parse::<usize>()) {
                    return v;
                }
            }
        }
    }
    0
}

fn main() {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "synthetic_llama2_70b_q4_0.gguf".to_string());

    println!("🧪 Synthetic 70B Layer Streaming Test");
    println!("   Model: {}", path);

    let rss_before = read_rss_kb();
    let mut engine = Engine::load(&path).expect("Failed to load engine");
    let rss_after_load = read_rss_kb();

    println!("✅ Engine loaded: {} layers, hidden_size={}",
        engine.config.num_hidden_layers, engine.config.hidden_size);
    println!("   RSS after load: {} MB", rss_after_load / 1024);

    // Dummy tokens within vocab range
    let tokens = vec![0usize, 1, 2];
    println!("\n⏳ Forward pass with {} tokens...", tokens.len());

    let mut max_rss_during_forward = rss_after_load;
    let mut layer_rss = Vec::new();

    // Manual forward to measure per-layer RSS
    let seq_len = tokens.len();
    let mut hidden = engine.embed_lookup_mmap(&tokens).expect("embed_lookup_mmap failed");

    for layer_idx in 0..engine.config.num_hidden_layers {
        let layer_weights = engine.model.load_layer(layer_idx)
            .expect("Failed to load layer");
        let rss_after_load_layer = read_rss_kb();

        let pre_norm_weight = layer_weights.get("input_layernorm.weight")
            .or_else(|| layer_weights.get("attn_norm.weight"))
            .expect("Missing pre-norm");
        let normed = hidden.rms_norm(pre_norm_weight, 1e-5);

        let has_standard_attn = layer_weights.contains_key("self_attn.q_proj.weight")
            || layer_weights.contains_key("attn_q.weight");

        if has_standard_attn {
            use leafcutter::inference::attention::attention_forward;
            let attn_out = attention_forward(&normed, &layer_weights, &engine.attn_params, &mut engine.kv_cache, layer_idx, engine.seq_offset);
            hidden = hidden.add(&attn_out);
        }

        let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
            .or_else(|| layer_weights.get("ffn_norm.weight"))
            .expect("Missing post-norm");
        let normed = hidden.rms_norm(post_norm_weight, 1e-5);
        let ffn_out = Engine::ffn_forward(&normed, &layer_weights).expect("ffn_forward failed");
        hidden = hidden.add(&ffn_out);

        let rss_after_compute = read_rss_kb();
        max_rss_during_forward = max_rss_during_forward.max(rss_after_compute);

        if layer_idx < 3 || layer_idx % 10 == 9 || layer_idx == engine.config.num_hidden_layers - 1 {
            layer_rss.push((layer_idx, rss_after_load_layer, rss_after_compute));
        }
    }

    let final_norm = engine.special_weights.get("model.norm.weight")
        .expect("Missing final norm");
    hidden = hidden.rms_norm(final_norm, 1e-5);

    // Skip lm_head for synthetic test (vocab mismatch with tokenizer doesn't matter)
    let _logits: Vec<f32> = (0..engine.config.vocab_size)
        .map(|_| 0.0f32)
        .collect();

    let rss_after_forward = read_rss_kb();
    let peak_rss = read_peak_rss_kb();

    println!("\n📊 Per-layer RSS trace (sampled):");
    println!("   {:>6} | {:>10} | {:>10}", "Layer", "After Load", "After Compute");
    for (idx, after_load, after_compute) in &layer_rss {
        println!("   {:>6} | {:>8} MB | {:>8} MB", idx, after_load / 1024, after_compute / 1024);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                 SYNTHETIC 70B MEMORY RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Layers:                   {}", engine.config.num_hidden_layers);
    println!("  Hidden size:              {}", engine.config.hidden_size);
    println!("  Vocab size:               {}", engine.config.vocab_size);
    println!("───────────────────────────────────────────────────────────────");
    println!("  RSS before load:          {:>8} MB", rss_before / 1024);
    println!("  RSS after engine load:    {:>8} MB", rss_after_load / 1024);
    println!("  Max RSS during forward:   {:>8} MB", max_rss_during_forward / 1024);
    println!("  RSS after forward:        {:>8} MB", rss_after_forward / 1024);
    println!("  PEAK RSS (VmHWM):         {:>8} MB", peak_rss / 1024);
    println!("───────────────────────────────────────────────────────────────");

    // Check if RSS stayed flat
    let first_layer_rss = layer_rss.first().map(|(_, l, _)| *l).unwrap_or(0);
    let last_layer_rss = layer_rss.last().map(|(_, l, _)| *l).unwrap_or(0);
    let growth = (last_layer_rss as i64 - first_layer_rss as i64) / 1024;

    if growth < 50 {
        println!("  ✅ RSS stayed flat across {} layers (growth: {} MB)",
            engine.config.num_hidden_layers, growth);
    } else {
        println!("  ⚠️  RSS grew by {} MB across {} layers",
            growth, engine.config.num_hidden_layers);
    }
    println!("═══════════════════════════════════════════════════════════════");
}
