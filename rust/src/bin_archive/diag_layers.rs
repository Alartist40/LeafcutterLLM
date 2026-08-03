use leafcutter::inference::engine::Engine;

fn rms(vec: &[f32]) -> f32 {
    let mean_sq: f32 = vec.iter().map(|&x| x * x).sum::<f32>() / vec.len() as f32;
    mean_sq.sqrt()
}

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let mut engine = Engine::load(path).expect("Failed to load");
    let prompt = vec![9906usize];
    let hidden_size = engine.config.hidden_size;
    
    let mut hidden = engine.embed_lookup_mmap(&prompt).expect("embed_lookup_mmap failed");
    println!("Embedding RMS: {:.6}", rms(&hidden.data));
    
    for layer_idx in 0..engine.config.num_hidden_layers {
        let layer_weights = engine.model.load_layer(layer_idx).expect("load layer");
        
        let pre_norm_weight = layer_weights.get("input_layernorm.weight").unwrap();
        let pre_norm = hidden.rms_norm(pre_norm_weight, 1e-5);
        
        let attn_out = leafcutter::inference::attention::attention_forward(
            &pre_norm, &layer_weights, &engine.attn_params,
            &mut engine.kv_cache, layer_idx, engine.seq_offset);
        hidden = hidden.add(&attn_out);
        
        let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
            .or_else(|| layer_weights.get("ffn_norm.weight")).unwrap();
        let post_norm = hidden.rms_norm(post_norm_weight, 1e-5);
        
        let ffn_out = Engine::ffn_forward(&post_norm, &layer_weights).expect("ffn_forward failed");
        hidden = hidden.add(&ffn_out);
    }
    
    let special = engine.model.load_special().expect("special");
    let final_norm_weight = special.get("model.norm.weight").unwrap();
    let hidden_norm = hidden.rms_norm(final_norm_weight, 1e-5);
    println!("Final hidden state (after norm) first 10: {:?}", &hidden_norm.data[0..10]);
    println!("Final hidden state RMS: {:.6}", rms(&hidden_norm.data));
    
    // Compute logits manually for first 10000 tokens (for speed)
    let mut logits = vec![0.0f32; engine.config.vocab_size];
    let gguf = &engine.model.file;
    for token_id in 0..engine.config.vocab_size.min(10000) {
        let row = gguf.get_tensor_row_f32("token_embd.weight", token_id).unwrap();
        let mut dot = 0.0f32;
        for i in 0..hidden_size {
            dot += hidden_norm.data[i] * row[i];
        }
        logits[token_id] = dot;
    }
    
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 10 logits (first 10000 tokens):");
    for i in 0..10 {
        println!("  token={}: logit={:.6}", indexed[i].0, indexed[i].1);
    }
}
