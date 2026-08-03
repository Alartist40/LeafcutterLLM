use leafcutter::inference::engine::Engine;

fn save_bin(name: &str, data: &[f32]) {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    std::fs::write(name, bytes).unwrap();
}

fn main() {
    let path = std::env::args().nth(1).expect("Usage: debug_deltanet_layer0 <model.gguf>");
    let mut engine = Engine::load(&path).unwrap();
    let tokens = vec![17usize, 10, 17, 28];
    let hidden = engine.embed_lookup_mmap(&tokens).expect("embed_lookup_mmap failed");
    save_bin("dbg_embed.bin", &hidden.data);
    
    let layer_weights = engine.model.load_layer(0).unwrap();
    let pre_norm_weight = layer_weights.get("input_layernorm.weight")
        .or_else(|| layer_weights.get("attn_norm.weight"))
        .expect("Missing pre-norm");
    let pre_norm = hidden.rms_norm(pre_norm_weight, engine.config.norm_eps);
    save_bin("dbg_pre_norm.bin", &pre_norm.data);
    
    for (name, tensor) in &layer_weights {
        let fname = format!("dbg_weight_{}.bin", name.replace(".", "_"));
        save_bin(&fname, &tensor.data);
    }
    
    let deltanet_out = leafcutter::inference::deltanet::deltanet_forward(
        &pre_norm, &layer_weights, &engine.deltanet_params, &mut engine.deltanet_cache, 0);
    save_bin("dbg_deltanet_out.bin", &deltanet_out.data);
    
    let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
        .or_else(|| layer_weights.get("ffn_norm.weight"))
        .expect("Missing post-norm");
    let post_norm = hidden.add(&deltanet_out).rms_norm(post_norm_weight, engine.config.norm_eps);
    let ffn_out = leafcutter::inference::engine::Engine::ffn_forward(&post_norm, &layer_weights).expect("ffn_forward failed");
    let final_hidden = hidden.add(&deltanet_out).add(&ffn_out);
    save_bin("dbg_layer1_out.bin", &final_hidden.data);
    
    println!("Dumps saved.");
}
