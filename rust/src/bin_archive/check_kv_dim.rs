fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_kv_dim <model.gguf>");
    let engine = leafcutter::inference::engine::Engine::load(&path).expect("load model");
    println!("head_dim = {}", engine.config.head_dim);
    println!("kv_head_dim = {}", engine.config.kv_head_dim);
    println!("num_heads = {}", engine.config.num_attention_heads);
    println!("num_kv_heads = {}", engine.config.num_key_value_heads);
    println!("hidden_size = {}", engine.config.hidden_size);
}
