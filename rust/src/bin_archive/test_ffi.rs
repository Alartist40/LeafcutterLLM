use leafcutter::inference::engine::Engine;

fn main() {
    let mut engine = Engine::load("../models/Qwen3.5-2B-BF16.gguf").expect("Failed to load");
    println!("is_ffi: {}", engine.is_ffi());
}
