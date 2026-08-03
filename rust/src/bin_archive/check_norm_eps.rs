use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = GGUFile::open(&path).unwrap();
    
    for key in ["qwen35.attention.layer_norm_rms_epsilon", "llama.attention.layer_norm_rms_epsilon"] {
        if let Some(v) = file.get_metadata_f32(key) {
            println!("{}: {}", key, v);
        } else {
            println!("{}: NOT FOUND", key);
        }
    }
}
