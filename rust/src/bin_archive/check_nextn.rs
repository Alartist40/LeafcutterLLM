use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = GGUFile::open(&path).unwrap();
    for key in ["qwen35.nextn_predict_layers", "llama.nextn_predict_layers", "qwen35.full_attention_interval", "llama.full_attention_interval"] {
        if let Some(v) = file.get_metadata_int(key) {
            println!("{}: {}", key, v);
        } else {
            println!("{}: NOT FOUND", key);
        }
    }
    
    // Count actual layers
    let mut layer_count = 0;
    for i in 0..100 {
        if file.get_tensor_info(&format!("blk.{}", i)).is_some() {
            layer_count += 1;
        } else {
            break;
        }
    }
    println!("Actual layer tensors: {}", layer_count);
}
