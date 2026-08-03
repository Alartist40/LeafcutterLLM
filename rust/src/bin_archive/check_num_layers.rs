use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = GGUFile::open(&path).unwrap();
    for key in ["qwen35.block_count", "llama.block_count", "general.architecture"] {
        if let Some(v) = file.get_metadata_int(key) {
            println!("{}: {}", key, v);
        } else {
            println!("{}: NOT FOUND", key);
        }
    }
}
