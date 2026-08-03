use leafcutter::model::gguf::GGUFile;
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&path).expect("open");
    for key in ["qwen35.rope.freq_base", "qwen35.rope.dim", "qwen35.attention.rope_dim", "qwen35.rope.scaling.type", "qwen35.rope.scaling.factor"] {
        if let Some(v) = file.metadata.get(key) {
            println!("{}: {:?}", key, v);
        }
    }
}
