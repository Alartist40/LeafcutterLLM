use leafcutter::model::gguf::GGUFile;
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&path).expect("open");
    for key in ["qwen35.attention.layer_norm_rms_epsilon", "qwen36.attention.layer_norm_rms_epsilon", "qwen35.ssm.layer_norm_rms_epsilon", "qwen35.embedding.rms_norm_epsilon", "qwen35.embedding.layer_norm_epsilon", "qwen35.ssm.group_norm_epsilon", "qwen35.ssm.norm_eps", "norm_epsilon"] {
        if let Some(v) = file.metadata.get(key) {
            println!("{}: {:?}", key, v);
        }
    }
}
