use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_ssm_meta <model.gguf>");
    let model = GGUFModel::load(&path).unwrap();
    let file = &model.file;
    for key in ["qwen35.ssm.n_groups", "qwen35.ssm.time_step_rank", "qwen35.ssm.d_inner", "qwen35.ssm.d_state"] {
        if let Some(v) = file.get_metadata_int(key) {
            println!("{}: {}", key, v);
        } else {
            println!("{}: NOT FOUND", key);
        }
    }
}
