use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = "/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf";
    let model = GGUFModel::load(path).unwrap();
    eprintln!("=== ALL tensors at blk.11 ===");
    for t in &model.file.tensors {
        if t.name.starts_with("blk.11.") {
            eprintln!("  {:<35} shape={:?}", t.name, t.dimensions);
        }
    }
}
