use leafcutter::model::loader::GGUFModel;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let model = GGUFModel::load(path).unwrap();
    println!("=== Layer 0 tensors ===");
    for t in &model.file.tensors {
        if t.name.starts_with("blk.0.") {
            println!("  {}", t.name);
        }
    }
    println!("=== Layer 4 tensors (likely attention) ===");
    for t in &model.file.tensors {
        if t.name.starts_with("blk.4.") {
            println!("  {}", t.name);
        }
    }
    println!("=== Layer 8 tensors ===");
    for t in &model.file.tensors {
        if t.name.starts_with("blk.8.") {
            println!("  {}", t.name);
        }
    }
}
