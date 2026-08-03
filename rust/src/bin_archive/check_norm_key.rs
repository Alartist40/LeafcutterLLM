use leafcutter::model::gguf::GGUFile;
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&path).expect("open");
    // Test all common final-norm key names
    for key in ["model.norm.weight", "model.norm", "output_norm.weight", "output_norm", "norm.weight", "model.final_norm.weight"] {
        let exists = file.tensors.iter().any(|t| t.name == key);
        println!("{}: {}", key, if exists { "EXISTS" } else { "MISSING" });
    }
    // Print first 30 tensor names that start with model
    println!("\nFirst 30 'model.*' tensors:");
    for t in file.tensors.iter().filter(|t| t.name.starts_with("model") && !t.name.contains("blk.")).take(30) {
        println!("  {}", t.name);
    }
}
