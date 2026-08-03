use leafcutter::model::loader::GGUFModel;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let model = GGUFModel::load(path).unwrap();
    // Print key shapes for layers 0 (DeltaNet) and 3 (attention?)
    for layer_idx in [0, 1, 2, 3, 4, 7, 11, 31] {
        eprintln!("=== Layer {} ===", layer_idx);
        let prefix = format!("blk.{}", layer_idx);
        for t in &model.file.tensors {
            if t.name.starts_with(&prefix) && (t.name.contains("qkv") || t.name.contains("q.weight")
                || t.name.contains("k.weight") || t.name.contains("v.weight")
                || t.name.contains("output") || t.name.contains("ssm_")) {
                eprintln!("  {:<32} shape={:?}", t.name, t.dimensions);
            }
        }
    }
}
