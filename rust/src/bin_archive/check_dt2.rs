use leafcutter::model::loader::GGUFModel;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let model = GGUFModel::load(path).unwrap();
    // Print first layer tensor info by directly searching
    for t in &model.file.tensors {
        if t.name.contains("ssm") || t.name.contains("attn") {
            if t.name.starts_with("blk.0.") || t.name == "blk.0.attn_qkv.weight" {
                println!("  {}: shape={:?} qtype={:?}",
                    t.name, t.dimensions, t.typ);
            }
        }
    }
}
