use leafcutter::model::loader::GGUFModel;

fn main() {
    for path in &[
        "/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf",
        "/home/xander/Downloads/models/ornith-1.0-9b-Q6_K.gguf",
        "/home/xander/Downloads/models/ornith-1.0-9b-Q8_0.gguf",
    ] {
        eprintln!("=== {} ===", path);
        let model = GGUFModel::load(path).unwrap();
        let info = model.file.get_tensor_info("blk.0.attn_qkv.weight").unwrap();
        eprintln!("blk.0.attn_qkv.weight: shape={:?} qtype={}", info.dimensions, info.typ);
    }
}
