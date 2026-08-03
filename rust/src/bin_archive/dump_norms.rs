use leafcutter::model::loader::GGUFModel;

fn main() {
    let model = GGUFModel::load("/home/xander/Documents/portfolio/LeafcutterLLM/models/Qwen3.5-0.8B-Q4_0.gguf").expect("load");
    for name in ["blk.0.attn_norm.weight", "blk.0.post_attention_norm.weight", "model.norm.weight"] {
        if let Some(info) = model.file.get_tensor_info(name) {
            if let Some(raw) = model.file.get_tensor_raw(name) {
                let shape_gguf: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                let shape_data = if shape_gguf.len() == 2 {
                    vec![shape_gguf[1], shape_gguf[0]]
                } else {
                    shape_gguf.clone()
                };
                let mut tensor = GGUFModel::dequantize(raw, info.typ, shape_data).expect("deq");
                if tensor.shape.len() == 2 {
                    tensor = tensor.transpose();
                }
                let out_name = name.replace(".", "_");
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(tensor.data.as_ptr() as *const u8, tensor.data.len() * 4)
                };
                std::fs::write(&format!("{}.bin", out_name), bytes).unwrap();
                println!("{}: shape={:?} mean={:.4} min={:.4} max={:.4}", name, tensor.shape,
                    tensor.data.iter().sum::<f32>() / tensor.data.len() as f32,
                    tensor.data.iter().cloned().fold(f32::INFINITY, f32::min),
                    tensor.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
            }
        }
    }
}
