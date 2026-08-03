use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_gate_weights <model.gguf>");
    let model = GGUFModel::load(&path).unwrap();
    let file = &model.file;
    
    for layer_idx in 0..30 {
        let name = format!("blk.{}.attn_gate.weight", layer_idx);
        if let Some(info) = file.get_tensor_info(&name) {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            println!("{}: shape={:?}", name, dims);
        }
    }
}
