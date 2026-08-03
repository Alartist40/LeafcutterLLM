use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_final_norm <model.gguf>");
    let model = GGUFModel::load(&path).unwrap();
    let file = &model.file;
    
    for name in ["output_norm.weight", "model.norm.weight"] {
        if let Some(info) = file.get_tensor_info(name) {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            let data = file.get_tensor_row_f32(name, 0).unwrap();
            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            println!("{}: shape={:?} min={:.4} max={:.4} mean={:.6}", name, dims, min, max, mean);
        }
    }
}
