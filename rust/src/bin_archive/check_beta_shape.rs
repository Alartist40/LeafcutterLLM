use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_beta_shape <model.gguf>");
    let model = GGUFModel::load(&path).unwrap();
    let file = &model.file;
    let layer = std::env::args().nth(2).unwrap_or_else(|| "0".to_string()).parse::<usize>().unwrap();
    
    for name in ["ssm_beta.weight", "ssm_alpha.weight", "ssm_dt.bias", "ssm_a"] {
        let full = format!("blk.{}.{}", layer, name);
        if let Some(info) = file.get_tensor_info(&full) {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            println!("{}: shape={:?}", full, dims);
        }
    }
}
