use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_ssm_norm <model.gguf>");
    let model = GGUFModel::load(&path).unwrap();
    let file = &model.file;
    let layer = std::env::args().nth(2).unwrap_or_else(|| "0".to_string()).parse::<usize>().unwrap();
    
    let name = format!("blk.{}.ssm_norm.weight", layer);
    if let Some(info) = file.get_tensor_info(&name) {
        let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
        println!("{}: shape={:?}", name, dims);
    } else {
        println!("{}: NOT FOUND", name);
    }
}
