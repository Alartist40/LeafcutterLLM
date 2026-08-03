use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_qkv <model.gguf>");
    let model = GGUFModel::load(&path).unwrap();
    let file = &model.file;
    let layer = std::env::args().nth(2).unwrap_or_else(|| "0".to_string()).parse::<usize>().unwrap();
    
    for name in [format!("blk.{}.attn_qkv.weight", layer), format!("blk.{}.ssm_conv1d.weight", layer)] {
        if let Some(info) = file.get_tensor_info(&name) {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            println!("{}: shape={:?}", name, dims);
        }
    }
}
