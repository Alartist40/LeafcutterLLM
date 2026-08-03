use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: list_deltanet_weights <model.gguf>");
    let model = GGUFModel::load(&path).unwrap();
    let file = &model.file;
    let layer = std::env::args().nth(2).unwrap_or_else(|| "0".to_string()).parse::<usize>().unwrap();
    
    let prefix = format!("blk.{}", layer);
    for tensor in &file.tensors {
        if tensor.name.starts_with(&prefix) {
            println!("{}", tensor.name);
        }
    }
}
