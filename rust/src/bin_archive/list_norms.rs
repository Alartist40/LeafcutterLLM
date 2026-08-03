use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: list_norms <model.gguf>");
    let model = GGUFModel::load(&path).unwrap();
    let file = &model.file;
    
    for tensor in &file.tensors {
        if tensor.name.contains("norm") {
            println!("{}", tensor.name);
        }
    }
}
