use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = GGUFile::open(&path).unwrap();
    let layer = std::env::args().nth(2).unwrap().parse::<usize>().unwrap();
    let prefix = format!("blk.{}", layer);
    
    // Check GGUF tensor names
    for t in &file.tensors {
        if t.name.starts_with(&prefix) && t.name.contains("gate") {
            println!("GGUF: {}", t.name);
        }
    }
    
    // Check loaded names
    let model = leafcutter::model::loader::GGUFModel::load(&path).unwrap();
    let weights = model.load_layer(layer).unwrap();
    for name in weights.keys() {
        if name.contains("gate") {
            println!("Loaded: {}", name);
        }
    }
}
