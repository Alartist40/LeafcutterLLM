use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_layer <path>");
    let model = GGUFModel::load(&path).expect("Failed to load model");
    let layer = model.load_layer(0).expect("Failed to load layer 0");
    
    println!("Loaded weights for layer 0:");
    for name in layer.keys() {
        println!("  {}", name);
    }
}
