use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let model = GGUFModel::load(path).unwrap();
    
    for layer_idx in 0..3 {
        let layer = model.load_layer(layer_idx).unwrap();
        println!("\nLayer {} keys:", layer_idx);
        let mut keys: Vec<_> = layer.keys().collect();
        keys.sort();
        for k in keys {
            println!("  {}", k);
        }
    }
}
