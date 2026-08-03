use std::collections::HashMap;
use leafcutter::model::loader::GGUFModel;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let model = GGUFModel::load(path).unwrap();
    let layer_weights = model.load_layer(0).unwrap();
    let names: Vec<&String> = layer_weights.keys().collect();
    let mut sorted = names.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    sorted.sort();
    println!("Layer 0 weight names:");
    for n in &sorted {
        println!("  {}", n);
    }
    // Check the FFN names
    println!();
    println!("FFN-style names found:");
    for key in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
        if layer_weights.contains_key(key) {
            println!("  {}: FOUND", key);
        }
    }
}
