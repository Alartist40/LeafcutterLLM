use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let model = GGUFModel::load(path).unwrap();
    
    let w1 = model.load_layer(0).unwrap();
    let w2 = model.load_layer(0).unwrap();
    
    let q1 = w1.get("self_attn.q_proj.weight").unwrap();
    let q2 = w2.get("self_attn.q_proj.weight").unwrap();
    
    let mut max_diff = 0.0f32;
    for i in 0..q1.data.len() {
        let diff = (q1.data[i] - q2.data[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("Max diff between two load_layer(0) calls: {}", max_diff);
}
