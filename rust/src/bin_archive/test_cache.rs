use leafcutter::inference::engine::Engine;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let engine = Engine::load(path).unwrap();
    
    let w1 = engine.model.load_layer(0).unwrap();
    let w2 = engine.model.load_layer(0).unwrap();
    
    for (name, t1) in &w1 {
        let t2 = w2.get(name).unwrap();
        if t1.data.len() != t2.data.len() {
            println!("{}: different data lengths! {} vs {}", name, t1.data.len(), t2.data.len());
            continue;
        }
        let mut max_diff = 0.0f32;
        for i in 0..t1.data.len() {
            let diff = (t1.data[i] - t2.data[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
        println!("{}: max_diff={:.6} (len={})", name, max_diff, t1.data.len());
    }
}
