use leafcutter::inference::engine::Engine;

fn main() {
    let path = "/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf";
    let engine = Engine::load(path).unwrap();
    for &id in &[9707usize, 83, 81, 72, 248046, 248044, 248068, 248069, 9707] {
        let s = engine.decode(&[id]);
        eprintln!("id={} -> {:?}", id, s);
    }
}
