fn main() {
    let path = std::env::args().nth(1).unwrap();
    let engine = leafcutter::inference::engine::Engine::load(&path).unwrap();
    println!("{:?}", engine.info());
}
