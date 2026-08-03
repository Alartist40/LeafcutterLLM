use leafcutter::model::gguf::GGUFile;
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&path).expect("open");
    let layer: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(0);
    println!("Layer {}", layer);
    let prefix = format!("blk.{}", layer);
    for t in file.tensors.iter().filter(|t| t.name.starts_with(&prefix)) {
        println!("  {}: dims={:?} type={:?}", t.name, t.dimensions, t.typ);
    }
}
