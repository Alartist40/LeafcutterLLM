fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = leafcutter::model::gguf::GGUFile::open(&path).unwrap();
    for (k, v) in &file.metadata {
        if k.contains("attention") || k.contains("head") || k.contains("embedding") || k.contains("hidden") {
            println!("{} = {:?}", k, v);
        }
    }
}
