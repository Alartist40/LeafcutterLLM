fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = leafcutter::model::gguf::GGUFile::open(&path).unwrap();
    for (k, v) in &file.metadata {
        if k.contains("block_count") || k.contains("layer") || k.contains("nextn") || k.contains("mtp") || k.contains("predict") {
            println!("{} = {:?}", k, v);
        }
    }
}
