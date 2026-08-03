use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: dump_tensors <gguf>");
    let file = GGUFile::open(&path).expect("open");
    for t in &file.tensors {
        let name = &t.name;
        if name.starts_with("blk.0") || name.starts_with("blk.1") || name.contains("norm") || name.contains("embd") {
            println!("{} {:?}", name, t.dimensions);
        }
    }
}
