use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: dump_all_tensors <gguf>");
    let file = GGUFile::open(&path).expect("open");
    for t in &file.tensors {
        if t.name.starts_with("blk.") {
            println!("{} {:?}", t.name, t.dimensions);
        }
    }
}
