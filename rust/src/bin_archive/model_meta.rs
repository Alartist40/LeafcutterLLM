use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: model_meta <gguf>");
    let file = GGUFile::open(&path).expect("open");
    for (k, v) in &file.metadata {
        if k.contains("arch") || k.contains("type") || k.contains("name") || k.contains("attention") || k.contains("ssm") {
            println!("{} = {:?}", k, v);
        }
    }
}
