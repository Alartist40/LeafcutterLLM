fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = leafcutter::model::gguf::GGUFile::open(&path).unwrap();
    for t in &file.tensors {
        if t.name.contains("output") || t.name.contains("token_embd") || t.name.contains("lm_head") {
            println!("{}: type={}", t.name, t.typ);
        }
    }
}
