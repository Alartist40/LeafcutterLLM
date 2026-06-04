fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = leafcutter::model::gguf::GGUFile::open(&path).unwrap();
    for t in &file.tensors {
        if t.name.contains("attn_q.weight") || t.name.contains("attn_k.weight") || t.name.contains("attn_v.weight") || t.name.contains("attn_output.weight") {
            println!("{}: type={}", t.name, t.typ);
        }
    }
}
