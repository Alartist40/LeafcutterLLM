use leafcutter::model::gguf::GGUFile;
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&path).expect("open");
    for tensor in &file.tensors {
        let name = &tensor.name;
        if name.contains("norm") && !name.contains("attn_norm") && !name.contains("ffn_norm") {
            println!("{}: dims={:?} type={:?}", name, tensor.dimensions, tensor.typ);
        }
    }
}
