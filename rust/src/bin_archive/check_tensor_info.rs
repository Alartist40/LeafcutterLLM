fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = leafcutter::model::gguf::GGUFile::open(&path).unwrap();
    for t in &file.tensors {
        if t.name == "token_embd.weight" || t.name == "output.weight" {
            let info = file.get_tensor_info(&t.name).unwrap();
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            let raw = file.get_tensor_raw(&t.name).unwrap();
            println!("{}: type={} dims={:?} raw_len={}", t.name, t.typ, dims, raw.len());
        }
    }
}
