fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = leafcutter::model::gguf::GGUFile::open(&path).unwrap();
    let prefix = "blk.0.";
    for t in &file.tensors {
        if t.name.starts_with(prefix) && t.name.contains("ssm") {
            let info = file.get_tensor_info(&t.name).unwrap();
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            println!("{}: type={} dims={:?}", t.name, t.typ, dims);
        }
    }
}
