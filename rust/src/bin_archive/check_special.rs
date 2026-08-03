use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = GGUFile::open(&path).unwrap();
    for name in ["output.weight", "token_embd.weight", "output_norm.weight"] {
        if let Some(info) = file.get_tensor_info(name) {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            println!("{}: shape={:?} dtype={}", name, dims, info.typ);
        } else {
            println!("{}: NOT FOUND", name);
        }
    }
}
