use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: list_gguf_tensors <path>");
    let file = GGUFile::open(&path).expect("Failed to open GGUF");
    
    println!("Architecture: {:?}", file.metadata.get("general.architecture"));
    println!("\nTensor names (first 50):");
    for tensor in file.tensors.iter().take(50) {
        let dims: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
        println!("  {}: {:?} type={}", tensor.name, dims, tensor.typ);
    }
}
