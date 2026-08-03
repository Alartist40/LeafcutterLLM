use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: test <model.gguf>");
    let file = GGUFile::open(&path).expect("Failed to open GGUF");

    let info = file.get_tensor_info("token_embd.weight").expect("Missing tensor");
    let raw = file.get_tensor_raw("token_embd.weight").expect("Missing raw data");

    let shape_gguf: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    println!("GGUF shape: {:?}", shape_gguf);

    let t = leafcutter::model::loader::GGUFModel::dequantize(raw, info.typ, shape_gguf.clone()).unwrap();
    println!("Tensor shape: {:?}", t.shape);
    println!("First 10: {:?}", &t.data[..10]);
    println!("Last 10: {:?}", &t.data[t.data.len()-10..]);
}
