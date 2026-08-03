use leafcutter::model::gguf::GGUFile;

fn main() {
    let file = GGUFile::open("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
    if let Some(info) = file.get_tensor_info("output.weight") {
        println!("output.weight: type={} dims={:?}", info.typ, info.dimensions);
    }
}
