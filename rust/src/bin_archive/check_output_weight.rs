use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let file = GGUFile::open(path).unwrap();
    
    for name in &["output.weight", "lm_head.weight", "token_embd.weight"] {
        if let Some(info) = file.get_tensor_info(name) {
            println!("{}: dims={:?}, type={}", name, info.dimensions, info.typ);
        } else {
            println!("{}: NOT FOUND", name);
        }
    }
}
