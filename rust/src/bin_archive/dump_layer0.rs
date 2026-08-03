use leafcutter::model::gguf::GGUFile;
use leafcutter::model::quant::QuantType;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let file = GGUFile::open(path).unwrap();
    
    for t in &file.tensors {
        if t.name.starts_with("blk.0.") || t.name == "token_embd.weight" || t.name == "output_norm.weight" {
            let qtype = QuantType::from_u32(t.typ).map(|q| format!("{:?}", q)).unwrap_or_else(|| format!("type_{}", t.typ));
            println!("{}: dims={:?}, type={}", t.name, t.dimensions, qtype);
        }
    }
}
