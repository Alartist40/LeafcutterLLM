use leafcutter::model::gguf::GGUFile;

fn main() {
    let file = GGUFile::open("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
    for name in ["blk.1.attn_k.weight", "blk.1.attn_q.weight", "blk.1.ffn_gate.weight"] {
        if let Some(info) = file.get_tensor_info(name) {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            println!("{}: rust_dims={:?}", name, dims);
        }
    }
}
