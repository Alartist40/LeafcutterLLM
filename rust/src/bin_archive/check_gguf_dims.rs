use leafcutter::model::gguf::GGUFile;

fn main() {
    let file = GGUFile::open("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
    for name in ["token_embd.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.attn_q.weight", "output_norm.weight"] {
        if let Some(info) = file.get_tensor_info(name) {
            println!("{}: type={} dims={:?}", name, info.typ, info.dimensions);
        }
    }
}
