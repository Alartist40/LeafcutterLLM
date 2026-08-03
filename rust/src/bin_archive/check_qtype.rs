use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage");
    let file = GGUFile::open(&path).expect("open");
    for name in ["blk.0.attn_qkv.weight", "blk.0.ssm_out.weight", "blk.0.attn_gate.weight"] {
        if let Some(info) = file.get_tensor_info(name) {
            println!("{}: typ={}, dims={:?}", name, info.typ, info.dimensions);
        }
    }
}
