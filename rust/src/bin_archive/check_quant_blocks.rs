use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_quant_blocks <gguf>");
    let file = GGUFile::open(&path).expect("open");
    
    for name in &["blk.0.attn_qkv.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_out.weight"] {
        if let Some(info) = file.get_tensor_info(name) {
            if let Some(raw) = file.get_tensor_raw(name) {
                let typ = info.typ;
                let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                println!("{}: typ={} dims={:?} raw_len={}", name, typ, dims, raw.len());
            }
        }
    }
}
