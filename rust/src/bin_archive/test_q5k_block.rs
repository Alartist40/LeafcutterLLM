use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: test <model.gguf>");
    let file = GGUFile::open(&path).expect("Failed to open GGUF");

    let info = file.get_tensor_info("blk.0.attn_q.weight").expect("Missing tensor");
    let raw = file.get_tensor_raw("blk.0.attn_q.weight").expect("Missing raw data");

    println!("Tensor: blk.0.attn_q.weight");
    println!("  dims: {:?}", info.dimensions);
    println!("  typ: {}", info.typ);

    let block = leafcutter::kernels::q5_k::Block::from_bytes(&raw[..176]);
    println!("Rust d: {} dmin: {}", block.d, block.dmin);
    println!("Rust scales: {:?}", &block.scales[..12]);
    println!("Rust qh first 8: {:?}", &block.qh[..8]);
    println!("Rust ql first 8: {:?}", &block.ql[..8]);

    let mut deq = [0.0f32; 256];
    block.dequantize(&mut deq);
    println!("Rust first 32 values: {:?}", &deq[..32]);
}
