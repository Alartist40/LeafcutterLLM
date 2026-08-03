use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: test <model.gguf>");
    let file = GGUFile::open(&path).expect("Failed to open GGUF");

    let info = file.get_tensor_info("blk.0.attn_q.weight").expect("Missing tensor");
    let raw = file.get_tensor_raw("blk.0.attn_q.weight").expect("Missing raw data");

    println!("Tensor: blk.0.attn_q.weight");
    println!("  dims: {:?}", info.dimensions);
    println!("  typ: {}", info.typ);

    let block = leafcutter::kernels::iq4_nl::Block::from_bytes(&raw[..18]);
    println!("Rust scale: {}", block.scale);
    println!("Raw bytes: {}", hex::encode(&raw[..18]));

    let mut deq = [0.0f32; 32];
    block.dequantize(&mut deq);
    println!("Rust first 32 values: {:?}", &deq[..]);
}
