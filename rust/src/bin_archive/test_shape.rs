use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: test <model.gguf>");
    let file = GGUFile::open(&path).expect("Failed to open GGUF");

    let info = file.get_tensor_info("blk.0.attn_k.weight").expect("Missing tensor");
    let raw = file.get_tensor_raw("blk.0.attn_k.weight").expect("Missing raw data");

    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    println!("GGUF shape: {:?}", shape);

    let q5 = leafcutter::kernels::q5_k::Matrix {
        rows: shape[0],
        cols: shape[1],
        blocks: leafcutter::kernels::q5_k::blocks_from_bytes(raw),
    };

    let deq = q5.dequantize();
    println!("Rust dequantize len: {}", deq.len());
    
    // Try reshaping as [1024, 3072] (outer, inner)
    let outer = shape[1];
    let inner = shape[0];
    println!("Reshaped as [{}, {}]:", outer, inner);
    println!("First row first 10: {:?}", &deq[..10]);
    println!("First row last 10: {:?}", &deq[inner-10..inner]);
    println!("Last row first 10: {:?}", &deq[(outer-1)*inner..(outer-1)*inner+10]);
}
