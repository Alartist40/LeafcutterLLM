fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = leafcutter::model::gguf::GGUFile::open(&path).unwrap();
    for i in [0, 3] {
        if let Some(info) = file.get_tensor_info(&format!("blk.{}.ssm_a", i)) {
            let raw = file.get_tensor_raw(&format!("blk.{}.ssm_a", i)).unwrap();
            let vals: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            println!("blk.{}.ssm_a: {:?}", i, &vals[..vals.len().min(8)]);
        }
    }
}
