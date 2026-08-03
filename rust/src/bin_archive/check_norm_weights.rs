use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_norm_weights <gguf>");
    let file = GGUFile::open(&path).expect("open");
    for li in 0..4 {
        let name = format!("blk.{}.ssm_norm.weight", li);
        if let Some(raw) = file.get_tensor_raw(&name) {
            let vals: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("blk.{}.ssm_norm.weight: n={}, mean={:.4}, min={:.4}, max={:.4}", li, vals.len(), mean, min, max);
            println!("  first 8: {:?}", &vals[..8.min(vals.len())]);
        }
    }
}
