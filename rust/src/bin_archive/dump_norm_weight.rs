use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "../models/Qwen3.5-2B-BF16.gguf".to_string());
    let file = GGUFile::open(&path).expect("open gguf");
    
    let info = file.get_tensor_info("blk.0.attn_norm.weight").expect("find");
    let raw = file.get_tensor_raw("blk.0.attn_norm.weight").expect("read");
    
    println!("blk.0.attn_norm.weight: dims={:?}, typ={}", info.dimensions, info.typ);
    
    // Dequantize based on type
    let data: Vec<f32> = match info.typ {
        0 => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        1 => raw.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        12 => raw.chunks_exact(2).map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        _ => {
            println!("Unsupported type {}, cannot dequantize", info.typ);
            return;
        }
    };
    
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let abs_mean = data.iter().map(|&v| v.abs()).sum::<f32>() / data.len() as f32;
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    println!("Dequantized: len={}, mean={:.6}, abs_mean={:.6}, min={:.6}, max={:.6}", 
        data.len(), mean, abs_mean, min, max);
    println!("First 10 values: {:?}", &data[..10]);
}
