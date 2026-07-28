use leafcutter::model::gguf::GGUFile;
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&path).expect("open");
    let embed = file.get_tensor_row_f32("token_embd.weight", 760).unwrap();
    let norm = file.get_tensor_row_f32("blk.0.attn_norm.weight", 0).unwrap();
    let mean_sq: f32 = embed.iter().map(|&v| v * v).sum::<f32>() / embed.len() as f32;
    let scale = 1.0 / (mean_sq + 1e-6).sqrt();
    let max: f32 = embed.iter().map(|&v| (v * scale * norm[0]).abs()).fold(f32::NEG_INFINITY, f32::max);
    println!("pre_norm of [0]: embed[0]={} norm[0]={}", embed[0], norm[0]);
    println!("mean_sq={:.6} scale={:.6}", mean_sq, scale);
    println!("pre_norm[0]={:.6}", embed[0] * scale * norm[0]);
    println!("max abs pre_norm = {:.5}", max);
}
