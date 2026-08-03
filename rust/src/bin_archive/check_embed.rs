fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = leafcutter::model::gguf::GGUFile::open(&path).unwrap();
    let mut row = vec![0.0f32; 1024];
    file.get_tensor_row_f32_into("token_embd.weight", 17, &mut row).unwrap();
    println!("token 17 embed: max={:.4} min={:.4} mean={:.4}", 
        row.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        row.iter().cloned().fold(f32::INFINITY, f32::min),
        row.iter().sum::<f32>() / row.len() as f32);
    
    let mut row2 = vec![0.0f32; 1024];
    file.get_tensor_row_f32_into("token_embd.weight", 10, &mut row2).unwrap();
    println!("token 10 embed: max={:.4} min={:.4} mean={:.4}", 
        row2.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        row2.iter().cloned().fold(f32::INFINITY, f32::min),
        row2.iter().sum::<f32>() / row2.len() as f32);
}
