use leafcutter::model::gguf::GGUFile;

fn main() {
    let file = GGUFile::open("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
    let row9906 = file.get_tensor_row_f32("token_embd.weight", 9906).unwrap();
    let dot_self: f32 = row9906.iter().map(|v| v * v).sum();
    let rms = (dot_self / row9906.len() as f32).sqrt();
    println!("Token 9906 embed: len={}, dot_self={:.4}, rms={:.4}", row9906.len(), dot_self, rms);
    
    // Also check token 0
    let row0 = file.get_tensor_row_f32("token_embd.weight", 0).unwrap();
    let dot0: f32 = row0.iter().map(|v| v * v).sum();
    let rms0 = (dot0 / row0.len() as f32).sqrt();
    println!("Token 0 embed: dot_self={:.4}, rms={:.4}", dot0, rms0);
    
    // Dot product between 9906 and 0
    let dot_9906_0: f32 = row9906.iter().zip(row0.iter()).map(|(a,b)| a*b).sum();
    println!("dot(9906, 0)={:.4}", dot_9906_0);
}
