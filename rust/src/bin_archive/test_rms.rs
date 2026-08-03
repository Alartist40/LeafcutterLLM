use leafcutter::model::gguf::GGUFile;
use leafcutter::model::tensor::Tensor;

fn main() {
    let path = "../models/Qwen3.5-0.8B-Q4_0.gguf";
    let file = GGUFile::open(path).expect("open gguf");
    
    // Load embeddings for tokens 17, 10, 17, 28
    let tokens = vec![17usize, 10, 17, 28];
    let hidden_size = 1024;
    let mut embed_data = vec![0.0f32; tokens.len() * hidden_size];
    for (i, &tok) in tokens.iter().enumerate() {
        let row = file.get_tensor_row_f32("token_embd.weight", tok).expect("read embed");
        embed_data[i * hidden_size..(i+1) * hidden_size].copy_from_slice(&row);
    }
    let embed = Tensor::from_vec(embed_data, vec![tokens.len(), hidden_size]);
    println!("Embed abs_mean: {:.6}", embed.data.iter().map(|&v| v.abs()).sum::<f32>() / embed.data.len() as f32);
    
    // Load norm weight
    let raw = file.get_tensor_raw("blk.0.attn_norm.weight").expect("read norm");
    let info = file.get_tensor_info("blk.0.attn_norm.weight").expect("info");
    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    println!("Norm weight shape: {:?}", shape);
    
    let norm_data: Vec<f32> = raw.chunks_exact(2).map(|b| {
        // Assuming BF16 storage for BF16 model... wait, this is Q4_0 model
        // Actually, norm weights are usually F32 or F16
        // Let me just use the dequantize function
        0.0f32
    }).collect();
    
    // Hmm, I need to know the quant type. Let me just use the model loader.
}
