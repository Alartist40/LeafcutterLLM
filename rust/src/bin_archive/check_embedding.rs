use leafcutter::model::gguf::GGUFile;
use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_embedding <gguf>");
    let model = GGUFModel::load(&path).expect("load");
    
    // Read embedding for token 17 ('2')
    let row = model.file.get_tensor_row_f32("token_embd.weight", 17).expect("read");
    println!("Token 17 embedding: len={}, first_5={:?}", row.len(), &row[..5.min(row.len())]);
    
    // Read embedding for token 10 ('+')
    let row = model.file.get_tensor_row_f32("token_embd.weight", 10).expect("read");
    println!("Token 10 embedding: len={}, first_5={:?}", row.len(), &row[..5.min(row.len())]);
    
    // Check if embedding is quantized
    if let Some(info) = model.file.get_tensor_info("token_embd.weight") {
        println!("Embedding type: {}", info.typ);
        println!("Embedding shape: {:?}", info.dimensions);
    }
}
