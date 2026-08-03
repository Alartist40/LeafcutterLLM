use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = "/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf";
    let model = GGUFModel::load(path).unwrap();
    // Get embedding for token 760 ("The")
    let row = model.file.get_tensor_row_f32("token_embd.weight", 760).unwrap();
    println!("Embed[760] first 10: {:?}", &row[..10]);
    println!("Embed[760] norm: {:.4}", row.iter().map(|x| x*x).sum::<f32>().sqrt());
    // Same for 6511
    let row = model.file.get_tensor_row_f32("token_embd.weight", 6511).unwrap();
    println!("Embed[6511] first 10: {:?}", &row[..10]);
    println!("Embed[6511] norm: {:.4}", row.iter().map(|x| x*x).sum::<f32>().sqrt());
}
