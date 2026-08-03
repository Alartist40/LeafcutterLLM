use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_attn_weights <model.gguf>");
    let model = GGUFModel::load(&path).unwrap();
    let file = &model.file;
    let layer = std::env::args().nth(2).unwrap_or_else(|| "3".to_string()).parse::<usize>().unwrap();
    
    for name in ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                 "attn_gate.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                 "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
        let full = format!("blk.{}.{}", layer, name);
        if let Some(info) = file.get_tensor_info(&full) {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            println!("{}: shape={:?}", full, dims);
        } else {
            println!("{}: NOT FOUND", full);
        }
    }
}
