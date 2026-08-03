use leafcutter::model::loader::GGUFModel;
use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_weights <path>");
    let model = GGUFModel::load(&path).expect("Failed to load model");
    let file = &model.file;
    
    for name in ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", 
                 "blk.0.attn_output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"] {
        if let Some(info) = file.get_tensor_info(name) {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            let cols = dims[0];
            let rows = dims.get(1).copied().unwrap_or(1);
            let mut data = Vec::with_capacity(cols * rows);
            for r in 0..rows {
                let row = file.get_tensor_row_f32(name, r).expect("dequant row");
                data.extend_from_slice(&row);
            }
            
            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            let nan_count = data.iter().filter(|&&x| x.is_nan()).count();
            let inf_count = data.iter().filter(|&&x| x.is_infinite()).count();
            
            println!("{}: shape={:?} min={:.4} max={:.4} mean={:.6} nan={} inf={}", 
                name, dims, min, max, mean, nan_count, inf_count);
        }
    }
}
