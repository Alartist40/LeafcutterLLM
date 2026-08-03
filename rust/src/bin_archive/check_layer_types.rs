use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_layer_types <model.gguf>");
    let model = GGUFModel::load(&path).unwrap();
    let file = &model.file;
    for layer_idx in 0..30 {
        let has_q = file.get_tensor_info(&format!("blk.{}.attn_q.weight", layer_idx)).is_some();
        let has_qkv = file.get_tensor_info(&format!("blk.{}.attn_qkv.weight", layer_idx)).is_some();
        let has_ssm = file.get_tensor_info(&format!("blk.{}.ssm_alpha.weight", layer_idx)).is_some();
        if has_q {
            println!("Layer {}: standard attention", layer_idx);
        } else if has_qkv {
            println!("Layer {}: DeltaNet (fused QKV)", layer_idx);
        } else if has_ssm {
            println!("Layer {}: SSM", layer_idx);
        }
    }
}
