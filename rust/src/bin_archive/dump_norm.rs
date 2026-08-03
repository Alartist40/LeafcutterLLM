
use leafcutter::model::loader::GGUFModel;
fn main() {
    let model = GGUFModel::load("../models/Qwen3.5-0.8B-Q4_0.gguf").unwrap();
    let file = &model.file;
    let data = file.get_tensor_row_f32("blk.0.attn_norm.weight", 0).unwrap();
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    std::fs::write("attn_norm_layer0.bin", bytes).unwrap();
}
