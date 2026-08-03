//! Dump all layer 0 weights from GGUF to binary files for Python comparison
//!
//! IMPORTANT: GGUF stores 2D tensors as [inner, outer]. The raw bytes are laid out
//! as outer chunks of inner elements. To dequantize correctly, we must pass
//! shape_data = [outer, inner] to dequantize(), then transpose to get the
//! logical [inner, outer] layout that matches PyTorch's [out, in].

use leafcutter::model::loader::GGUFModel;

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../models/Qwen3.5-0.8B-Q4_0.gguf".to_string());

    let model = GGUFModel::load(&model_path).expect("load model");
    let layer_idx = 0;

    let tensors = vec![
        format!("blk.{}.attn_qkv.weight", layer_idx),
        format!("blk.{}.attn_gate.weight", layer_idx),
        format!("blk.{}.ssm_conv1d.weight", layer_idx),
        format!("blk.{}.ssm_alpha.weight", layer_idx),
        format!("blk.{}.ssm_beta.weight", layer_idx),
        format!("blk.{}.ssm_dt.bias", layer_idx),
        format!("blk.{}.ssm_a", layer_idx),
        format!("blk.{}.ssm_norm.weight", layer_idx),
        format!("blk.{}.ssm_out.weight", layer_idx),
    ];

    for name in tensors {
        if let Some(info) = model.file.get_tensor_info(&name) {
            if let Some(raw) = model.file.get_tensor_raw(&name) {
                let shape_gguf: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                let is_2d = shape_gguf.len() == 2;
                
                // GGUF stores 2D as [inner, outer]; dequantize needs [outer, inner]
                let shape_data: Vec<usize> = if is_2d {
                    vec![shape_gguf[1], shape_gguf[0]]
                } else {
                    shape_gguf.clone()
                };
                
                let mut tensor = leafcutter::model::loader::GGUFModel::dequantize(raw, info.typ, shape_data)
                    .expect(&format!("dequantize {}", name));
                
                // For 2D weights, transpose from [outer, inner] to [inner, outer]
                // to match PyTorch's [out, in] layout.
                if is_2d {
                    tensor = tensor.transpose();
                }
                
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(tensor.data.as_ptr() as *const u8, tensor.data.len() * 4)
                };
                let out_name = name.replace(".", "_");
                std::fs::write(&format!("{}.bin", out_name), bytes).unwrap();
                println!("Dumped {}: logical shape={:?}, mean={:.6}", name, tensor.shape,
                    tensor.data.iter().sum::<f32>() / tensor.data.len() as f32);
            }
        } else {
            println!("MISSING: {}", name);
        }
    }
}
