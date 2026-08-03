use leafcutter::model::loader::GGUFModel;
use leafcutter::model::gguf::GGUFValue;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let model = GGUFModel::load(path).unwrap();
    // Print dtype info for layer 0 DeltaNet tensors
    for name in &["self_attn.qkv_proj.weight", "attn_gate.weight", "ssm_alpha.weight",
                  "ssm_beta.weight", "ssm_a", "ssm_dt.bias", "ssm_conv1d.weight",
                  "ssm_norm.weight", "ssm_out.weight"] {
        if let Some(info) = model.file.get_tensor_info(name) {
            println!("  {}: shape={:?} qtype={:?}",
                name,
                info.dimensions,
                info.typ);
        } else {
            println!("  {}: NOT FOUND", name);
        }
    }
    // Check norm_eps in metadata
    if let Some(GGUFValue::F32(v)) = model.file.metadata.get("qwen35.attention.layer_norm_rms_epsilon") {
        println!("norm_eps = {}", v);
    } else if let Some(GGUFValue::F32(v)) = model.file.metadata.get("attention.layer_norm_rms_epsilon") {
        println!("norm_eps = {}", v);
    }
}
