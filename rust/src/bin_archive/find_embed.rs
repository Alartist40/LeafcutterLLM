//! Test: find the embed and lm_head tensor names in Ornith safetensors.
use leafcutter::safetensors_loader::Shards;
use std::path::Path;

fn main() {
    let dir = Path::new("/home/xander/Downloads/models/ornith safetensor");
    let shards = Shards::open_dir(dir).unwrap();
    for name in shards.tensor_names() {
        if name.contains("embed") || name.contains("lm_head") || name.contains("model.norm") || name == "model.norm.weight" {
            if let Some(meta) = shards.lookup(name) {
                println!("{} shape={:?}", name, meta.shape);
            }
        }
    }
}
