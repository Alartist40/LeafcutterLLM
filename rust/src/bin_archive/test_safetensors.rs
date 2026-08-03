//! Test: load safetensors from the Ornith model and print tensor info.
use leafcutter::safetensors_loader::Shards;
use std::path::Path;

fn main() {
    let dir = Path::new("/home/xander/Downloads/models/ornith safetensor");
    println!("Loading safetensors from {}", dir.display());

    let mut shards = match Shards::open_dir(dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    let names = shards.tensor_names();
    println!("\nTotal tensors: {}", names.len());

    // Print first 10 tensor names with shapes.
    for name in names.iter().take(10) {
        if let Some(meta) = shards.lookup(name) {
            println!("  {} shape={:?} dtype={:?} offset={} nbytes={}",
                name, meta.shape, meta.dtype, meta.offset, meta.nbytes);
        }
    }
    println!("  ...");

    // Try to read a few key tensors and verify values.
    let test_tensors = [
        "model.embed_tokens.weight",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    ];

    for name in &test_tensors {
        match shards.read_tensor_f32(name) {
            Ok(data) => {
                let first8: Vec<f32> = data.iter().take(8).copied().collect();
                println!("\n{name}: {} elements, first 8: {:?}", data.len(), first8);
            }
            Err(e) => println!("\n{name}: READ ERROR: {e}"),
        }
    }
}
