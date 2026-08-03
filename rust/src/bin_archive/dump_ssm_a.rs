//! Dump ssm_a tensor values to verify sign (positive A_log vs negative -exp(A_log)).
//!
//! Usage: cargo run --release --bin dump_ssm_a -- <model.gguf>

use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: dump_ssm_a <model.gguf>");
    let file = GGUFile::open(&path).expect("Failed to open GGUF");

    // Check layers 0, 1, 2, 3 for ssm_a
    for layer in 0..4 {
        let name = format!("blk.{}.ssm_a", layer);
        if let Some(info) = file.get_tensor_info(&name) {
            println!("{}: dims={:?} type={}", name, info.dimensions, info.typ);
            let raw = file.get_tensor_raw(&name);
            if let Some(data) = raw {
                // ssm_a is typically f32, one value per head
                let n = data.len() / 4;
                println!("  {} values ({} bytes)", n, data.len());
                for i in 0..n.min(8) {
                    let bytes = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
                    let val = f32::from_le_bytes(bytes);
                    println!("  ssm_a[{}] = {:.6} (exp={:.6}, -exp={:.6})", i, val, val.exp(), (-val.exp()));
                }
            }
        } else {
            println!("{}: NOT FOUND", name);
        }

        // Also check ssm_dt.bias
        let dt_name = format!("blk.{}.ssm_dt.bias", layer);
        if let Some(info) = file.get_tensor_info(&dt_name) {
            println!("{}: dims={:?} type={}", dt_name, info.dimensions, info.typ);
            let raw = file.get_tensor_raw(&dt_name);
            if let Some(data) = raw {
                let n = data.len() / 4;
                for i in 0..n.min(4) {
                    let bytes = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
                    let val = f32::from_le_bytes(bytes);
                    println!("  dt_bias[{}] = {:.6}", i, val);
                }
            }
        }

        // Check ssm_alpha (the projection)
        let alpha_name = format!("blk.{}.ssm_alpha.weight", layer);
        if let Some(info) = file.get_tensor_info(&alpha_name) {
            println!("{}: dims={:?} type={}", alpha_name, info.dimensions, info.typ);
        }
        println!();
    }
}
