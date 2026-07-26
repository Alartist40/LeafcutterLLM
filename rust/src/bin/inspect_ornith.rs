// Inspects a GGUF model's metadata + MoE expert tensor shapes without loading weights.
// Usage: cargo run --release --bin inspect_ornith -- <path-to-model.gguf>

use leafcutter::model::gguf::GGUFile;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = if args.len() >= 2 {
        &args[1]
    } else {
        "/home/xander/Downloads/models/ornith-1.0-35b-Q4_K_M.gguf"
    };

    eprintln!("Opening {}...", path);
    let file = match GGUFile::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: {:?}", e);
            std::process::exit(1);
        }
    };

    // Print key metadata
    let keys = [
        "general.architecture",
        "general.name",
        "qwen35moe.block_count",
        "qwen35moe.embedding_length",
        "qwen35moe.expert_count",
        "qwen35moe.expert_used_count",
        "qwen35moe.expert_feed_forward_length",
        "qwen35moe.expert_shared_feed_forward_length",
        "qwen35moe.full_attention_interval",
        "qwen35moe.attention.head_count",
        "qwen35moe.attention.head_count_kv",
        "qwen35moe.attention.key_length",
    ];
    println!("\n=== Metadata ===");
    for k in &keys {
        if let Some(v) = file.metadata.get(*k) {
            println!("  {} = {:?}", k, v);
        }
    }

    // Print MoE expert tensor shapes for layers 0 and 3
    println!("\n=== Tensor shapes (MoE-relevant) ===");
    let moe_suffixes = [
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_inp_shexp.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
    ];
    for layer in [0usize, 3usize, 39usize] {
        println!("\n-- blk.{} --", layer);
        for s in &moe_suffixes {
            let name = format!("blk.{}.{}", layer, s);
            if let Some(info) = file.get_tensor_info(&name) {
                println!("  {}: dims={:?} type={}", name, info.dimensions, info.typ);
            }
        }
    }
    println!("\nTotal tensors: {}", file.tensors.len());
}
