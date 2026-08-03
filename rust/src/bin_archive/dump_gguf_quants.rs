//! dump_gguf_quants — print the quant-type composition of a GGUF file,
//! plus a per-tensor list (first N) and a flag for unsupported types.
//!
//! Usage:
//!     cargo run --release --bin dump_gguf_quants -- <model-path> [max-tensors]

use leafcutter::model::gguf::{calculate_tensor_size, GGUFile};
use leafcutter::model::quant::QuantType;
use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("Usage: dump_gguf_quants <model-path> [max-tensors]");
    let max_tensors: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);

    let file = GGUFile::open(path).expect("failed to open GGUF");

    let mut counts: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut unsupported = Vec::new();
    let mut shown = 0;

    for t in &file.tensors {
        let qtype = QuantType::from_u32(t.typ)
            .map(|q| format!("{:?}", q))
            .unwrap_or_else(|| {
                unsupported.push((t.name.clone(), t.typ));
                format!("type_{} (UNSUPPORTED)", t.typ)
            });
        let entry = counts.entry(qtype.clone()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += calculate_tensor_size(&t.dimensions, t.typ) as u64;

        if shown < max_tensors && !t.name.starts_with("blk.") {
            println!("{}: dims={:?} {}", t.name, t.dimensions, qtype);
            shown += 1;
        }
    }

    println!("\n=== Quant-type composition ({}) ===", path);
    for (qtype, (count, bytes)) in &counts {
        println!("{:>20}  {:>5} tensors  {:>12} bytes ({:.2} GB)", qtype, count, bytes, *bytes as f64 / 1e9);
    }

    if !unsupported.is_empty() {
        println!("\n=== UNSUPPORTED QUANT TYPES (would corrupt/skip weights) ===");
        for (name, typ) in unsupported.iter().take(20) {
            println!("  {}: type {}", name, typ);
        }
    } else {
        println!("\nAll quant types are supported by the Rust dequant kernels.");
    }
}
