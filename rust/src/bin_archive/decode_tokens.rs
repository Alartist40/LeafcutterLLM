//! Decode a list of token IDs using the GGUF tokenizer.
//! Reads space-separated IDs from stdin. Usage:
//!     cargo run --release --bin decode_tokens -- <gguf> < <ids.txt>

use leafcutter::model::gguf::{GGUFile, GGUFValue};
use std::collections::HashMap;
use std::io::Read;

fn main() {
    let path = std::env::args().nth(1).expect("usage: decode_tokens <gguf>");
    let file = GGUFile::open(&path).expect("Failed to open GGUF");

    // Read vocab
    let vocab = if let Some(GGUFValue::Array(arr)) = file.metadata.get("tokenizer.ggml.tokens") {
        arr.iter()
            .filter_map(|v| {
                if let GGUFValue::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    } else {
        eprintln!("No tokenizer.ggml.tokens");
        std::process::exit(1);
    };

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    let line_count = input.lines().count();
    eprintln!("Read {} lines from stdin", line_count);

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Try JSON-like "[1, 2, 3]" or space-separated
        let ids: Vec<usize> = if line.starts_with('[') {
            line.trim_start_matches('[')
                .trim_end_matches(']')
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect()
        } else {
            line.split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect()
        };
        let pieces: Vec<String> = ids
            .iter()
            .map(|&t| {
                vocab
                    .get(t)
                    .cloned()
                    .unwrap_or_else(|| format!("<unk:{t}>"))
            })
            .collect();
        println!("{}\n---", pieces.join("").replace("\u{0120}", " "));
    }
}
