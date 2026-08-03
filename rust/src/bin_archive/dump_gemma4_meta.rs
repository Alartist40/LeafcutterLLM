//! Dump ALL GGUF metadata keys for a given file, with full values.
//! Usage: cargo run --release --bin dump_gemma4_meta -- <path_to.gguf>

use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: dump_gemma4_meta <path>");
    let model = GGUFile::open(&path).expect("Failed to load GGUF");

    let mut keys: Vec<_> = model.metadata.keys().collect();
    keys.sort();

    for key in keys {
        if let Some(v) = model.metadata.get(key) {
            let s = format!("{:?}", v);
            // Print full value (no truncation for arrays — let user see all elts)
            println!("{} = {}", key, s);
        }
    }
}
