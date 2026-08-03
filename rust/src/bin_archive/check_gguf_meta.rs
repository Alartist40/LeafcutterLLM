//! Quick GGUF metadata inspector
//! Usage: cargo run --release --bin check_gguf_meta -- <path_to.gguf>

use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: check_gguf_meta <path>");
    let model = GGUFile::open(&path).expect("Failed to load GGUF");

    println!("=== GGUF Metadata for: {} ===\n", path);

    // Print key metadata fields
    for key in [
        "general.name",
        "general.architecture",
        "general.finetune",
        "general.basename",
        "general.quantization_version",
        "tokenizer.chat_template",
    ] {
        if let Some(v) = model.metadata.get(key) {
            let s = format!("{:?}", v);
            let truncated = if s.len() > 2000 {
                format!("{}...", &s[..2000])
            } else {
                s
            };
            println!("{} = {}", key, truncated);
        } else {
            println!("{} = <not present>", key);
        }
    }

    // Also print any other keys that look relevant
    println!("\n=== Other relevant keys ===");
    let mut keys: Vec<_> = model.metadata.keys().collect();
    keys.sort();
    for key in keys {
        let lk = key.to_lowercase();
        if lk.contains("chat")
            || lk.contains("template")
            || lk.contains("instruct")
            || lk.contains("base")
            || lk.contains("reason")
            || lk.contains("system")
            || lk.contains("finetune")
        {
            if let Some(v) = model.metadata.get(key) {
                let s = format!("{:?}", v);
                let truncated = if s.len() > 2000 {
                    format!("{}...", &s[..2000])
                } else {
                    s
                };
                println!("{} = {}", key, truncated);
            }
        }
    }
}
