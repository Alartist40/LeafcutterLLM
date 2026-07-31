fn main() {
    let path = "/home/xander/Downloads/models/ornith-1.0-9b-Q8_0.gguf";
    let f = leafcutter::model::gguf::GGUFile::open(path).expect("open");
    for (k, v) in &f.metadata {
        if k.contains("ssm") || k.contains("head") || k.contains("rope") || k.contains("attention.") || k.contains("block_count") || k.contains("embedding") || k.contains("feed_forward") {
            println!("{} = {:?}", k, v);
        }
    }
    println!("=== ssm tensor shapes (blk.0) ===");
    for t in &f.tensors {
        if t.name.starts_with("blk.0.") && (t.name.contains("ssm") || t.name.contains("attn") || t.name.contains("ffn")) {
            println!("{} dims={:?}", t.name, t.dimensions);
        }
    }
}
