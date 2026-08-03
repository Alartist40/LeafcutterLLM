use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: inspect_gguf <model.gguf>");
    let file = GGUFile::open(&path).expect("Failed to open GGUF");

    let summary = file.quant_summary();
    println!("{}", summary.report());
    println!("Fully supported: {}", summary.is_fully_supported());
}
