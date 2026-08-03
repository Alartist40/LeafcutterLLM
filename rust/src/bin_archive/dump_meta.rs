use leafcutter::model::gguf::GGUFile;
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let file = GGUFile::open(&path).expect("open");
    let mut keys: Vec<_> = file.metadata.keys().collect();
    keys.sort();
    for k in keys.iter().filter(|k| k.contains("epsilon") || k.contains("eps") || k.contains("norm") || k.contains("ssm") || k.contains("conv") || k.contains("dt")) {
        println!("{}: {:?}", k, file.metadata.get(*k).unwrap());
    }
}
