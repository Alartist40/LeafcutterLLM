fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = leafcutter::model::gguf::GGUFile::open(&path).unwrap();
    for i in 0..4 {
        if let Some(t) = file.tensors.iter().find(|t| t.name == format!("blk.{}.ssm_conv1d.weight", i)) {
            println!("blk.{}.ssm_conv1d.weight: type={}", i, t.typ);
        }
    }
}
