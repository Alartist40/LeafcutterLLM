use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = GGUFile::open(&path).unwrap();
    let layer = std::env::args().nth(2).unwrap().parse::<usize>().unwrap();
    let prefix = format!("blk.{}", layer);
    
    let mut names: Vec<String> = file.tensors.iter()
        .filter(|t| t.name.starts_with(&prefix))
        .map(|t| t.name.clone())
        .collect();
    names.sort();
    
    for name in names {
        if let Some(info) = file.get_tensor_info(&name) {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            println!("{}: shape={:?} dtype={}", name, dims, info.typ);
        }
    }
}
