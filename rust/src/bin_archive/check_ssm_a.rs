fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = leafcutter::model::gguf::GGUFile::open(&path).unwrap();
    for i in 0..4 {
        if let Some(info) = file.get_tensor_info(&format!("blk.{}.ssm_a", i)) {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            println!("blk.{}.ssm_a: dims={:?} total={}", i, dims, dims.iter().product::<usize>());
        }
    }
    for i in 0..4 {
        if let Some(info) = file.get_tensor_info(&format!("blk.{}.attn_qkv.weight", i)) {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            println!("blk.{}.attn_qkv.weight: dims={:?}", i, dims);
        }
    }
    for i in 0..4 {
        if let Some(info) = file.get_tensor_info(&format!("blk.{}.ssm_out.weight", i)) {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            println!("blk.{}.ssm_out.weight: dims={:?}", i, dims);
        }
    }
}
