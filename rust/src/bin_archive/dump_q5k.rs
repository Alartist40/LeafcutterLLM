use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let file = GGUFile::open(path).unwrap();
    
    let name = "blk.0.attn_q.weight";
    let info = file.get_tensor_info(name).unwrap();
    let cols = info.dimensions[0] as usize;
    let rows = info.dimensions.get(1).copied().unwrap_or(1) as usize;
    println!("{}: dims={:?}, type={}", name, info.dimensions, info.typ);
    
    let qtype = leafcutter::model::quant::QuantType::from_u32(info.typ).unwrap();
    let block_size = qtype.block_size();
    let block_bytes = qtype.block_bytes();
    let blocks_per_row = (cols + block_size - 1) / block_size;
    let row_bytes = blocks_per_row * block_bytes;
    
    let tensor_start = (file.data_offset + info.offset) as usize;
    println!("tensor_start={}, blocks_per_row={}, row_bytes={}", tensor_start, blocks_per_row, row_bytes);
    
    let row_f32 = file.get_tensor_row_f32(name, 0).unwrap();
    println!("Leafcutter first 16 values: {:?}", &row_f32[0..16]);
    println!("Leafcutter min={}, max={}", 
             row_f32.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             row_f32.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
}
