use leafcutter::model::gguf::GGUFile;
use leafcutter::kernels::q4_0::Matrix as Q4Matrix;
use leafcutter::kernels::q4_0::blocks_from_bytes;

fn main() {
    let path = std::env::args().nth(1).expect("Usage");
    let file = GGUFile::open(&path).expect("open");
    let info = file.get_tensor_info("blk.0.attn_qkv.weight").expect("find");
    let raw = file.get_tensor_raw("blk.0.attn_qkv.weight").expect("read");
    let inner = info.dimensions[0] as usize;
    let outer = info.dimensions[1] as usize;

    let q4mat = Q4Matrix {
        rows: outer,
        cols: inner,
        blocks: blocks_from_bytes(raw),
    };

    let dequant = q4mat.dequantize();
    println!("Dequantized: len={} shape=[{}, {}] mean={:.6}", dequant.len(), outer, inner, dequant.iter().sum::<f32>() / dequant.len() as f32);

    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(dequant.as_ptr() as *const u8, dequant.len() * 4)
    };
    std::fs::write("q4_direct_dequant.bin", bytes).expect("write");
    println!("Wrote q4_direct_dequant.bin");
}
