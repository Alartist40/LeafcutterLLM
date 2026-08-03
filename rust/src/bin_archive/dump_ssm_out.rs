use leafcutter::model::gguf::GGUFile;
use leafcutter::kernels::q5_k::Matrix as Q5KMatrix;
use leafcutter::kernels::q5_k::blocks_from_bytes;

fn main() {
    let path = std::env::args().nth(1).expect("Usage");
    let file = GGUFile::open(&path).expect("open");
    let info = file.get_tensor_info("blk.0.ssm_out.weight").expect("find");
    let raw = file.get_tensor_raw("blk.0.ssm_out.weight").expect("read");
    let inner = info.dimensions[0] as usize;
    let outer = info.dimensions[1] as usize;

    let q5mat = Q5KMatrix {
        rows: outer,
        cols: inner,
        blocks: blocks_from_bytes(raw),
    };

    let dequant = q5mat.dequantize();
    println!("ssm_out dequantized: len={} shape=[{}, {}]", dequant.len(), outer, inner);
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(dequant.as_ptr() as *const u8, dequant.len() * 4)
    };
    std::fs::write("ssm_out_direct_dequant.bin", bytes).expect("write");
}
