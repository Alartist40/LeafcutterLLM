// Dump a dequantized GGUF tensor to a binary file
use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = std::env::args().nth(1).expect("gguf path");
    let gguf_name = std::env::args().nth(2).expect("gguf tensor name (e.g. blk.0.attn_qkv.weight)");
    let out_path = std::env::args().nth(3).expect("output bin path");
    let model = GGUFModel::load(&path).unwrap();
    let info = model.file.get_tensor_info(&gguf_name).unwrap();
    let raw = model.file.get_tensor_raw(&gguf_name).unwrap();
    let cols = info.dimensions[0] as usize;
    let rows = info.dimensions[1] as usize;
    let count = cols * rows;
    let mut out = vec![0.0f32; count];
    let qtype = leafcutter::model::quant::QuantType::from_u32(info.typ).unwrap();
    match qtype {
        leafcutter::model::quant::QuantType::Q4_K => leafcutter::kernels::dequantize_q4_k(raw, &mut out),
        leafcutter::model::quant::QuantType::Q6_K => leafcutter::kernels::dequantize_q6_k(raw, &mut out),
        leafcutter::model::quant::QuantType::Q8_0 => leafcutter::kernels::dequantize_q8_0(raw, &mut out),
        leafcutter::model::quant::QuantType::F32 => {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(raw.as_ptr() as *const u8, raw.len())
            };
            let f32_slice: &[f32] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, count)
            };
            out.copy_from_slice(f32_slice);
        }
        _ => panic!("not supported: {:?}", qtype),
    }
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 4)
    };
    std::fs::write(&out_path, bytes).unwrap();
    eprintln!("Wrote {} f32 ({}x{}) to {}", count, cols, rows, out_path);
    eprintln!("First 8: {:?}", &out[..8]);
    eprintln!("Last 8: {:?}", &out[count-8..]);
    eprintln!("Min={:.6}, Max={:.6}", out.iter().cloned().fold(f32::INFINITY, f32::min), out.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
}
