use leafcutter::model::loader::GGUFModel;
use leafcutter::kernels;

fn main() {
    let path = "/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf";
    let model = GGUFModel::load(path).unwrap();
    let info = model.file.get_tensor_info("token_embd.weight").unwrap();
    eprintln!("token_embd.weight shape={:?} qtype={:?}", info.dimensions, info.typ);
    // qtype 12 = Q6_K typically
    let raw = model.file.get_tensor_raw("token_embd.weight").unwrap();
    let count: usize = info.dimensions.iter().map(|&d| d as usize).product();
    let mut out = vec![0.0f32; count];
    let qtype = leafcutter::model::quant::QuantType::from_u32(info.typ).unwrap();
    eprintln!("qtype = {:?}", qtype);
    match qtype {
        leafcutter::model::quant::QuantType::Q6_K => leafcutter::kernels::dequantize_q6_k(raw, &mut out),
        leafcutter::model::quant::QuantType::Q4_K => leafcutter::kernels::dequantize_q4_k(raw, &mut out),
        leafcutter::model::quant::QuantType::F32 => {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(raw.as_ptr() as *const u8, raw.len())
            };
            let f32_slice: &[f32] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, count)
            };
            out.copy_from_slice(f32_slice);
        }
        _ => panic!("unsupported"),
    }
    eprintln!("First 10 dequantized: {:?}", &out[..10]);
    eprintln!("Last 10 dequantized: {:?}", &out[count-10..]);
    eprintln!("Dequantized norm: {:.4}", out.iter().map(|x| x*x).sum::<f32>().sqrt());
}
