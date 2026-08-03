use leafcutter::model::gguf::GGUFile;
use leafcutter::kernels::q4_0::Matrix as Q4Matrix;
use leafcutter::kernels::q4_0::blocks_from_bytes;

fn main() {
    let path = "../models/Qwen3.5-0.8B-Q4_0.gguf";
    let file = GGUFile::open(path).expect("open gguf");
    
    let info = file.get_tensor_info("blk.0.attn_qkv.weight").expect("find");
    let raw = file.get_tensor_raw("blk.0.attn_qkv.weight").expect("read");
    let inner = info.dimensions[0] as usize;
    let outer = info.dimensions[1] as usize;
    
    let q4mat = Q4Matrix {
        rows: outer,
        cols: inner,
        blocks: blocks_from_bytes(raw),
    };
    let dequant1 = q4mat.dequantize();
    
    let mut dequant2 = vec![0.0f32; outer * inner];
    leafcutter::kernels::dequantize_q4_0(raw, &mut dequant2);
    
    let mae = dequant1.iter().zip(dequant2.iter()).map(|(a,b)| (a-b).abs()).sum::<f32>() / dequant1.len() as f32;
    let cos_sim = {
        let mut dot = 0.0f32; let mut a_sq = 0.0f32; let mut b_sq = 0.0f32;
        for i in 0..dequant1.len() { dot += dequant1[i] * dequant2[i]; a_sq += dequant1[i] * dequant1[i]; b_sq += dequant2[i] * dequant2[i]; }
        dot / (a_sq.sqrt() * b_sq.sqrt() + 1e-10)
    };
    println!("Q4Matrix::dequantize vs kernels::dequantize_q4_0:");
    println!("  MAE: {:.6}", mae);
    println!("  CosSim: {:.6}", cos_sim);
}
