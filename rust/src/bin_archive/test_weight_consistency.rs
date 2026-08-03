//! Check if dumped f32 weight matches direct GGUF dequantization

use leafcutter::model::gguf::GGUFile;
use leafcutter::kernels::q4_0::{Matrix as Q4Matrix, blocks_from_bytes};

fn main() {
    let file = GGUFile::open("../models/Qwen3.5-0.8B-Q4_0.gguf").expect("open gguf");
    let raw = file.get_tensor_raw("blk.0.attn_qkv.weight").expect("read raw");

    // Direct dequantization: shape_data = [6144, 1024]
    let q4mat = Q4Matrix {
        rows: 6144,
        cols: 1024,
        blocks: blocks_from_bytes(raw),
    };
    let direct_deq = q4mat.dequantize(); // [6144, 1024] row-major

    // Load dumped weight: should be [1024, 6144] row-major (transposed)
    let dumped_data = std::fs::read("blk_0_attn_qkv_weight.bin").expect("read dump");
    let dumped: Vec<f32> = dumped_data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    println!("Direct dequantized len: {}, dumped len: {}", direct_deq.len(), dumped.len());

    // Compare: dumped[k, o] should equal direct_deq[o, k]
    let mut diff = 0.0f32;
    let mut max_diff = 0.0f32;
    for o in 0..6144 {
        for k in 0..1024 {
            let direct_val = direct_deq[o * 1024 + k];
            let dumped_val = dumped[k * 6144 + o];
            let d = (direct_val - dumped_val).abs();
            diff += d;
            max_diff = max_diff.max(d);
        }
    }
    let mae = diff / (6144.0 * 1024.0);
    println!("MAE between direct dequant and dumped: {:.10}", mae);
    println!("Max diff: {:.10}", max_diff);

    // Also verify the other way: if we transpose dumped back to [6144, 1024]
    let mut dumped_transposed = vec![0.0f32; 6144 * 1024];
    for o in 0..6144 {
        for k in 0..1024 {
            dumped_transposed[o * 1024 + k] = dumped[k * 6144 + o];
        }
    }

    let mut diff2 = 0.0f32;
    let mut max_diff2 = 0.0f32;
    for i in 0..direct_deq.len() {
        let d = (direct_deq[i] - dumped_transposed[i]).abs();
        diff2 += d;
        max_diff2 = max_diff2.max(d);
    }
    let mae2 = diff2 / direct_deq.len() as f32;
    println!("MAE after transposing dumped: {:.10}", mae2);
    println!("Max diff after transpose: {:.10}", max_diff2);
}
