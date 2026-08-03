//! Diagnostic: Load dumped pre_norm + weight, do qkv_proj matmul,
//! compare dumped native output vs reference dequantized matmul.

use leafcutter::model::tensor::Tensor;
use leafcutter::kernels::q4_0::Matrix as Q4Matrix;
use leafcutter::kernels::q4_0::blocks_from_bytes;
use leafcutter::kernels::int8_gemm::q4_0_matmul_transposed_b;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut a_sq = 0.0f32;
    let mut b_sq = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        a_sq += a[i] * a[i];
        b_sq += b[i] * b[i];
    }
    dot / (a_sq.sqrt() * b_sq.sqrt() + 1e-10)
}

fn main() {
    // 1. Load dumped post_norm [4, 1024]
    let pre_norm_data = std::fs::read("native_l0_input_post_norm_4096.bin").expect("read post_norm");
    let pre_norm_f32: Vec<f32> = pre_norm_data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let pre_norm = Tensor::from_vec(pre_norm_f32, vec![4, 1024]);
    println!("pre_norm shape: {:?}, abs_mean={:.6}", pre_norm.shape,
        pre_norm.data.iter().map(|&v| v.abs()).sum::<f32>() / pre_norm.data.len() as f32);

    // 2. Load dumped weight [1024, 6144] f32 (from fixed dump)
    let weight_data = std::fs::read("blk_0_attn_qkv_weight.bin").expect("read weight");
    let weight_f32: Vec<f32> = weight_data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let weight_tensor = Tensor::from_vec(weight_f32, vec![1024, 6144]);
    println!("weight shape: {:?}, abs_mean={:.6}", weight_tensor.shape,
        weight_tensor.data.iter().map(|&v| v.abs()).sum::<f32>() / weight_tensor.data.len() as f32);

    // 3. Reference matmul: pre_norm @ weight
    let ref_qkv = pre_norm.matmul(&weight_tensor);
    println!("ref_qkv shape: {:?}, abs_mean={:.6}", ref_qkv.shape,
        ref_qkv.data.iter().map(|&v| v.abs()).sum::<f32>() / ref_qkv.data.len() as f32);

    // 4. Load native dumped qkv_proj
    let native_qkv_data = std::fs::read("native_l0_qkv_proj_s4.bin").expect("read native qkv");
    let native_qkv_f32: Vec<f32> = native_qkv_data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let native_qkv = Tensor::from_vec(native_qkv_f32, vec![4, 6144]);
    println!("native_qkv shape: {:?}, abs_mean={:.6}", native_qkv.shape,
        native_qkv.data.iter().map(|&v| v.abs()).sum::<f32>() / native_qkv.data.len() as f32);

    // 5. Compare ref vs native
    let cos_sim = cosine_similarity(&ref_qkv.data, &native_qkv.data);
    let mae = ref_qkv.data.iter().zip(native_qkv.data.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / ref_qkv.data.len() as f32;
    println!("\n=== Reference f32 matmul vs Native dumped qkv_proj ===");
    println!("MAE:    {:.6}", mae);
    println!("CosSim: {:.6}", cos_sim);

    // 6. Now test with Q4_0 matmul using the same raw bytes
    // Re-quantize the f32 weight to Q4_0, then matmul
    let q4_bytes = leafcutter::kernels::q4_0::quantize_f32_to_q4_0(&weight_tensor.data);
    let q4mat = Q4Matrix {
        rows: 6144,
        cols: 1024,
        blocks: blocks_from_bytes(&q4_bytes),
    };
    let mut fast_qkv = vec![0.0f32; 4 * 6144];
    q4_0_matmul_transposed_b(&pre_norm.data, &q4mat, &mut fast_qkv, 4, 1024, 6144);

    let cos_sim_fast = cosine_similarity(&ref_qkv.data, &fast_qkv);
    let mae_fast = ref_qkv.data.iter().zip(fast_qkv.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / ref_qkv.data.len() as f32;
    println!("\n=== Reference f32 matmul vs Re-quantized Q4_0 matmul ===");
    println!("MAE:    {:.6}", mae_fast);
    println!("CosSim: {:.6}", cos_sim_fast);
}
