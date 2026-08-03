//! Test: dequantize Q6_K then matmul vs q6_k_matmul_transposed_b.
//! These should produce identical results.

use leafcutter::model::gguf::GGUFile;
use leafcutter::model::loader::GGUFModel;

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                s += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn main() {
    let path = "/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf";
    let model = GGUFModel::load(path).expect("load");
    let file = &model.file;

    let info = file.get_tensor_info("blk.0.attn_qkv.weight").unwrap();
    println!("qkv info: typ={} dims={:?}", info.typ, info.dimensions);

    // Get token embedding row as a small input
    let embed = file.get_tensor_row_f32("token_embd.weight", 760).unwrap();
    println!("embed len={} (first 4): {:?}", embed.len(), &embed[..4]);

    // 1. Dequantize the QKV weight fully and matmul
    let count: usize = info.dimensions.iter().map(|&d| d as usize).product();
    let raw = file.get_tensor_raw("blk.0.attn_qkv.weight").unwrap();
    let mut dequant_w = vec![0.0f32; count];
    leafcutter::kernels::dequantize_q6_k(raw, &mut dequant_w);

    let qkv_ref = matmul(&embed, &dequant_w, 1, embed.len(), 8192);
    let qkv_ref_max = qkv_ref.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let qkv_ref_min = qkv_ref.iter().cloned().fold(f32::INFINITY, f32::min);
    let qkv_ref_abs: f32 = qkv_ref.iter().map(|v| v.abs()).sum::<f32>() / qkv_ref.len() as f32;
    println!("\n[dequant+matmul] qkv: min={:.5} max={:.5} abs_mean={:.5}",
             qkv_ref_min, qkv_ref_max, qkv_ref_abs);

    // 2. Load via Tensor struct (uses engine's Q6KMatrix path)
    // GGUF stores weights as [in_dim, out_dim] = [k, n]. For the matmul
    // C = A @ B^T, we need Q6KMatrix with rows = n (output), cols = k (input).
    let mat = leafcutter::kernels::q6_k::Matrix {
        rows: 8192,
        cols: 4096,
        blocks: leafcutter::kernels::q6_k::blocks_from_bytes(raw),
    };

    // Compare: Q6KMatrix::dequantize() vs full dequantize_q6_k
    let mat_dequant = mat.dequantize();
    println!("\n[Q6KMatrix.dequantize]");
    println!("  min={:.5} max={:.5} abs_mean={:.5}",
             mat_dequant.iter().cloned().fold(f32::INFINITY, f32::min),
             mat_dequant.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
             mat_dequant.iter().map(|v| v.abs()).sum::<f32>() / mat_dequant.len() as f32);
    println!("  first 16: {:?}", &mat_dequant[..16]);
    let max_dd = dequant_w.iter().zip(mat_dequant.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("  diff vs dequantize_q6_k: max = {:.5}", max_dd);
    // Now do the matmul through Q6KMatrix
    let mut qkv_e = vec![0.0f32; 8192];
    leafcutter::kernels::q6_k_gemm::q6_k_matmul_transposed_b(
        &embed, &mat, &mut qkv_e, 1, 4096, 8192,
    );
    let qkv_e_max = qkv_e.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let qkv_e_min = qkv_e.iter().cloned().fold(f32::INFINITY, f32::min);
    let qkv_e_abs: f32 = qkv_e.iter().map(|v| v.abs()).sum::<f32>() / qkv_e.len() as f32;
    println!(
        "\n[engine q6_k_matmul] qkv: min={:.5} max={:.5} abs_mean={:.5}",
        qkv_e_min, qkv_e_max, qkv_e_abs
    );

    // Element-wise diff
    let max_diff = qkv_ref
        .iter()
        .zip(qkv_e.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let abs_diff_sum: f32 = qkv_ref
        .iter()
        .zip(qkv_e.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>();
    println!(
        "\nDIFF: max_diff = {:.5}, sum_abs_diff = {:.5}",
        max_diff, abs_diff_sum
    );

    println!("\n[qkv_ref first 16]");
    for i in 0..16 {
        println!("  [{:2}] = {:>+10.5}", i, qkv_ref[i]);
    }
    println!("\n[qkv_e first 16]");
    for i in 0..16 {
        println!("  [{:2}] = {:>+10.5}", i, qkv_e[i]);
    }
}
