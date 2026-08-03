//! Diagnostic: Compare native Q4_0 matmul against dequantized reference
//! using real model weights.
//!
//! Usage: cargo run --release --bin test_q4_0_real_weight -- <model.gguf>

use leafcutter::model::gguf::GGUFile;
use leafcutter::kernels::q4_0::Matrix as Q4Matrix;
use leafcutter::kernels::q4_0::blocks_from_bytes;
use leafcutter::kernels::int8_gemm::q4_0_matmul_transposed_b;
use std::env;

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
    let model_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "../models/Qwen3.5-0.8B-Q4_0.gguf".to_string());

    let file = GGUFile::open(&model_path).expect("open gguf");

    let info = file.get_tensor_info("blk.0.attn_qkv.weight")
        .expect("find attn_qkv.weight");
    let raw_bytes = file.get_tensor_raw("blk.0.attn_qkv.weight")
        .expect("read raw bytes");

    // GGUF dimensions: [inner, outer]
    let inner = info.dimensions[0] as usize; // 1024
    let outer = info.dimensions[1] as usize; // 6144

    println!("GGUF dimensions (metadata): {:?}", info.dimensions);
    println!("inner (K) = {}, outer (N) = {}", inner, outer);
    println!("Raw bytes length: {}", raw_bytes.len());
    println!("Expected bytes: {} blocks * 18 = {}",
        outer * inner / 32,
        (outer * inner / 32) * 18);

    assert_eq!(raw_bytes.len(), (outer * inner / 32) * 18,
        "Byte count mismatch");

    // ============================================================
    // 1. Build Q4Matrix with CORRECT orientation for our kernel
    // ============================================================
    // Our loader uses shape_data = [outer, inner] = [6144, 1024]
    // So Q4Matrix has rows = outer = 6144, cols = inner = 1024
    let q4mat = Q4Matrix {
        rows: outer,
        cols: inner,
        blocks: blocks_from_bytes(raw_bytes),
    };
    println!("\nQ4Matrix: rows={}, cols={}, blocks_per_row={}",
        q4mat.rows, q4mat.cols, q4mat.blocks_per_row());

    // ============================================================
    // 2. Reference: Dequantize to f32, then matmul
    // ============================================================
    // dequantize() produces [rows, cols] = [6144, 1024] row-major
    let weight_f32 = q4mat.dequantize();
    println!("Dequantized weight shape: [{}, {}] (row-major)", outer, inner);

    // For PyTorch-style linear: y = x @ W^T
    // W (PyTorch) = [out, in] = [6144, 1024]
    // W^T = [1024, 6144]
    // Our dequantized buffer is [6144, 1024] row-major.
    // Element [r, c] is at index r * 1024 + c.
    // To get W^T [in, out] = [1024, 6144], we need:
    //   W^T[i, o] = W[o, i] = weight_f32[o * 1024 + i]
    //
    // Reference matmul: x[1, 1024] @ W^T[1024, 6144] -> [1, 6144]
    //   out[o] = sum_i x[i] * W^T[i, o]
    //          = sum_i x[i] * W[o, i]
    //          = sum_i x[i] * weight_f32[o * 1024 + i]

    let seq_len = 1;
    let input_f32: Vec<f32> = (0..inner).map(|i| ((i % 7) as f32 - 3.0) / 10.0).collect();

    let mut ref_out = vec![0.0f32; seq_len * outer];
    for i in 0..seq_len {
        for o in 0..outer {
            let mut sum = 0.0;
            for k in 0..inner {
                // W[o, k] at index o * inner + k
                sum += input_f32[i * inner + k] * weight_f32[o * inner + k];
            }
            ref_out[i * outer + o] = sum;
        }
    }

    // ============================================================
    // 3. Fast path: Native Q4_0 matmul
    // ============================================================
    // q4_0_matmul_transposed_b computes C = A @ B^T
    // A is [m, k], B is [n, k] in memory, C is [m, n]
    // Here: m=1, k=1024 (inner), n=6144 (outer)
    // B (Q4Matrix) has rows=n=6144, cols=k=1024
    let mut fast_out = vec![0.0f32; seq_len * outer];
    q4_0_matmul_transposed_b(&input_f32, &q4mat, &mut fast_out, seq_len, inner, outer);

    // ============================================================
    // 4. Compare
    // ============================================================
    let mut diff_sum = 0.0f32;
    let mut diff_max = 0.0f32;
    for i in 0..ref_out.len() {
        let d = (ref_out[i] - fast_out[i]).abs();
        diff_sum += d;
        diff_max = diff_max.max(d);
    }
    let mae = diff_sum / ref_out.len() as f32;
    let cos_sim = cosine_similarity(&ref_out, &fast_out);

    println!("\n=== Q4_0 Matmul vs Reference (orientation: rows=outer, cols=inner) ===");
    println!("MAE:    {:.6}", mae);
    println!("MaxE:   {:.6}", diff_max);
    println!("CosSim: {:.6}", cos_sim);

    // ============================================================
    // 5. Try SWAPPED orientation
    // ============================================================
    println!("\n=== Trying swapped orientation ===");
    let q4mat_swapped = Q4Matrix {
        rows: inner,
        cols: outer,
        blocks: blocks_from_bytes(raw_bytes),
    };

    // If we swap rows/cols, the kernel asserts will fail unless we also swap k and n:
    // b.cols must == k, b.rows must == n
    // If b.rows = inner = 1024 and b.cols = outer = 6144,
    // then we'd need k = 6144, n = 1024... but that changes the output dimensions.
    // So swapping orientation like this is dimensionally inconsistent.
    // Instead, let's just verify the first orientation was the right one.

    // Actually, let's try a different interpretation:
    // Maybe the blocks are laid out for [inner, outer] = [1024, 6144]
    // i.e. 1024 rows of 6144 columns each.
    // blocks_per_row = 6144 / 32 = 192 blocks per row
    // total rows = 1024
    let q4mat_alt = Q4Matrix {
        rows: inner,
        cols: outer,
        blocks: blocks_from_bytes(raw_bytes),
    };
    println!("Alt Q4Matrix: rows={}, cols={}, blocks_per_row={}",
        q4mat_alt.rows, q4mat_alt.cols, q4mat_alt.blocks_per_row());

    // This would mean B is [1024, 6144] in memory.
    // For q4_0_matmul_transposed_b, B must be [n, k].
    // If we want output [1, 6144], n=6144, k=1024.
    // But B.cols = 6144 != k=1024, so this will panic.

    // Let's instead compute a reference where we treat the dequantized buffer
    // as [inner, outer] = [1024, 6144] row-major and see if that matches fast_out.
    let weight_f32_alt_layout = q4mat_alt.dequantize();
    // weight_f32_alt_layout[i, o] is at i * outer + o
    // Reference: x[1, 1024] @ W[1024, 6144] -> [1, 6144]
    //   out[o] = sum_i x[i] * W[i, o]
    let mut ref_out_alt = vec![0.0f32; seq_len * outer];
    for i in 0..seq_len {
        for o in 0..outer {
            let mut sum = 0.0;
            for k in 0..inner {
                sum += input_f32[i * inner + k] * weight_f32_alt_layout[k * outer + o];
            }
            ref_out_alt[i * outer + o] = sum;
        }
    }

    let mae_alt = ref_out_alt.iter().zip(fast_out.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / ref_out_alt.len() as f32;
    let cos_sim_alt = cosine_similarity(&ref_out_alt, &fast_out);

    println!("Alt MAE:    {:.6}", mae_alt);
    println!("Alt CosSim: {:.6}", cos_sim_alt);

    // ============================================================
    // 6. What if fast_out should be compared to x @ W (not W^T)?
    // ============================================================
    // i.e. what if the kernel computes C = A @ B (not A @ B^T)?
    // Then for A[1, 1024] @ B[1024, 6144], B must be [1024, 6144].
    // Our q4mat_alt IS [1024, 6144] in memory.
    // But q4_0_matmul_transposed_b expects B in [n, k] format...
    // Let's use the non-transposed scalar kernel directly.
    println!("\n=== Using q4_0_matmul (non-transposed) with alt layout ===");
    let mut fast_out_nt = vec![0.0f32; seq_len * outer];
    leafcutter::kernels::int8_gemm::q4_0_matmul(
        &input_f32, &q4mat_alt, &mut fast_out_nt, seq_len, inner, outer);

    let mae_nt = ref_out_alt.iter().zip(fast_out_nt.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / ref_out_alt.len() as f32;
    let cos_sim_nt = cosine_similarity(&ref_out_alt, &fast_out_nt);

    println!("Non-transposed MAE:    {:.6}", mae_nt);
    println!("Non-transposed CosSim: {:.6}", cos_sim_nt);

    // ============================================================
    // 7. Summary
    // ============================================================
    println!("\n=== SUMMARY ===");
    println!("transposed_b + [outer,inner] layout:  CosSim = {:.6}", cos_sim);
    println!("transposed_b + [inner,outer] layout:  CosSim = {:.6}", cos_sim_alt);
    println!("q4_0_matmul + [inner,outer] layout:   CosSim = {:.6}", cos_sim_nt);
}
