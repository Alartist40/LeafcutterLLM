//! gemv_bench — micro-benchmark comparing the fused Q4_K/Q6_K GEMV vs the
//! Q8_K-activation integer-dot GEMV for the streaming (m == 1) case.
//!
//! Usage:
//!     cargo run --release --bin gemv_bench -- q4|q6 [k] [n]

use leafcutter::kernels::q4_k::Matrix as Q4KMatrix;
use leafcutter::kernels::q4_k::Block as Q4KBlock;
use leafcutter::kernels::q6_k::Matrix as Q6KMatrix;
use leafcutter::kernels::q6_k::Block as Q6KBlock;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = args.get(1).map(|s| s.as_str()).unwrap_or("q4");
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(12288);

    let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.17).cos() * 1.2).collect();
    let iters = 200usize;

    match kind {
        "q4" => {
            let b = make_q4(n, k);
            let mut c = vec![0.0f32; n];
            // warmup
            leafcutter::kernels::q4_k_gemm::q4_k_matmul_transposed_b(&a, &b, &mut c, 1, k, n);
            let t0 = Instant::now();
            for _ in 0..iters {
                leafcutter::kernels::q4_k_gemm::q4_k_matmul_transposed_b(&a, &b, &mut c, 1, k, n);
            }
            report("Q4_K fused f32", t0, iters);
            let t0 = Instant::now();
            for _ in 0..iters {
                leafcutter::kernels::q8_k_gemm::q4_k_matmul_transposed_b_q8(&a, &b, &mut c, 1, k, n);
            }
            report("Q4_K x Q8_K int", t0, iters);
        }
        "q6" => {
            let b = make_q6(n, k);
            let mut c = vec![0.0f32; n];
            leafcutter::kernels::q6_k_gemm::q6_k_matmul_transposed_b(&a, &b, &mut c, 1, k, n);
            let t0 = Instant::now();
            for _ in 0..iters {
                leafcutter::kernels::q6_k_gemm::q6_k_matmul_transposed_b(&a, &b, &mut c, 1, k, n);
            }
            report("Q6_K fused f32", t0, iters);
            let t0 = Instant::now();
            for _ in 0..iters {
                leafcutter::kernels::q8_k_gemm::q6_k_matmul_transposed_b_q8(&a, &b, &mut c, 1, k, n);
            }
            report("Q6_K x Q8_K int", t0, iters);
        }
        _ => {
            eprintln!("usage: gemv_bench q4|q6 [k] [n]");
            std::process::exit(2);
        }
    }
}

fn report(name: &str, t0: Instant, iters: usize) {
    let total = t0.elapsed().as_secs_f64();
    println!("{:<20} {:>9.3} ms/iter", name, total / iters as f64 * 1000.0);
}

fn make_q4(rows: usize, cols: usize) -> Q4KMatrix {
    let bpr = cols / 256;
    let mut blocks = Vec::with_capacity(rows * bpr);
    for row in 0..rows {
        for _b in 0..bpr {
            let d = 0.03f32 * ((row as f32) + 1.0) / (rows as f32);
            let dmin = 0.01f32;
            let mut scales = [0u8; 12];
            for s in 0..12 {
                scales[s] = (((s * 3 + row) as i32) % 64) as u8;
            }
            let mut qs = [0u8; 128];
            for s in 0..128 {
                qs[s] = ((s * 5 + row * 7 + s / 4 * 11) % 256) as u8;
            }
            blocks.push(Q4KBlock { d, dmin, scales, qs });
        }
    }
    Q4KMatrix { rows, cols, blocks }
}

fn make_q6(rows: usize, cols: usize) -> Q6KMatrix {
    let bpr = cols / 256;
    let mut blocks = Vec::with_capacity(rows * bpr);
    for row in 0..rows {
        for _b in 0..bpr {
            let d = 0.02f32 * ((row as f32) + 1.0) / (rows as f32);
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            for s in 0..128 {
                ql[s] = ((s * 3 + row + s / 4 * 13) % 256) as u8;
            }
            for s in 0..64 {
                qh[s] = ((s * 7 + row * 5) % 256) as u8;
            }
            let mut scales = [0u8; 16];
            for s in 0..16 {
                scales[s] = (((s * 5 + row) as i32) % 61 - 30) as u8;
            }
            blocks.push(Q6KBlock { d, ql, qh, scales });
        }
    }
    Q6KMatrix { rows, cols, blocks }
}
