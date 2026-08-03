//! Debug test: verify Q5_K block-to-row mapping for shape_data vs shape_gguf

use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let file = GGUFile::open(path).unwrap();

    let name = "blk.0.ffn_gate.weight";
    let info = file.get_tensor_info(name).unwrap();
    let raw = file.get_tensor_raw(name).unwrap();

    let shape_gguf: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    let shape_data: Vec<usize> = vec![shape_gguf[1], shape_gguf[0]];

    println!("Tensor: {}", name);
    println!("  GGUF dims:  {:?}", shape_gguf);
    println!("  shape_data: {:?}", shape_data);
    println!("  Quant type: {:?}", info.typ);

    // Build Q5KMatrix with shape_data (current code)
    let q5_data = leafcutter::kernels::q5_k::Matrix {
        rows: shape_data[0],
        cols: shape_data[1],
        blocks: leafcutter::kernels::q5_k::blocks_from_bytes(raw),
    };
    let f32_data = q5_data.dequantize();
    println!("\nWith shape_data (rows={}, cols={}):", q5_data.rows, q5_data.cols);
    println!("  dequantized shape: {} elements", f32_data.len());
    println!("  first 8: {:?}", &f32_data[..8.min(f32_data.len())]);
    println!("  last 8:  {:?}", &f32_data[f32_data.len().saturating_sub(8)..]);

    // Build Q5KMatrix with shape_gguf (proposed patch)
    let q5_gguf = leafcutter::kernels::q5_k::Matrix {
        rows: shape_gguf[0],
        cols: shape_gguf[1],
        blocks: leafcutter::kernels::q5_k::blocks_from_bytes(raw),
    };
    let f32_gguf = q5_gguf.dequantize();
    println!("\nWith shape_gguf (rows={}, cols={}):", q5_gguf.rows, q5_gguf.cols);
    println!("  dequantized shape: {} elements", f32_gguf.len());
    println!("  first 8: {:?}", &f32_gguf[..8.min(f32_gguf.len())]);
    println!("  last 8:  {:?}", &f32_gguf[f32_gguf.len().saturating_sub(8)..]);

    // Compare
    let diff: f32 = f32_data.iter().zip(f32_gguf.iter()).map(|(a, b)| (a - b).abs()).sum();
    let max_diff = f32_data.iter().zip(f32_gguf.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("\n  diff sum={:.6} max={:.6}", diff, max_diff);

    if max_diff > 0.01 {
        println!("  ⚠️  DIFFERENT — shape_gguf does NOT match file layout");
    } else {
        println!("  ✅ SAME — shape_gguf matches file layout");
    }

    // Now test transpose: f32_data is [shape_data], transpose to [shape_gguf]
    let mut transposed = vec![0.0f32; f32_data.len()];
    let (m, n) = (shape_data[0], shape_data[1]);
    for i in 0..m {
        for j in 0..n {
            transposed[j * m + i] = f32_data[i * n + j];
        }
    }
    let diff2: f32 = transposed.iter().zip(f32_gguf.iter()).map(|(a, b)| (a - b).abs()).sum();
    let max_diff2 = transposed.iter().zip(f32_gguf.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("\n  transpose(f32_data) vs f32_gguf: max_diff={:.6}", max_diff2);
    if max_diff2 < 0.01 {
        println!("  ✅ f32_gguf == transpose(dequant(shape_data))");
    } else {
        println!("  ⚠️  f32_gguf != transpose(dequant(shape_data))");
    }
}
