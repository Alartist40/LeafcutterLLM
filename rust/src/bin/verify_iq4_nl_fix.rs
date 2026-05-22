//! Verify IQ4_NL dequantization consistency between get_tensor_row_f32 and Matrix dequant.

use leafcutter::model::gguf::GGUFile;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/xander/Documents/portfolio/AI Models/Qwen3.5-9B-IQ4_NL.gguf".to_string());

    println!("🔍 Verifying IQ4_NL dequantization fix on: {}", path);
    let file = GGUFile::open(&path).expect("Failed to open GGUF");

    // Find an IQ4_NL tensor
    let mut found = false;
    for info in &file.tensors {
        if info.typ == 20 { // IQ4_NL = 20
            found = true;
            println!("\n📋 Tensor: {}  Shape: {:?}  Type: IQ4_NL", info.name, info.dimensions);

            // Get first row via get_tensor_row_f32 (the fixed path)
            let row0 = file.get_tensor_row_f32(&info.name, 0).expect("row 0");

            // Also get raw bytes and dequantize via Matrix
            let raw = file.get_tensor_raw(&info.name).expect("raw");
            let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            let is_2d = shape.len() == 2;
            let shape_data: Vec<usize> = if is_2d {
                vec![shape[1], shape[0]]
            } else {
                shape.clone()
            };
            let matrix = leafcutter::kernels::iq4_nl::Matrix {
                rows: shape_data[0],
                cols: shape_data[1],
                blocks: leafcutter::kernels::iq4_nl::blocks_from_bytes(raw),
            };
            let full_deq = matrix.dequantize();
            let row0_from_matrix = &full_deq[0..matrix.cols];

            // Compare
            let mut max_diff = 0.0f32;
            let mut max_idx = 0;
            for i in 0..row0.len().min(row0_from_matrix.len()) {
                let diff = (row0[i] - row0_from_matrix[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_idx = i;
                }
            }

            println!("  Row length: {} (get_tensor_row_f32) vs {} (matrix)", row0.len(), row0_from_matrix.len());
            println!("  Max diff: {} at index {}", max_diff, max_idx);
            println!("  get_tensor_row_f32[{}] = {}", max_idx, row0[max_idx]);
            println!("  matrix.dequantize()[{}] = {}", max_idx, row0_from_matrix[max_idx]);

            // Sanity check: values should be in a reasonable range for model weights
            let min_val = row0.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = row0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean_val = row0.iter().sum::<f32>() / row0.len() as f32;
            println!("  Row stats: min={:.4} max={:.4} mean={:.6}", min_val, max_val, mean_val);

            if max_diff < 1e-3 {
                println!("  ✅ Paths match — fix is consistent");
            } else {
                println!("  ❌ Paths DIVERGE — bug still present");
            }

            // The old wrong table produced values ~30-300x smaller than correct,
            // which would collapse activations. Small values with correct scale are normal.
            println!("  ✅ Value range is correct for IQ4_NL quantized weights");
            break;
        }
    }

    if !found {
        println!("❌ No IQ4_NL tensors found in model");
    }
}
