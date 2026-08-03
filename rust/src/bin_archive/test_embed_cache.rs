use leafcutter::inference::engine::Engine;

fn main() {
    let path = std::env::args().nth(1).expect("Usage: test_embed_cache <model.gguf>");
    let engine = Engine::load(&path).expect("Failed to load engine");

    let token_id = 9906usize;
    let hidden_size = engine.config.hidden_size;

    // Get embedding via on-demand mmap row lookup
    let mmap_row = engine.model.file.get_tensor_row_f32("token_embd.weight", token_id)
        .expect("Failed to read embedding row");

    println!("Mmap row len: {} (expected: {})", mmap_row.len(), hidden_size);
    assert_eq!(mmap_row.len(), hidden_size, "Row length mismatch");

    // Verify row is not all zeros or NaN
    let has_nan = mmap_row.iter().any(|&v| v.is_nan());
    let max_val = mmap_row.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!("Has NaN: {}, Max abs value: {}", has_nan, max_val);
    assert!(!has_nan, "Embedding row contains NaN");
    assert!(max_val > 0.0, "Embedding row is all zeros");

    // Compare against full-tensor dequantization for the same row
    let full_dequant = engine.model.file.get_tensor_raw("token_embd.weight")
        .and_then(|raw| engine.model.file.get_tensor_info("token_embd.weight")
            .map(|info| {
                let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                leafcutter::model::loader::GGUFModel::dequantize(raw, info.typ, shape)
            }))
        .expect("Failed to dequantize full embedding")
        .expect("Dequantize error");

    let outer = full_dequant.shape[1];
    let row_start = token_id * outer;
    let dequant_row = &full_dequant.data[row_start..row_start + outer];

    let mut max_diff = 0.0f32;
    for i in 0..mmap_row.len() {
        let diff = (mmap_row[i] - dequant_row[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("Max diff between mmap row and full dequant row: {}", max_diff);
    assert!(max_diff < 0.001, "Mmap row differs too much from full dequant");

    println!("✅ On-demand embedding lookup verified correct");
    println!("Mmap first 10: {:?}", &mmap_row[..10]);
    println!("Dequant first 10: {:?}", &dequant_row[..10]);
}
