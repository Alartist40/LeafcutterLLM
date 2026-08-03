use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = "/home/xander/Downloads/models/ornith-1.0-9b-Q6_K.gguf";
    let model = GGUFModel::load(path).expect("load");
    let layer = model.load_layer(0).expect("layer");
    let conv_w = layer.get("ssm_conv1d.weight").expect("conv1d");
    println!("conv1d shape: {:?}", conv_w.shape);
    println!("conv1d data[0..16] (first kernel, first 16 channels):");
    for i in 0..16 {
        println!("  data[{}] = {:+.6}", i, conv_w.data[i]);
    }
    println!("\nVerify GGUF raw read directly:");
    let raw = model.file.get_tensor_raw("blk.0.ssm_conv1d.weight").expect("raw");
    let mut deq = vec![0.0f32; raw.len() / 4];
    for (i, chunk) in raw.chunks_exact(4).enumerate() {
        deq[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    println!("raw deq (shape treated as [4, 8192]) data[0..16] (kernel 0, ch 0..16):");
    for i in 0..16 {
        println!("  deq[{}] = {:+.6}", i, deq[i]);
    }
    // Also show raw[0..32] (kernel 0 cols 0..8) to compare with engine
    println!("\nraw[0..32] (linear — this is what transpose reads as columns):");
    for i in 0..32 {
        print!("{:.4} ", deq[i]);
        if (i+1) % 8 == 0 { println!(); }
    }
}
