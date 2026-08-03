//! Debug: verify weight tensor statistics match expected values.
use leafcutter::safetensors_loader::Shards;
use std::path::Path;

fn mean_abs(data: &[f32]) -> f32 {
    data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32
}

fn softplus(x: f32) -> f32 {
    if x > 30.0 { x } else { (1.0 + x.exp()).ln() }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn main() {
    let dir = Path::new("/home/xander/Downloads/models/ornith safetensor");
    let shards = Shards::open_dir(dir).expect("open");

    // Check dt_bias and A_log for layer 0
    let dt_bias = shards.read_tensor_f32("model.language_model.layers.0.linear_attn.dt_bias").unwrap();
    let a_log = shards.read_tensor_f32("model.language_model.layers.0.linear_attn.A_log").unwrap();
    
    println!("dt_bias (32 values):");
    for i in 0..32 {
        print!("  {:.4}", dt_bias[i]);
        if (i+1) % 8 == 0 { println!(); }
    }
    println!("\ndt_bias min={:.4} max={:.4} mean_abs={:.4}", 
        dt_bias.iter().cloned().fold(f32::MAX, f32::min),
        dt_bias.iter().cloned().fold(f32::MIN, f32::max),
        mean_abs(&dt_bias));

    println!("\nA_log (32 values):");
    for i in 0..32 {
        print!("  {:.4}", a_log[i]);
        if (i+1) % 8 == 0 { println!(); }
    }
    println!("");
    println!("A_log min={:.4} max={:.4} mean_abs={:.4}",
        a_log.iter().cloned().fold(f32::MAX, f32::min),
        a_log.iter().cloned().fold(f32::MIN, f32::max),
        mean_abs(&a_log));

    // Compute decay for a typical input
    println!("\nDecay for each head (assuming alpha=0):");
    for i in 0..32 {
        let a = (-a_log[i].exp()).exp();
        let dt = softplus(dt_bias[i]);
        let decay = (dt * -a_log[i].exp()).exp();
        println!("  head {i}: A={:.6} dt={:.4} decay={:.6}", -a_log[i].exp(), dt, decay);
    }

    // Check in_proj_b (beta) and in_proj_z (z-gate) 
    let b_w = shards.read_tensor_f32("model.language_model.layers.0.linear_attn.in_proj_b.weight").unwrap();
    let z_w = shards.read_tensor_f32("model.language_model.layers.0.linear_attn.in_proj_z.weight").unwrap();
    println!("\nin_proj_b.weight mean_abs={:.6}", mean_abs(&b_w));
    println!("in_proj_z.weight mean_abs={:.6}", mean_abs(&z_w));

    // Check the proj_qkv weight distribution (first few rows)
    let qkv_w = shards.read_tensor_f32("model.language_model.layers.0.linear_attn.in_proj_qkv.weight").unwrap();
    let h = 4096;
    // Section Q (rows 0..2048), K (2048..4096), V (4096..8192)
    let q_rows = &qkv_w[0 * h .. 2048 * h];
    let k_rows = &qkv_w[2048 * h .. 4096 * h];
    let v_rows = &qkv_w[4096 * h .. 8192 * h];
    println!("\nin_proj_qkv.weight section mean_abs:");
    println!("  Q rows (0..2048): {:.6}", mean_abs(q_rows));
    println!("  K rows (2048..4096): {:.6}", mean_abs(k_rows));
    println!("  V rows (4096..8192): {:.6}", mean_abs(v_rows));
}
