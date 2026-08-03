use leafcutter::model::gguf::GGUFile;
use leafcutter::model::loader::GGUFModel;

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(&x, &w)| x * scale * w).collect()
}

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v * (1.0 / (1.0 + (-v).exp()))).collect()
}

fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&a, &b)| a + b).collect()
}

fn vec_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&a, &b)| a * b).collect()
}

fn apply_rope(q: &mut [f32], k: &mut [f32], num_heads: usize, num_kv_heads: usize, head_dim: usize, theta: f32) {
    for h in 0..num_heads {
        for d in 0..head_dim / 2 {
            let freq = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
            let angle = 0.0 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let base = h * head_dim;
            let x1 = q[base + d];
            let x2 = q[base + d + head_dim / 2];
            q[base + d] = x1 * cos_a - x2 * sin_a;
            q[base + d + head_dim / 2] = x1 * sin_a + x2 * cos_a;
        }
    }
    for h in 0..num_kv_heads {
        for d in 0..head_dim / 2 {
            let freq = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
            let angle = 0.0 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let base = h * head_dim;
            let x1 = k[base + d];
            let x2 = k[base + d + head_dim / 2];
            k[base + d] = x1 * cos_a - x2 * sin_a;
            k[base + d + head_dim / 2] = x1 * sin_a + x2 * cos_a;
        }
    }
}

fn main() {
    let path = "/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf";
    let model = GGUFModel::load(path).unwrap();
    let file = &model.file;
    let hidden_size = 3072;
    let num_heads = 24;
    let num_kv_heads = 8;
    let head_dim = 128;
    let intermediate_size = 8192;
    
    let embed = file.get_tensor_row_f32("token_embd.weight", 9906).unwrap();
    
    let mut dequantize = |name: &str| -> Vec<f32> {
        let info = file.get_tensor_info(name).unwrap();
        let count: usize = info.dimensions.iter().map(|&d| d as usize).product();
        let raw = file.get_tensor_raw(name).unwrap();
        let mut out = vec![0.0f32; count];
        let qtype = leafcutter::model::quant::QuantType::from_u32(info.typ).unwrap();
        match qtype {
            leafcutter::model::quant::QuantType::Q5_K => leafcutter::kernels::dequantize_q5_k(raw, &mut out),
            leafcutter::model::quant::QuantType::Q6_K => leafcutter::kernels::dequantize_q6_k(raw, &mut out),
            leafcutter::model::quant::QuantType::Q4_K => leafcutter::kernels::dequantize_q4_k(raw, &mut out),
            _ => panic!("Unsupported quant type for {}: {:?}", name, qtype),
        }
        out
    };
    
    let w_q = dequantize("blk.0.attn_q.weight");
    let w_k = dequantize("blk.0.attn_k.weight");
    let w_v = dequantize("blk.0.attn_v.weight");
    let w_o = dequantize("blk.0.attn_output.weight");
    let w_gate = dequantize("blk.0.ffn_gate.weight");
    let w_up = dequantize("blk.0.ffn_up.weight");
    let w_down = dequantize("blk.0.ffn_down.weight");
    
    let norm_pre = file.get_tensor_row_f32("blk.0.attn_norm.weight", 0).unwrap();
    let norm_post = file.get_tensor_row_f32("blk.0.ffn_norm.weight", 0).unwrap();
    
    // Reference layer 0 computation
    let pre_norm = rms_norm(&embed, &norm_pre, 1e-5);
    let q = matmul(&pre_norm, &w_q, 1, hidden_size, hidden_size);
    let k = matmul(&pre_norm, &w_k, 1, hidden_size, num_kv_heads * head_dim);
    let v = matmul(&pre_norm, &w_v, 1, hidden_size, num_kv_heads * head_dim);
    
    let mut q_rope = q.clone();
    let mut k_rope = k.clone();
    apply_rope(&mut q_rope, &mut k_rope, num_heads, num_kv_heads, head_dim, 500000.0);
    
    // Correct single-token attention: output is just V (replicated for GQA) @ W_o
    let num_kv_groups = num_heads / num_kv_heads;
    let mut attn = vec![0.0f32; num_heads * head_dim];
    for h in 0..num_heads {
        let kv_h = h / num_kv_groups;
        for d in 0..head_dim {
            attn[h * head_dim + d] = v[kv_h * head_dim + d];
        }
    }
    let attn_out = matmul(&attn, &w_o, 1, hidden_size, hidden_size);
    
    let hidden_after_attn = vec_add(&embed, &attn_out);
    let post_norm = rms_norm(&hidden_after_attn, &norm_post, 1e-5);
    let gate_proj = matmul(&post_norm, &w_gate, 1, hidden_size, intermediate_size);
    let up_proj = matmul(&post_norm, &w_up, 1, hidden_size, intermediate_size);
    let activated = silu(&gate_proj);
    let fused = vec_mul(&activated, &up_proj);
    let ffn_out = matmul(&fused, &w_down, 1, intermediate_size, hidden_size);
    let hidden_ref = vec_add(&hidden_after_attn, &ffn_out);
    
    println!("Reference layer 0 output: min={:.6} max={:.6} mean={:.6}",
             hidden_ref.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             hidden_ref.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
             hidden_ref.iter().sum::<f32>() / hidden_ref.len() as f32);
    
    // Leafcutter layer 0 computation
    let mut engine = leafcutter::inference::engine::Engine::load(path).expect("Failed to load");
    let prompt = vec![9906usize];
    let mut hidden_lc = engine.embed_lookup_mmap(&prompt).expect("embed_lookup_mmap failed");
    
    let layer_weights = engine.model.load_layer(0).expect("load layer");
    let pre_norm_weight = layer_weights.get("input_layernorm.weight").unwrap();
    let pre_norm_lc = hidden_lc.rms_norm(pre_norm_weight, 1e-5);
    let attn_out_lc = leafcutter::inference::attention::attention_forward(
        &pre_norm_lc, &layer_weights, &engine.attn_params,
        &mut engine.kv_cache, 0, engine.seq_offset);
    hidden_lc = hidden_lc.add(&attn_out_lc);
    let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
        .or_else(|| layer_weights.get("ffn_norm.weight")).unwrap();
    let post_norm_lc = hidden_lc.rms_norm(post_norm_weight, 1e-5);
    let ffn_out_lc = leafcutter::inference::engine::Engine::ffn_forward(&post_norm_lc, &layer_weights).expect("ffn_forward failed");
    hidden_lc = hidden_lc.add(&ffn_out_lc);
    
    println!("Leafcutter layer 0 output: min={:.6} max={:.6} mean={:.6}",
             hidden_lc.data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             hidden_lc.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
             hidden_lc.data.iter().sum::<f32>() / hidden_lc.data.len() as f32);
    
    let mut max_diff = 0.0f32;
    for i in 0..hidden_ref.len() {
        let diff = (hidden_ref[i] - hidden_lc.data[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("Max diff: {:.6}", max_diff);
}
