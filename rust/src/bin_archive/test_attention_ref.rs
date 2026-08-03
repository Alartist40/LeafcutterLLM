use leafcutter::inference::attention::{attention_forward, AttentionParams};
use leafcutter::model::loader::GGUFModel;
use leafcutter::model::tensor::Tensor;
use leafcutter::cache::KVCache;

fn main() {
    let model = GGUFModel::load("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
    let weights = model.load_layer(0).unwrap();
    
    let hidden_size = 3072;
    let pre_norm = Tensor::from_vec(vec![0.1f32; hidden_size], vec![1, hidden_size]);
    
    let params = AttentionParams { window_size: 0,
        num_heads: 24, num_kv_heads: 8, head_dim: 128, kv_head_dim: 128,
        rope_theta: 500000.0, rope_dim: 0, use_fused_qkv: false, use_gate: false,
    };
    
    let mut kv_cache = KVCache::new(28);
    let attn_out = attention_forward(&pre_norm, &weights, &params, &mut kv_cache, 0, 0);
    
    // Reference computation for single-token attention:
    // v = pre_norm @ v_proj
    // attn_heads = replicate(v, num_kv_groups)
    // attn_out_ref = attn_heads @ o_proj
    let v_proj = weights.get("self_attn.v_proj.weight").unwrap();
    let o_proj = weights.get("self_attn.o_proj.weight").unwrap();
    
    let v = pre_norm.matmul(v_proj);
    println!("v shape: {:?}, min={:.4} max={:.4}", v.shape,
        v.data.iter().cloned().fold(f32::INFINITY, f32::min),
        v.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    // Replicate v for GQA
    let num_kv_groups = 3;
    let mut attn_heads = vec![0.0f32; 24 * 128];
    for h in 0..24 {
        let kv_h = h / num_kv_groups;
        for d in 0..128 {
            attn_heads[h * 128 + d] = v.data[kv_h * 128 + d];
        }
    }
    let attn_heads_tensor = Tensor::from_vec(attn_heads, vec![1, 3072]);
    let ref_out = attn_heads_tensor.matmul(o_proj);
    
    println!("attn_out  min={:.4} max={:.4} mean={:.6}", 
        attn_out.data.iter().cloned().fold(f32::INFINITY, f32::min),
        attn_out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        attn_out.data.iter().sum::<f32>() / attn_out.data.len() as f32);
    println!("ref_out   min={:.4} max={:.4} mean={:.6}", 
        ref_out.data.iter().cloned().fold(f32::INFINITY, f32::min),
        ref_out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        ref_out.data.iter().sum::<f32>() / ref_out.data.len() as f32);
    
    let mut max_diff = 0.0f32;
    for i in 0..attn_out.data.len() {
        let diff = (attn_out.data[i] - ref_out.data[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("max_diff: {:.6}", max_diff);
}
