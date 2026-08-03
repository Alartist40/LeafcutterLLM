use leafcutter::inference::attention::{attention_forward, AttentionParams};
use leafcutter::model::loader::GGUFModel;
use leafcutter::model::tensor::Tensor;
use leafcutter::cache::KVCache;

fn main() {
    let model = GGUFModel::load("/home/xander/Documents/portfolio/AI Models/Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf").unwrap();
    let weights = model.load_layer(0).unwrap();
    
    let hidden_size = 3072;
    let pre_norm = Tensor::from_vec(vec![0.1f32; 2 * hidden_size], vec![2, hidden_size]);
    
    let params = AttentionParams { window_size: 0,
        num_heads: 24, num_kv_heads: 8, head_dim: 128, kv_head_dim: 128,
        rope_theta: 500000.0, rope_dim: 0, use_fused_qkv: false, use_gate: false,
    };
    
    let mut kv_cache = KVCache::new(28);
    let attn_out = attention_forward(&pre_norm, &weights, &params, &mut kv_cache, 0, 0);
    
    println!("attn_out shape: {:?}", attn_out.shape);
    println!("attn_out token 0 min={:.4} max={:.4} mean={:.6}", 
        attn_out.data[0..hidden_size].iter().cloned().fold(f32::INFINITY, f32::min),
        attn_out.data[0..hidden_size].iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        attn_out.data[0..hidden_size].iter().sum::<f32>() / hidden_size as f32);
    println!("attn_out token 1 min={:.4} max={:.4} mean={:.6}", 
        attn_out.data[hidden_size..2*hidden_size].iter().cloned().fold(f32::INFINITY, f32::min),
        attn_out.data[hidden_size..2*hidden_size].iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        attn_out.data[hidden_size..2*hidden_size].iter().sum::<f32>() / hidden_size as f32);
}
