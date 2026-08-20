use leafcutter::inference::moe::{MoeConfig, moe_forward_one_token};
use leafcutter::model::loader::GGUFModel;

fn main() {
    let path = "/home/orangepi/Downloads/models/ornith-1.0-35b-Q4_K_M.gguf";
    let model = GGUFModel::load(path).expect("load");
    println!(
        "budget_bytes={} fits={}",
        model.layer_cache_budget_bytes(),
        model.model_fits_available_ram()
    );
    let w = model.load_layer(0).expect("layer 0");
    for (k, t) in w.iter() {
        if k.contains("expert") || k.contains("_exps") || k == "mlp.gate.weight" {
            println!(
                "  {}: shape={:?} data_len={} quantized={}",
                k,
                t.shape,
                t.data.len(),
                t.is_quantized()
            );
        }
    }
    let gate = w.get("mlp.expert_gate.weight").expect("expert gate");
    let down = w.get("mlp.expert_down.weight").expect("expert down");
    for e in [0usize, 1, 255] {
        if let Some(s) = gate.expert_slice(e) {
            println!(
                "  gate slice {}: shape={:?} data_len={} quantized={}",
                e,
                s.shape,
                s.data.len(),
                s.is_quantized()
            );
        }
        if let Some(s) = down.expert_slice(e) {
            println!(
                "  down slice {}: shape={:?} data_len={} quantized={}",
                e,
                s.shape,
                s.data.len(),
                s.is_quantized()
            );
        }
    }
    println!("cached_bytes={}", model.cached_bytes());

    // MoE forward end-to-end on layer 0.
    let get_int = |keys: &[&str]| -> usize {
        for key in keys {
            if let Some(v) = model.file.get_metadata_int(key) {
                return v as usize;
            }
        }
        0
    };
    let num_experts = get_int(&["qwen35moe.expert_count", "llama.expert_count"]);
    let num_experts_used = get_int(&["qwen35moe.expert_used_count", "llama.expert_used_count"]);
    let expert_ffn = get_int(&[
        "qwen35moe.expert_feed_forward_length",
        "llama.expert_feed_forward_length",
    ]);
    let cfg = MoeConfig {
        num_experts,
        num_experts_used,
        expert_ffn,
        gating_func: 2,
        norm_topk_prob: true,
        routed_scaling_factor: 1.0,
        norm_eps: 1e-5,
    };
    println!("MoeConfig: experts={} used={} ffn={}", num_experts, num_experts_used, expert_ffn);

    let hidden = leafcutter::model::tensor::Tensor::from_vec(
        (0..2048).map(|j| ((j % 101) as f32 - 50.0) / 100.0).collect(),
        vec![1, 2048],
    );
    let t0 = std::time::Instant::now();
    let out = moe_forward_one_token(&hidden, &w, &cfg);
    let dt = t0.elapsed();
    println!(
        "moe_forward_one_token: shape={:?} finite={} first={:.4} last={:.4} elapsed={:.3}s",
        out.shape,
        out.data.iter().all(|v| v.is_finite()),
        out.data[0],
        out.data[out.data.len() - 1],
        dt.as_secs_f64()
    );
}