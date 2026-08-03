//! Reference DeltaNet layer-0 for Ornith/Qwen3.5 — single-token forward.
//!
//! Computes layer 0 (a DeltaNet layer) two ways and compares:
//!   1. Pure-Rust reference (no engine code) — RMSNorm → DeltaNet → residual
//!      → RMSNorm → SwiGLU FFN → residual
//!   2. Leafcutter engine: Engine::load, embed_lookup, load_layer(0),
//!      deltanet_forward, ffn_forward
//!
//! For seq_len=1 the DeltaNet state starts at zero — so the first call
//! is deterministic and isolates the math (no state accumulation).

use leafcutter::model::gguf::GGUFile;
use leafcutter::model::loader::GGUFModel;

// ── Pure-Rust math primitives (independent of engine) ──────────────────

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(&x, &w)| x * scale * w).collect()
}

/// matmul: a [m,k] @ b [k,n] = c [m,n]   (row-major, b NOT transposed)
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                s += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// matmul_t: a [m,k] @ b^T [k,n] = c [m,n]
/// where b is stored row-major as [n, k] (output-major, GGUF K-quant layout).
/// c[i,j] = sum_l a[i*k + l] * b[j*k + l]
/// This is the CORRECT nn.Linear forward: x @ W.T where W is [n_out, k_in].
fn matmul_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                s += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = s;
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
fn softplus(x: f32) -> f32 {
    // numerically stable softplus
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { x.exp() / (1.0 + x.exp()) }
}

// ── GGUF tensor dequantization helper ───────────────────────────────────

fn dequant(file: &GGUFile, name: &str) -> Vec<f32> {
    let info = file.get_tensor_info(name).unwrap_or_else(|| panic!("missing {}", name));
    let count: usize = info.dimensions.iter().map(|&d| d as usize).product();
    let raw = file.get_tensor_raw(name).unwrap();
    let mut out = vec![0.0f32; count];
    let qtype = leafcutter::model::quant::QuantType::from_u32(info.typ).unwrap();
    use leafcutter::model::quant::QuantType::*;
    match qtype {
        Q4_K => leafcutter::kernels::dequantize_q4_k(raw, &mut out),
        Q5_K => leafcutter::kernels::dequantize_q5_k(raw, &mut out),
        Q6_K => leafcutter::kernels::dequantize_q6_k(raw, &mut out),
        F32 => {
            out.copy_from_slice(bytemuckcast(raw));
        }
        _ => panic!("unsupported qtype {:?} for {}", qtype, name),
    }
    out
}

fn bytemuckcast(bytes: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let path = std::env::args().nth(1)
        .unwrap_or("/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf".to_string());
    let token_id: usize = std::env::args().nth(2)
        .and_then(|s| s.parse().ok()).unwrap_or(760);  // "The" by default
    eprintln!("ref_deltanet0: model={} token_id={}", path, token_id);

    let model = GGUFModel::load(&path).expect("load");
    let file = &model.file;

    // Model layout (Ornith / Qwen3.5)
    let hidden = 4096;
    let conv_dim = 8192;
    let num_qk_heads = 16;
    let num_v_heads = 32;
    let head_k_dim = 128;
    let head_v_dim = 128;
    let inter = 12288;
    let eps = 1e-6;
    let v_heads_per_qk = num_v_heads / num_qk_heads; // 2

    // ── Load weights (dequantized) ──
    let w_qkv       = dequant(file, "blk.0.attn_qkv.weight");      // [4096, 8192]
    let w_gate      = dequant(file, "blk.0.attn_gate.weight");      // [4096, 4096]
    let w_ssm_alpha = dequant(file, "blk.0.ssm_alpha.weight");      // [4096, 32]
    let w_ssm_beta  = dequant(file, "blk.0.ssm_beta.weight");       // [4096, 32]
    let w_ssm_conv1d = dequant(file, "blk.0.ssm_conv1d.weight");    // [4, 8192]
    let ssm_dt_bias = dequant(file, "blk.0.ssm_dt.bias");          // [32]
    let ssm_a       = dequant(file, "blk.0.ssm_a");                 // [32]
    let w_ssm_norm  = dequant(file, "blk.0.ssm_norm.weight");      // [128]
    let w_ssm_out   = dequant(file, "blk.0.ssm_out.weight");       // [4096, 4096]
    let w_ffn_gate  = dequant(file, "blk.0.ffn_gate.weight");      // [4096, 12288]
    let w_ffn_up    = dequant(file, "blk.0.ffn_up.weight");        // [4096, 12288]
    let w_ffn_down  = dequant(file, "blk.0.ffn_down.weight");      // [12288, 4096]
    let norm_pre    = dequant(file, "blk.0.attn_norm.weight");    // [4096]
    let norm_post   = dequant(file, "blk.0.post_attention_norm.weight"); // [4096]

    // ── Input: embedding lookup ──
    let embed = file.get_tensor_row_f32("token_embd.weight", token_id).unwrap();

    // ============================================================
    // 1. Pure-Rust reference DeltaNet layer 0 — single token
    // ============================================================
    let pre_norm = rms_norm(&embed, &norm_pre, eps);

    // Optional: dump pre_norm to a file for downstream tests.
    if let Ok(path) = std::env::var("DUMP_PRE_NORM") {
        let mut s = String::with_capacity(pre_norm.len() * 16);
        for v in &pre_norm {
            s.push_str(&format!("{}\n", v));
        }
        std::fs::write(&path, s).expect("write pre_norm");
        eprintln!("Wrote pre_norm to {}", path);
    }

    // QKV projection: [1, 4096] @ [4096, 8192] -> [8192]
    let qkv = matmul_t(&pre_norm, &w_qkv, 1, hidden, conv_dim);

    // Now compare against the engine's matmul result (engine uses Q4_K
    // matmul-transposed-b path; we use full dequant + naive matmul).
    // We need to replicate that path exactly here.
    let info = file.get_tensor_info("blk.0.attn_qkv.weight").unwrap();
    let raw = file.get_tensor_raw("blk.0.attn_qkv.weight").unwrap();
    // Build the same matmul as the engine and compare
    use leafcutter::model::quant::QuantType;
    let qtype = QuantType::from_u32(info.typ).unwrap();
    let dims = info.dimensions.iter().map(|&d| d as usize).collect::<Vec<_>>();
    let (n_qkv, k_qkv) = (dims[0], dims[1]);
    eprintln!("w_qkv qtype={:?} dims={:?}", qtype, dims);
    // Try matching the engine by going through the engine's matmul path
    let mut engine = leafcutter::inference::engine::Engine::load(&path).expect("engine load");
    let mut hidden_e = engine.embed_lookup_mmap(&[token_id]).expect("embed lookup");
    let layer_weights = engine.model.load_layer(0).expect("load layer 0");
    let pre_norm_weight = layer_weights.get("input_layernorm.weight")
        .or_else(|| layer_weights.get("attn_norm.weight"))
        .expect("pre-norm");
    let normed_e = hidden_e.rms_norm(pre_norm_weight, engine.config.norm_eps);
    if let Some(qkv_w) = layer_weights.get("self_attn.qkv_proj.weight")
        .or_else(|| layer_weights.get("attn_qkv.weight")) {
        let qkv_e = normed_e.matmul(qkv_w);
        let qkv_e_max = qkv_e.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let qkv_e_min = qkv_e.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let qkv_e_abs_mean: f32 = qkv_e.data.iter().map(|v| v.abs()).sum::<f32>()
            / qkv_e.data.len() as f32;
        eprintln!("engine qkv: min={:.5} max={:.5} abs_mean={:.5}", qkv_e_min, qkv_e_max, qkv_e_abs_mean);
        let qkv_e_shape = &qkv_e.shape;
        eprintln!("engine qkv shape: {:?}", qkv_e_shape);
    }
    println!("\n-- conv1d output (first 16 of 8192) --");
    // For the very first token the conv sum reduces to:
    //   out[c] = conv_w[0, c] * x[c]   (since the rest of the kernel
    //   window sees the zero initial state).
    // Then SiLU.
    let mut conv_out = vec![0.0f32; conv_dim];
    for c in 0..conv_dim {
        // kernel_size=4; single token: position 0 of the kernel is the
        // "current" element; the other 3 positions weight zero state.
        let w0 = w_ssm_conv1d[0 * conv_dim + c];
        let v_pre = qkv[c] * w0;
        let v = v_pre * sigmoid(v_pre); // SiLU(x) = x * sigmoid(x)
        conv_out[c] = v;
        if c < 16 {
            println!("  [c={:4}] qkv={:>+8.4} w={:>+8.5} pre={:>+10.5} post={:>+10.5}",
                c, qkv[c], w0, v_pre, v);
        }
    }
    println!("\n-- conv1d output (first 16 of 8192) --");
    for c in 0..16 {
        println!("  [{:4}] = {:>+10.5}", c, conv_out[c]);
    }
    let qkv_max = qkv.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let qkv_min = qkv.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("qkv: min={:.5} max={:.5}", qkv_min, qkv_max);
    let w0_max = (0..conv_dim).map(|c| w_ssm_conv1d[c]).fold(f32::NEG_INFINITY, f32::max);
    let w0_min = (0..conv_dim).map(|c| w_ssm_conv1d[c]).fold(f32::INFINITY, f32::min);
    println!("w_conv1d[0,*]: min={:.5} max={:.5}", w0_min, w0_max);
    let conv_max = conv_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let conv_min = conv_out.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("conv_out: min={:.5} max={:.5}", conv_min, conv_max);

    // Split conv output into Q, K, V
    let q_total = num_qk_heads * head_k_dim; // 2048
    let k_total = num_qk_heads * head_k_dim; // 2048
    let v_total = num_v_heads * head_v_dim;  // 4096
    let mut q = vec![0.0f32; q_total];
    let mut k = vec![0.0f32; k_total];
    let v: Vec<f32> = conv_out[q_total + k_total..q_total + k_total + v_total].to_vec();
    q.copy_from_slice(&conv_out[..q_total]);
    k.copy_from_slice(&conv_out[q_total..q_total + k_total]);

    // L2-normalize Q and K per head
    for h in 0..num_qk_heads {
        let base = h * head_k_dim;
        let mut sq = 0.0f32;
        for d in 0..head_k_dim {
            sq += q[base + d] * q[base + d];
            sq += k[base + d] * k[base + d];
        }
        let nq = (sq / (2.0 * head_k_dim as f32) + eps).sqrt();
        for d in 0..head_k_dim {
            q[base + d] /= nq;
            k[base + d] /= nq;
        }
    }
    // Wait — that normalizes Q and K together. Let me fix: each head has its own Q norm and K norm.

    // Re-do L2 normalization properly:
    let mut q = conv_out[..q_total].to_vec();
    let mut k = conv_out[q_total..q_total + k_total].to_vec();
    let v: Vec<f32> = conv_out[q_total + k_total..].to_vec();

    for h in 0..num_qk_heads {
        let base = h * head_k_dim;
        let mut qsq = 0.0f32;
        for d in 0..head_k_dim { qsq += q[base + d] * q[base + d]; }
        let qn = (qsq + eps).sqrt();
        for d in 0..head_k_dim { q[base + d] /= qn; }

        let mut ksq = 0.0f32;
        for d in 0..head_k_dim { ksq += k[base + d] * k[base + d]; }
        let kn = (ksq + eps).sqrt();
        for d in 0..head_k_dim { k[base + d] /= kn; }
    }
    // Scale Q by 1/sqrt(head_k_dim)
    let scale = 1.0f32 / (head_k_dim as f32).sqrt();
    for v in q.iter_mut() { *v *= scale; }

    // Decay rates:  dt = softplus(alpha + dt_bias); decay = exp(dt * ssm_a)
    // alpha = hidden @ ssm_alpha.weight  -> [32]
    let alpha_vec = matmul_t(&pre_norm, &w_ssm_alpha, 1, hidden, num_v_heads);
    let mut decay = vec![0.0f32; num_v_heads];
    for h in 0..num_v_heads {
        let dt = softplus(alpha_vec[h] + ssm_dt_bias[h]);
        let a = ssm_a[h];          // already A = -exp(A_log), negative
        decay[h] = (dt * a).exp(); // (0, 1)
    }

    // Beta gates:  beta = sigmoid(hidden @ ssm_beta.weight) -> [32]
    let beta_vec = matmul_t(&pre_norm, &w_ssm_beta, 1, hidden, num_v_heads);
    let beta: Vec<f32> = (0..num_v_heads).map(|h| sigmoid(beta_vec[h])).collect();

    // Delta rule state update + output, for seq_len=1 with zero initial state:
    //   S starts at 0
    //   v_pred = S @ k = 0  (first token, S=0)
    //   delta_v = v - v_pred = v
    //   S = decay*S + beta * delta_v ⊗ k = beta * v ⊗ k
    //     -> S[i,j] = beta_h * v_h[i] * k_h[j]
    //   o_h[i] = S[i,j] @ q[j] = beta_h * v_h[i] * (k_h . q_h)
    // For h_qk -> h_v mapping: h_v = h_qk * v_heads_per_qk + v_idx
    let mut delta_out = vec![0.0f32; num_v_heads * head_v_dim];
    for h_qk in 0..num_qk_heads {
        let q_h = &q[h_qk * head_k_dim..(h_qk + 1) * head_k_dim];
        let k_h = &k[h_qk * head_k_dim..(h_qk + 1) * head_k_dim];
        // Dot product k_h . q_h (already scaled)
        let mut kq_dot = 0.0f32;
        for d in 0..head_k_dim {
            kq_dot += k_h[d] * q_h[d];
        }
        for v_idx in 0..v_heads_per_qk {
            let h_v = h_qk * v_heads_per_qk + v_idx;
            let v_h = &v[h_v * head_v_dim..(h_v + 1) * head_v_dim];
            let beta_h = beta[h_v];
            let decay_h = decay[h_v];
            // First-token output: beta * v * (k · q)   (decay*S=0)
            // Note: state init is zero so decay_h doesn't enter first output
            let out_base = h_v * head_v_dim;
            for d in 0..head_v_dim {
                delta_out[out_base + d] = beta_h * v_h[d] * kq_dot;
            }
        }
    }

    // Per-head RMSNorm with ssm_norm.weight [128]
    let mut normed_out = vec![0.0f32; num_v_heads * head_v_dim];
    for h in 0..num_v_heads {
        let base = h * head_v_dim;
        let mut sq = 0.0f32;
        for d in 0..head_v_dim { sq += delta_out[base + d] * delta_out[base + d]; }
        let rms = (sq / head_v_dim as f32 + eps).sqrt();
        for d in 0..head_v_dim {
            normed_out[base + d] = (delta_out[base + d] / rms) * w_ssm_norm[d];
        }
    }

    // Gate: z = hidden @ attn_gate.weight  -> [4096] ; g = SiLU(z)
    let z = matmul_t(&pre_norm, &w_gate, 1, hidden, hidden);
    let mut gated = vec![0.0f32; hidden];
    for i in 0..hidden {
        gated[i] = normed_out[i] * silu(&[z[i]])[0];
    }
    // NOTE: the gate uses the *current* hidden (pre-norm) input. Worth verifying
    // against deltanet.rs later, but matches the comment at line 246.

    // Output projection: [4096] @ [4096, 4096] -> [4096]
    let ssm_proj_out = matmul_t(&gated, &w_ssm_out, 1, hidden, hidden);

    // Residual: hidden += ssm_proj_out
    let after_attn = vec_add(&embed, &ssm_proj_out);

    // FFN: post_norm → SwiGLU(gate, up) → down → residual
    let post_norm = rms_norm(&after_attn, &norm_post, eps);
    let gate_proj = matmul_t(&post_norm, &w_ffn_gate, 1, hidden, inter);
    let up_proj   = matmul_t(&post_norm, &w_ffn_up,   1, hidden, inter);
    let activated = silu(&gate_proj);
    let fused = vec_mul(&activated, &up_proj);
    let ffn_out = matmul_t(&fused, &w_ffn_down, 1, inter, hidden);
    let hidden_ref = vec_add(&after_attn, &ffn_out);

    // ── Print reference stats ──
    let ref_min = hidden_ref.iter().cloned().fold(f32::INFINITY, f32::min);
    let ref_max = hidden_ref.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ref_mean: f32 = hidden_ref.iter().sum::<f32>() / hidden_ref.len() as f32;
    let ref_abs_mean: f32 = hidden_ref.iter().map(|v| v.abs()).sum::<f32>() / hidden_ref.len() as f32;
    println!("ref: min={:.6} max={:.6} mean={:.6} abs_mean={:.6}",
             ref_min, ref_max, ref_mean, ref_abs_mean);

    // ============================================================
    // 2. Leafcutter engine layer 0 (same token)
    // ============================================================
    let mut engine = leafcutter::inference::engine::Engine::load(&path)
        .expect("engine load");
    let mut hidden_lc = engine.embed_lookup_mmap(&[token_id])
        .expect("embed lookup");

    let layer_weights = engine.model.load_layer(0).expect("load layer 0");
    let pre_norm_weight = layer_weights.get("input_layernorm.weight")
        .or_else(|| layer_weights.get("attn_norm.weight"))
        .expect("missing pre-norm");
    let normed = hidden_lc.rms_norm(pre_norm_weight, engine.config.norm_eps);

    let deltanet_out = leafcutter::inference::deltanet::deltanet_forward(
        &normed, &layer_weights, &engine.deltanet_params,
        &mut engine.deltanet_cache, 0);
    hidden_lc = hidden_lc.add(&deltanet_out);

    let post_norm_weight = layer_weights.get("post_attention_layernorm.weight")
        .or_else(|| layer_weights.get("ffn_norm.weight"))
        .expect("missing post-norm");
    let post_normed = hidden_lc.rms_norm(post_norm_weight, engine.config.norm_eps);
    let ffn_out_lc = leafcutter::inference::engine::Engine::ffn_forward(
        &post_normed, &layer_weights).expect("ffn");
    hidden_lc = hidden_lc.add(&ffn_out_lc);

    let lc_min = hidden_lc.data.iter().cloned().fold(f32::INFINITY, f32::min);
    let lc_max = hidden_lc.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lc_mean: f32 = hidden_lc.data.iter().sum::<f32>() / hidden_lc.data.len() as f32;
    let lc_abs_mean: f32 = hidden_lc.data.iter().map(|v| v.abs()).sum::<f32>() / hidden_lc.data.len() as f32;
    println!("lc:  min={:.6} max={:.6} mean={:.6} abs_mean={:.6}",
             lc_min, lc_max, lc_mean, lc_abs_mean);

    // ── Element-wise diff ──
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0usize;
    let mut rel_sum = 0.0f32;
    let mut rel_count = 0usize;
    for i in 0..hidden_ref.len() {
        let d = (hidden_ref[i] - hidden_lc.data[i]).abs();
        if d > max_diff { max_diff = d; max_diff_idx = i; }
        let denom = hidden_ref[i].abs().max(1e-6);
        if denom > 1e-6 {
            rel_sum += d / denom;
            rel_count += 1;
        }
    }
    println!("max_diff = {:.6} at idx {}", max_diff, max_diff_idx);
    if rel_count > 0 {
        println!("mean_rel_diff = {:.6}", rel_sum / rel_count as f32);
    }

    // Detailed diff of the DeltaNet-only output (root cause isolation):
    println!("\n-- DeltaNet output element-wise (first 16 of 4096) --");
    for i in 0..16 {
        println!("  [{:4}] ref={:>+10.5}  lc={:>+10.5}  diff={:>9.5}",
                 i, ssm_proj_out[i], deltanet_out.data[i],
                 (ssm_proj_out[i] - deltanet_out.data[i]).abs());
    }

    // And the pre-normed hidden (should match exactly):
    println!("\n-- Pre-normed hidden (first 16) --");
    for i in 0..16 {
        println!("  [{:4}] ref={:>+10.5}  lc={:>+10.5}",
                 i, pre_norm[i], normed.data[i]);
    }

    let ssm_proj_min = ssm_proj_out.iter().cloned().fold(f32::INFINITY, f32::min);
    let ssm_proj_max = ssm_proj_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ssm_proj_mean: f32 = ssm_proj_out.iter().sum::<f32>() / ssm_proj_out.len() as f32;
    let ssm_proj_abs_mean: f32 = ssm_proj_out.iter().map(|v| v.abs()).sum::<f32>() / ssm_proj_out.len() as f32;
    println!("\nref deltanet_out: min={:.5} max={:.5} mean={:.5} abs_mean={:.5}",
             ssm_proj_min, ssm_proj_max, ssm_proj_mean, ssm_proj_abs_mean);
    let lo_min = deltanet_out.data.iter().cloned().fold(f32::INFINITY, f32::min);
    let lo_max = deltanet_out.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lo_mean: f32 = deltanet_out.data.iter().sum::<f32>() / deltanet_out.data.len() as f32;
    let lo_abs_mean: f32 = deltanet_out.data.iter().map(|v| v.abs()).sum::<f32>() / deltanet_out.data.len() as f32;
    println!("lc  deltanet_out: min={:.5} max={:.5} mean={:.5} abs_mean={:.5}",
             lo_min, lo_max, lo_mean, lo_abs_mean);
}

