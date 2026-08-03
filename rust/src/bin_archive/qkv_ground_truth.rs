//! Compare engine Q6_K/Q4_K matmul vs full dequantization ground truth.
//!
//! For the same input vector `a`, we compute:
//!   c1 = matmul_engine (a, qtensor)        -- uses the engine's Q6_K/Q4_K kernel
//!   c2 = matmul_naive  (a, dequant(qtensor))-- ground truth
//!
//! If the engine kernel is correct, c1 == c2 within fp32 epsilon.
//!
//! Run with:
//!   cargo build --release --no-default-features --bin qkv_ground_truth
//!   ./target/release/qkv_ground_truth /path/to/model.gguf blk.0.attn_qkv.weight
use leafcutter::model::loader::GGUFModel;
use leafcutter::model::tensor::Tensor;

/// Map engine tensor name to GGUF tensor name for layer 0 (test only).
/// Engine name: "self_attn.qkv_proj.weight"
/// GGUF name  : "blk.0.attn_qkv.weight"
fn gguf_name_for_layer0(engine_name: &str) -> String {
    // Known mappings for the tensors we'll test against
    match engine_name {
        "self_attn.qkv_proj.weight" => "blk.0.attn_qkv.weight".to_string(),
        "self_attn.out_proj.weight" => "blk.0.attn_output.weight".to_string(),
        "mlp.gate_proj.weight" => "blk.0.ffn_gate.weight".to_string(),
        "mlp.up_proj.weight" => "blk.0.ffn_up.weight".to_string(),
        "mlp.down_proj.weight" => "blk.0.ffn_down.weight".to_string(),
        "ssm_alpha.weight" => "blk.0.ssm_alpha.weight".to_string(),
        "ssm_conv1d.weight" => "blk.0.ssm_conv1d.weight".to_string(),
        "ssm_dt.weight" => "blk.0.ssm_dt.weight".to_string(),
        "ssm_out.weight" => "blk.0.ssm_out.weight".to_string(),
        "ssm_b.weight" => "blk.0.ssm_b.weight".to_string(),
        "pre_norm.weight" => "blk.0.attn_norm.weight".to_string(),
        "post_norm.weight" => "blk.0.post_attention_norm.weight".to_string(),
        _ => panic!("unknown engine tensor name: {}", engine_name),
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf".to_string());
    let tensor_name = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "self_attn.qkv_proj.weight".to_string());

    let model = GGUFModel::load(&path).expect("load gguf");
    let weights = model.load_layer(0).expect("load layer 0");
    let qtensor = weights.get(&tensor_name).expect("missing tensor");

    println!("== qkv_ground_truth ==");
    println!("path    : {}", path);
    println!("tensor  : {}", tensor_name);
    println!("shape   : {:?}", qtensor.shape);

    let gguf_name = gguf_name_for_layer0(&tensor_name);
    println!("gguf    : {} (for raw lookup)", gguf_name);
    let info_dbg = model
        .file
        .get_tensor_info(&gguf_name)
        .expect("info (gguf name)");
    println!("qtype   : {} (typ code)", info_dbg.typ);

    let m = 1;
    // Use the FIRST dim as k (input), second as n (output)
    let k = qtensor.shape[0];
    let n = qtensor.shape[1];

    // Build a small but representative input vector. Use a simple ramp —
    // it makes element-wise diff easy to inspect and verifies the dot product
    // sums correctly.
    let pre_norm_path = std::env::args().nth(3);
    let a: Vec<f32> = if let Some(p) = &pre_norm_path {
        let raw_a = std::fs::read_to_string(p)
            .unwrap_or_else(|e| panic!("read {}: {}", p, e));
        raw_a
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f32>().expect("parse f32"))
            .collect()
    } else {
        (0..k).map(|i| (i as f32) * 0.001 - 0.5).collect()
    };
    assert_eq!(a.len(), k, "input len {} must equal k={}", a.len(), k);
    println!("a       : len={}, [0]={:.6}, [k-1]={:.6}", a.len(), a[0], a[k - 1]);
    let atensor = Tensor::from_vec(a.clone(), vec![m, k]);

    // Path 1: engine matmul (uses the K-quantized kernel via q_data)
    let c1 = atensor.matmul(qtensor);
    println!(
        "\n[engine matmul] min={:.5} max={:.5} abs_mean={:.5}",
        c1.data.iter().cloned().fold(f32::INFINITY, f32::min),
        c1.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        c1.data.iter().map(|v| v.abs()).sum::<f32>() / c1.data.len() as f32
    );

    // Path 2: full dequantization + naive matmul (ground truth)
    // GGUF stores Q4_K/Q6_K in [out, in] = [n, k] layout (each block = 256 elements,
    // blocks laid out row-major as [n, k]).  So the dequantized tensor shape is [n, k],
    // and the standard nn.Linear forward computes `input @ W^T` where W is [n, k].
    // We do that matmul here as ground truth.
    let raw = model.file.get_tensor_raw(&gguf_name).unwrap_or_else(|| {
        let prefix = format!("blk.0.");
        let names: Vec<&str> = model
            .file
            .tensors
            .iter()
            .filter(|t| t.name.starts_with(&prefix))
            .map(|t| t.name.as_str())
            .take(20)
            .collect();
        panic!(
            "missing tensor raw for {}. layer-0 tensors: {:?}",
            gguf_name, names
        )
    });
    let info = model
        .file
        .get_tensor_info(&gguf_name)
        .expect("missing tensor info");
    let count: usize = info.dimensions.iter().map(|&d| d as usize).product();
    let mut deq = vec![0.0f32; count];
    match info.typ {
        12 => leafcutter::kernels::dequantize_q4_k(raw, &mut deq),
        14 => leafcutter::kernels::dequantize_q6_k(raw, &mut deq),
        n => panic!("unsupported qtype={}", n),
    }
    println!(
        "\n[dequant] min={:.5} max={:.5} abs_mean={:.5}",
        deq.iter().cloned().fold(f32::INFINITY, f32::min),
        deq.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        deq.iter().map(|v| v.abs()).sum::<f32>() / deq.len() as f32
    );

    // The naive matmul uses the dequant shape directly. Tensor::matmul requires
    // self.shape[1] == other.shape[0]. atensor is [m=1, k=4096].
    //   - as [k, n]: matmul does a @ deq -> [1, n=8192]. Asserts pass.
    //   - as [n, k]: matmul requires self.shape[1]=k=4096 == other.shape[0]=n=8192 (FAILS).
    let dtensor = Tensor::from_vec(deq.clone(), vec![k, n]);
    let c2 = atensor.matmul(&dtensor);
    println!(
        "[dequant+matmul (dequant as [k,n])] min={:.5} max={:.5} abs_mean={:.5}",
        c2.data.iter().cloned().fold(f32::INFINITY, f32::min),
        c2.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        c2.data.iter().map(|v| v.abs()).sum::<f32>() / c2.data.len() as f32
    );

    // For "dequant as [n, k]" we need a transposed input — compute C = A @ D^T
    // where D is the [n, k] view of dequant.
    //   C[i, j] = sum_l A[i, l] * D[j, l]
    // Build it as C[j] = sum_l A[l] * D[j, l] for j in 0..n
    let mut c3 = vec![0.0f32; n];
    for j in 0..n {
        let mut acc = 0.0f32;
        for l in 0..k {
            acc += a[l] * deq[j * k + l];
        }
        c3[j] = acc;
    }
    println!(
        "[dequant+matmul (dequant as [n,k])] min={:.5} max={:.5} abs_mean={:.5}",
        c3.iter().cloned().fold(f32::INFINITY, f32::min),
        c3.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        c3.iter().map(|v| v.abs()).sum::<f32>() / c3.len() as f32
    );

    let diff_c1c2 = c1
        .data
        .iter()
        .zip(c2.data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let diff_c1c3 = c1.data.iter().zip(c3.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("\n== RESULTS ==");
    println!("max_diff(engine vs dequant_as_[k,n]) = {:.5}", diff_c1c2);
    println!("max_diff(engine vs dequant_as_[n,k]) = {:.5}", diff_c1c3);

    // Show first 8 elements of each
    println!("\nfirst 8 elements:");
    for i in 0..8.min(c1.data.len()) {
        println!(
            "  [{:2}] engine={:>+10.5}  deq[k,n]={:>+10.5}  deq[n,k]={:>+10.5}",
            i, c1.data[i], c2.data[i], c3[i]
        );
    }
}
