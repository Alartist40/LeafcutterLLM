//! Debug: ONE layer, ONE token — print ALL intermediate values.
use leafcutter::streaming_ornith::StreamingOrnith;
use std::path::Path;

fn main() {
    let dir = Path::new("/home/xander/Downloads/models/ornith safetensor");
    let mut model = StreamingOrnith::open(dir).expect("open model");

    let tid = 760;
    let pos = 0;

    // Directly call forward_one_token — but intercept the first layer's computations.
    // We'll use only forward_one_token and read its debug output.
    let logits = model.forward_one_token(tid, pos).expect("forward");

    // Top-1
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("Top-1: id={} logit={:.4} text=\"{}\"",
        indexed[0].0, indexed[0].1, model.tok.decode(&[indexed[0].0 as i32]));
}
