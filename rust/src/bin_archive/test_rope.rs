use leafcutter::inference::attention::{apply_rotary_emb};
use leafcutter::model::tensor::Tensor;

fn main() {
    // Create a simple 2-token, 2-head, 4-dim tensor
    // head 0: [1, 0, 0, 0] for token 0, [1, 0, 0, 0] for token 1
    // head 1: [0, 1, 0, 0] for token 0, [0, 1, 0, 0] for token 1
    let mut q = Tensor::from_vec(vec![
        1.0, 0.0, 0.0, 0.0,  // token 0, head 0
        0.0, 1.0, 0.0, 0.0,  // token 0, head 1
        1.0, 0.0, 0.0, 0.0,  // token 1, head 0
        0.0, 1.0, 0.0, 0.0,  // token 1, head 1
    ], vec![2, 2, 4]);
    
    apply_rotary_emb(&mut q, 2, 2, 4, 0, 10000.0, 0);
    
    println!("After RoPE with position_offset=0:");
    for s in 0..2 {
        for h in 0..2 {
            let base = s * 2 * 4 + h * 4;
            println!("  token {}, head {}: [{:.4}, {:.4}, {:.4}, {:.4}]", 
                s, h, q.data[base], q.data[base+1], q.data[base+2], q.data[base+3]);
        }
    }
}
