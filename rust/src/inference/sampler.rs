//! Token sampling strategies

use rand::Rng;

/// Sample next token using temperature + top-p (nucleus sampling)
pub fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> usize {
    if temperature <= 0.0 {
        // Greedy: pick highest logit (use total_cmp to handle NaN safely)
        return logits.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i).unwrap_or(0);
    }

    // Apply temperature
    let mut probs: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v / temperature)).collect();

    // Softmax
    let max_logit = probs.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = probs.iter().map(|(_, v)| (v - max_logit).exp()).sum();
    for (_, v) in probs.iter_mut() {
        *v = (*v - max_logit).exp() / exp_sum;
    }

    // Sort by probability descending (handle NaN safely)
    probs.sort_by(|(_, a), (_, b)| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Top-p filtering
    let mut cumsum = 0.0f32;
    let cutoff_idx = probs.iter().position(|(_, p)| {
        cumsum += p;
        cumsum > top_p
    }).unwrap_or(probs.len() - 1);

    let filtered: Vec<(usize, f32)> = probs[..=cutoff_idx].to_vec();

    // Renormalize and sample
    let sum: f32 = filtered.iter().map(|(_, p)| p).sum();
    let mut rng = rand::thread_rng();
    let rand_val: f32 = rng.gen::<f32>() * sum;

    let mut cum = 0.0f32;
    for (idx, p) in &filtered {
        cum += *p;
        if rand_val <= cum {
            return *idx;
        }
    }

    filtered.last().map(|(idx, _)| *idx).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy() {
        let logits = vec![0.1, 0.5, 0.3];
        let token = sample_top_p(&logits, 0.0, 0.9);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_temperature() {
        let logits = vec![1.0, 2.0, 3.0];
        let token = sample_top_p(&logits, 1.0, 1.0);
        // Should usually pick index 2, but with randomness it's probabilistic
        assert!(token < logits.len());
    }
}
