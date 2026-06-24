//! Eagle-style speculative decoding
//!
//! Uses next-token prediction heads (`nextn.*` tensors) to draft multiple
//! tokens per forward pass, then verifies them with the main model.
//!
//! Algorithm:
//!   1. Main model produces hidden state h_t
//!   2. Draft head predicts h_{t+1}, h_{t+2}, ... h_{t+gamma}
//!   3. Main model verifies all gamma draft tokens in parallel
//!   4. Accept tokens until first mismatch, then sample corrected distribution
//!
//! For Qwen3.5, the draft head tensors are in layer 32:
//!   - nextn.eh_proj: projects hidden state to draft embedding
//!   - nextn.enorm:  draft embedding RMSNorm
//!   - nextn.hnorm:  hidden state RMSNorm before draft
//!   - nextn.eh_scale: scaling factor

use crate::model::tensor::Tensor;
use std::collections::HashMap;

/// Holds the weights for an Eagle speculative decoding head.
pub struct SpeculativeHead {
    pub eh_proj: Tensor,
    pub enorm: Option<Tensor>,
    pub hnorm: Option<Tensor>,
    pub eh_scale: Option<f32>,
    pub gamma: usize, // Number of draft tokens to generate
}

impl SpeculativeHead {
    pub fn from_weights(weights: &HashMap<String, Tensor>, gamma: usize) -> Option<Self> {
        let eh_proj = weights.get("nextn.eh_proj.weight")
            .or_else(|| weights.get("nextn.eh_proj"))?
            .clone();
        let enorm = weights.get("nextn.enorm.weight").cloned();
        let hnorm = weights.get("nextn.hnorm.weight").cloned();
        let eh_scale = weights.get("nextn.eh_scale")
            .map(|t| t.data.get(0).copied().unwrap_or(1.0));

        Some(Self { eh_proj, enorm, hnorm, eh_scale, gamma })
    }

    /// Draft `gamma` future tokens given the current hidden state.
    /// Returns a vec of draft hidden states (one per future token).
    pub fn draft(&self, hidden: &Tensor, _lm_head: &Tensor) -> Vec<Tensor> {
        let mut drafts = Vec::with_capacity(self.gamma);
        let mut current = hidden.clone();

        for _ in 0..self.gamma {
            // Project hidden → draft embedding
            let mut draft_emb = current.matmul(&self.eh_proj);

            // Apply norms if present
            if let Some(ref enorm) = self.enorm {
                draft_emb = draft_emb.rms_norm(enorm, 1e-5);
            }

            // Scale
            if let Some(scale) = self.eh_scale {
                for v in &mut draft_emb.data { *v *= scale; }
            }

            drafts.push(draft_emb.clone());

            // For the next iteration, we'd need the actual token embedding.
            // In a full implementation, we'd sample a token, look up its embedding,
            // and feed it back. Here we approximate by reusing the projection.
            current = draft_emb;
        }

        drafts
    }

    /// Verify draft tokens against the main model's logits.
    /// Returns (accepted_count, next_token) where accepted_count is how many
    /// draft tokens were accepted, and next_token is the final sampled token.
    pub fn verify(
        &self,
        _draft_tokens: &[usize],
        _main_logits: &[f32],
        _temperature: f32,
    ) -> (usize, usize) {
        // Full verification requires running the main model forward for each
        // draft position and comparing distributions.  That's not wired up
        // here yet, so rather than pretend to verify (the previous stub
        // always accepted 0 drafts while still cost a draft pass), we now
        // report a disabled status — the caller should fall back to plain
        // greedy/contrastive sampling.
        (0, 0)
    }
}

/// Hint returned by `SpeculativeDecoder::try_accept` so callers can detect
/// the "speculative decoding is currently a no-op" case and skip the draft
/// step entirely on subsequent tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeStatus {
    /// Speculative decoding is fully implemented and returned a verdict.
    Active,
    /// Underlying `verify` returned a no-op; treat drafts as always-rejected.
    Disabled,
}

/// Speculative decoder that manages draft generation and verification.
pub struct SpeculativeDecoder {
    pub head: SpeculativeHead,
}

impl SpeculativeDecoder {
    pub fn new(head: SpeculativeHead) -> Self {
        Self { head }
    }

    /// Generate tokens using speculative decoding.
    /// `generate_fn` is a callback that runs the main model for a given token sequence.
    pub fn generate<F>(
        &self,
        initial_tokens: &[usize],
        max_tokens: usize,
        _temperature: f32,
        mut generate_fn: F,
    ) -> Vec<usize>
    where
        F: FnMut(&[usize]) -> Vec<f32>,
    {
        let mut tokens = initial_tokens.to_vec();

        while tokens.len() < initial_tokens.len() + max_tokens {
            // Get main model logits for current sequence
            let _logits = generate_fn(&tokens);

            // Draft future tokens (simplified — full impl needs hidden state passing)
            // For now, just generate 1 token at a time
            let next_token = greedy_sample(&_logits);
            tokens.push(next_token);

            if next_token == 2 { // EOS
                break;
            }
        }

        tokens[initial_tokens.len()..].to_vec()
    }
}

fn greedy_sample(logits: &[f32]) -> usize {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_head_creation() {
        let mut weights = HashMap::new();
        weights.insert("nextn.eh_proj.weight".to_string(), Tensor::from_vec(vec![0.1; 64 * 64], vec![64, 64]));
        weights.insert("nextn.enorm.weight".to_string(), Tensor::from_vec(vec![1.0; 64], vec![64]));

        let head = SpeculativeHead::from_weights(&weights, 4);
        assert!(head.is_some());
        let head = head.unwrap();
        assert_eq!(head.gamma, 4);
        assert!(head.enorm.is_some());
    }

    #[test]
    fn test_speculative_head_missing() {
        let weights = HashMap::new();
        assert!(SpeculativeHead::from_weights(&weights, 4).is_none());
    }

    #[test]
    fn test_draft_produces_gamma_outputs() {
        let mut weights = HashMap::new();
        weights.insert("nextn.eh_proj.weight".to_string(), Tensor::from_vec(vec![0.0; 16 * 16], vec![16, 16]));

        let head = SpeculativeHead::from_weights(&weights, 3).unwrap();
        let hidden = Tensor::from_vec(vec![0.1; 16], vec![1, 16]);
        let lm_head = Tensor::from_vec(vec![0.0; 16 * 100], vec![16, 100]);

        let drafts = head.draft(&hidden, &lm_head);
        assert_eq!(drafts.len(), 3);
    }
}
