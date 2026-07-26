//! Inference-time doom-loop detection + suppression.
//!
//! Ported from Liquid4All/antidoom (Apache-2.0) — the offline preference
//! data generator — to an online inference hook for LeafcutterLLM's native
//! generation loop.  See `references/antidoom-README.md` for the upstream
//! project.  Originally written for training-time FTPO pair generation;
//! adapted here as a sampler-after-pass that runs after each forward pass,
//! detects the start of an inner repetition, and zeroes the offending
//! token's logit so the sampler picks a coherent alternative next step.
//!
//! This does NOT train the model — it is a runtime guard for the
//! "every token counts given slow tok/s" property the user requires.
//!
//! Gate: `LEAFCUTTER_ANTIDOOM=1` opt-in (debug default off).

use std::env;
use std::time::Instant;

/// A detected inner repetition in generated text.
///
/// `start`/`end` are byte offsets into the generated text. `period` is the
/// length of the repeated unit. `repeats` is the total count (forward +
/// backward extension). `snippet` is a short prefix of the repeated unit
/// for diagnostics.
#[derive(Debug, Clone)]
pub struct RepeatHit {
    pub start: usize,
    pub end: usize,
    pub period: usize,
    pub repeats: usize,
    pub snippet: String,
}

impl RepeatHit {
    /// Byte index in the generated text where the *second* occurrence of
    /// the repeated unit begins.  This is the position where the model
    /// "chose to enter the loop" rather than diverge — i.e. where we want
    /// to intervene.
    pub fn repeat_start(&self) -> usize {
        self.start + self.period
    }
}

/// Verify that `text` has an inner repetition of length `period` anchored
/// at byte `start_pos`.  Walks forward (counting repeats) and backward
/// (extending the start).  Returns `Some(RepeatHit)` if the matching
/// repetition meets both the min_repeats and min_total_repeated
/// thresholds.  Otherwise `None`.
pub fn verify_repetition_at(
    text: &str,
    start_pos: usize,
    period: usize,
    min_repeats: usize,
    min_total_repeated: usize,
) -> Option<RepeatHit> {
    if period < 1 || start_pos + period > text.len() {
        return None;
    }
    // Ensure start_pos and start_pos + period are on char boundaries
    if !text.is_char_boundary(start_pos) {
        return None;
    }
    if !text.is_char_boundary(start_pos + period) {
        return None;
    }

    let pattern = &text[start_pos..start_pos + period];

    // Forward count
    let mut reps = 0;
    let mut pos = start_pos;
    while pos + period <= text.len() && text.get(pos..pos + period) == Some(pattern) {
        reps += 1;
        pos += period;
    }
    let end_pos = pos;

    // Backward extension
    let mut actual_start = start_pos;
    let mut back_pos = start_pos as isize - period as isize;
    while back_pos >= 0 {
        let bp = back_pos as usize;
        if text.get(bp..bp + period) == Some(pattern) {
            reps += 1;
            actual_start = bp;
            back_pos -= period as isize;
        } else {
            break;
        }
    }

    let total = reps * period;
    if reps >= min_repeats && total >= min_total_repeated {
        let snippet = if pattern.len() <= 100 {
            pattern.to_string()
        } else {
            // Ensure char boundary for snippet
            let mut snip_end = 100;
            while snip_end < pattern.len() && !pattern.is_char_boundary(snip_end) {
                snip_end += 1;
            }
            format!("{}...", &pattern[..snip_end])
        };
        Some(RepeatHit {
            start: actual_start,
            end: end_pos,
            period,
            repeats: reps,
            snippet,
        })
    } else {
        None
    }
}

/// Scan the generated `text` for an inner repetition.  Returns the first
/// detected `RepeatHit` (if any).  Algorithm matches antidoom's
/// `find_inner_repetition`:
///   1) Sample positions at `sample_interval` stride.
///   2) At each, take a `sample_len`-char fingerprint.
///   3) Look for the fingerprint elsewhere in the text.  If found, the gap
///      is a candidate `period`.
///   4) Call `verify_repetition_at` to confirm a repetition that meets
///      the count/total thresholds.
///
/// Defaults tuned for ONLINE inference (more sensitive than antidoom's
/// offline training pipeline).  Sample every 16 chars with a 16-byte
/// fingerprint, min 2 reps, min 16 bytes total.  This catches short
/// repetition loops ("the seat of the government, " × 2) that would
/// otherwise waste many slow tokens before being caught.
pub fn find_inner_repetition(text: &str) -> Option<RepeatHit> {
    const MIN_REPEATS: usize = 2;
    const MAX_PERIOD: usize = 1024;
    const MIN_PERIOD: usize = 1;
    const MIN_TOTAL_REPEATED: usize = 16;
    const SAMPLE_LEN: usize = 16;
    const SAMPLE_INTERVAL: usize = 16;

    if text.len() < MIN_TOTAL_REPEATED {
        return None;
    }
    let n = text.len();
    let mut pos = 0;
    while pos + SAMPLE_LEN < n {
        // Ensure pos is on a char boundary
        while pos < n && !text.is_char_boundary(pos) {
            pos += 1;
        }
        if pos + SAMPLE_LEN >= n {
            break;
        }

        // Clamp end to char boundary to avoid panicking on multi-byte
        // BPE tokens (e.g. Ġ = U+0120 is 2 bytes).
        let mut end = pos + SAMPLE_LEN;
        while end < n && !text.is_char_boundary(end) {
            end += 1;
        }
        if end >= n {
            break;
        }
        let fingerprint = &text[pos..end];

        // Ensure the search start is on a char boundary
        let search_start = pos + SAMPLE_LEN;
        let search_start = {
            let mut s = search_start;
            while s < n && !text.is_char_boundary(s) {
                s += 1;
            }
            s
        };
        if search_start >= n { pos += SAMPLE_LEN.max(1); continue; }

        // Forward search
        if let Some(other_pos) = text[search_start..].find(fingerprint) {
            let absolute_other = search_start + other_pos;
            let candidate_period = absolute_other - pos;
            if (MIN_PERIOD..=MAX_PERIOD).contains(&candidate_period) {
                if let Some(hit) =
                    verify_repetition_at(text, pos, candidate_period, MIN_REPEATS, MIN_TOTAL_REPEATED)
                {
                    return Some(hit);
                }
            }
        }

        // Backward search
        if pos > 0 && text.is_char_boundary(pos) {
            if let Some(other_pos) = text[..pos].rfind(fingerprint) {
                let candidate_period = pos - other_pos;
                if (MIN_PERIOD..=MAX_PERIOD).contains(&candidate_period) {
                    if let Some(hit) = verify_repetition_at(
                        text,
                        other_pos,
                        candidate_period,
                        MIN_REPEATS,
                        MIN_TOTAL_REPEATED,
                    ) {
                        return Some(hit);
                    }
                }
            }
        }

        // Advance to next sample position on a char boundary
        let mut next_pos = pos + SAMPLE_INTERVAL;
        while next_pos < n && !text.is_char_boundary(next_pos) {
            next_pos += 1;
        }
        pos = next_pos;
    }

    None
}

/// Token-id n-gram repetition detector.
///
/// Scans the last `token_ids` for any k-gram (k=2..6) that has appeared
/// a sufficient number of times in a RECENT window.  This catches doom
/// loops where:
///   (a) the byte-level `find_inner_repetition` misses because the
///       repeated segments don't align at byte-exact fingerprint samples;
///   (b) the loop doesn't repeat CONTIGUOUSLY at the tail (the cycle is
///       disrupted by single-token variation but still repeating overall).
///
/// Returns a `RepeatHit` describing the most frequent k-gram in the
/// recent window, when it has appeared at least 3 times.  `period` is set
/// to the gap to the most-recent occurrence so downstream code can compute
/// the continuation token (the first id of that k-gram).
pub fn find_token_ngram_loop(token_ids: &[usize]) -> Option<RepeatHit> {
    let n = token_ids.len();
    if n < 8 {
        return None;
    }

    // Scan the recent window — last 48 tokens by default.  Lends the
    // detector to short-range repetition while ignoring coincidental
    // earlier echoes of the same k-gram from a paragraph back.
    let window_start = n.saturating_sub(48);

    // Try k=6 (longer), k=4, k=3, k=2 in order.  Pick the LONGEST k with a
    // sufficient repetition count in the window; longer k means the
    // intervention is more targeted (suppress fewer wrong tokens).
    for &k in &[6usize, 5, 4, 3, 2] {
        if token_ids.len() < window_start + k {
            continue;
        }
        // Frequency count of k-grams in the window.
        let mut counts: std::collections::HashMap<&[usize], (usize, usize)> =
            std::collections::HashMap::new();
        // Don't include the last k tokens because we want the gram to
        // APPEAR sufficiently close to the tail but not REQUIRE the tail
        // itself to be the start of a cycle.
        for i in window_start..(n - k + 1) {
            let g = &token_ids[i..i + k];
            let entry = counts.entry(g).or_insert((0, i));
            entry.0 += 1;
            entry.1 = i; // record the LAST occurrence start
        }

        // Find the k-gram with the highest count whose last start is
        // within `k` tokens of the current tail (so the cycle is recent).
        let mut best: Option<(&[usize], usize, usize)> = None;
        for (gram, (count, last_start)) in &counts {
            if *count >= 3 && n - *last_start <= 2 * k {
                if best.is_none() || *count > best.unwrap().1 {
                    best = Some((gram, *count, *last_start));
                }
            }
        }

        if let Some((gram, count, last_start)) = best {
            // The next expected continuation token is the FIRST id of the
            // k-gram if the last_start is at the very tail (i.e. the most
            // recent token was the LAST id of gram), otherwise it's the
            // token AT position last_start + k (which continues the gram).
            // We set period in TOKENS = k (semantically: a k-token cycle).
            let period = k;
            return Some(RepeatHit {
                start: last_start,
                end: last_start + k,
                period,
                repeats: count,
                snippet: format!("<{}-token gram, {} reps>", k, count),
            });
        }
    }

    None
}

/// Is the anti-doom runtime gate enabled?
///
/// Defaults ON as of 2026-07-25: temp=0.7 verification on Ministral-3B
/// showed divergent non-loop output (commit aaec49d + 0b1ec36
/// measurement).  Opt-out via `LEAFCUTTER_ANTIDOOM=0` or `=false`.
pub fn is_enabled() -> bool {
    env::var("LEAFCUTTER_ANTIDOOM")
        .map(|v| v != "0" && v != "false")
        .unwrap_or(true)
}

/// Inner state for the anti-doom guard during one generation.
///
/// Holds the decoded text accumulated across the decode loop so the
/// repetition detector can see history, plus a per-step working buffer.
pub struct AntiDoomState {
    /// Decoded text appended so far (post-prompt).
    generated_text: String,
    /// Token boundary byte offsets — `decoded_lens[i]` is the byte length
    /// of token `i` after decoding.  Used to map char positions back to
    /// token ids (when needed for chosen/rejected pair extraction).
    decoded_lens: Vec<usize>,
    /// Token ids in decode order (post-prompt).
    token_ids: Vec<usize>,
    /// Whether we have already intervened once this generation.  We back
    /// off after one intervention per loop to avoid overcorrecting and
    /// forcing weird tokens.
    last_intervention_step: isize,
    /// Count of interventions this generation.
    interventions: usize,
    /// Cumulative time spent in detection (for logging).
    detection_ns: u128,
    /// Step counter (incremented per sampled token).
    step: usize,
}

impl AntiDoomState {
    pub fn new() -> Self {
        Self {
            generated_text: String::new(),
            decoded_lens: Vec::new(),
            token_ids: Vec::new(),
            last_intervention_step: -10,
            interventions: 0,
            detection_ns: 0,
            step: 0,
        }
    }

    /// Record a sampled token id and its decoded text.  Must be called
    /// AFTER the sampler picks the token.
    pub fn record(&mut self, token_id: usize, decoded: &str) {
        self.token_ids.push(token_id);
        self.decoded_lens.push(decoded.len());
        self.generated_text.push_str(decoded);
        self.step += 1;
    }

    /// Inspect the generated text for a doom loop.  If found, return a
    /// set of token ids whose decoded text **starts with the loop's
    /// next continuation prefix** — those should be suppressed in the
    /// NEXT sampler pass.
    ///
    /// `vocab` should be the GGUF tokenizer's per-token surface strings
    /// (so we can match prefixes).  The prefix length is at most 4 bytes
    /// to keep the suppression tight (we don't want to nuke a whole
    /// substring for all tokens).
    pub fn detect(&mut self, vocab: &[String]) -> Option<DoomIntervention> {
        let t0 = Instant::now();
        // Two-pass detection: try the byte-level anti-doom detector first;
        // if it finds nothing, try the token-id n-gram detector which
        // catches loops that the byte fingerprint misses.
        let byte_hit = find_inner_repetition(&self.generated_text);
        let token_hit = find_token_ngram_loop(&self.token_ids);
        self.detection_ns += t0.elapsed().as_nanos();

        // Prefer the byte-level hit — it gives a real text prefix we can
        // match in the vocab.  Fall back to the token-ngram hit if it
        // exists and the byte-level didn't fire.
        let is_token_hit = byte_hit.is_none() && token_hit.is_some();
        let hit = byte_hit.or(token_hit)?;
        // Only intervene once per ~16 tokens — gives the sampler room to
        // escape via natural noise after we punish the loop.
        if (self.step as isize) - self.last_intervention_step < 16 {
            return None;
        }

        // ── Token-ngram hit path: suppress the exact cycle's next token ids ──
        // The hit.start/end/period are all in TOKEN coords here.  We look
        // up the actual token ids that repeat in the cycle (the pattern is
        // self.token_ids[end-k..end]) and the next expected continuation is
        // the very first token of that pattern.
        if is_token_hit {
            let n = self.token_ids.len();
            let k = hit.period; // period in tokens
            if k == 0 || k > n || n < k {
                return None;
            }
            let pattern_ids = &self.token_ids[n - k..n];
            // The cycle is at the tail, so the next continuation is the
            // FIRST token of the k-gram pattern (since we've already
            // emitted `pattern_ids[k-1]` at the most recent step).
            let continuation_token = pattern_ids[0];
            // Also suppress any token whose decoded surface matches the
            // continuation's surface, so we catch morphological variants.
            let continuation_surface: String = if continuation_token < vocab.len() {
                vocab[continuation_token].clone()
            } else { String::new() };

            let mut suppress_ids: Vec<usize> = vec![continuation_token];
            // Add any token ids whose decoded surface starts with the same
            // first 4 bytes as continuation_surface.
            if !continuation_surface.is_empty() {
                let prefix_len = continuation_surface.len().min(4);
                let prefix = &continuation_surface[..prefix_len];
                for (i, surface) in vocab.iter().enumerate() {
                    if i == continuation_token { continue; }
                    if surface.starts_with(prefix) {
                        suppress_ids.push(i);
                        if suppress_ids.len() >= 64 { break; }
                    }
                }
            }

            self.last_intervention_step = self.step as isize;
            self.interventions += 1;
            return Some(DoomIntervention {
                hit,
                suppress_ids,
                next_prefix: continuation_surface,
            });
        }

        // ── Byte-level hit path: vocab prefix matching ──
        // The loop's repeated at the END of `generated_text` so the next
        // continuation is the next `pattern_prefix_len` bytes of the
        // period starting from (end - start) % period.
        let period = hit.period;
        let pattern_start = hit.start;
        if pattern_start + period > self.generated_text.len() {
            return None;
        }
        let pattern = &self.generated_text[pattern_start..pattern_start + period];

        // Compute the next char prefix that would continue the cycle.
        // end_pos is where forward matching stopped; the next expected
        // continuation is pattern[(end - start) % period .. period].
        let bytes_consumed_in_cycle = hit.end - pattern_start;
        let next_offset = bytes_consumed_in_cycle % period;
        let next_prefix_len = period.min(4).max(1); // 1..4 bytes
        let next_prefix_len = next_prefix_len.min(period - next_offset);
        if next_prefix_len == 0 {
            return None;
        }
        let next_prefix = &pattern[next_offset..next_offset + next_prefix_len];

        // Find all token ids whose decoded surface starts with `next_prefix`.
        // Cap the suppression set at 64 ids to keep the operation bounded.
        let mut suppress_ids: Vec<usize> = Vec::new();
        for (i, surface) in vocab.iter().enumerate() {
            if surface.starts_with(next_prefix) {
                suppress_ids.push(i);
                if suppress_ids.len() >= 64 {
                    break;
                }
            }
        }

        if suppress_ids.is_empty() {
            return None;
        }

        self.last_intervention_step = self.step as isize;
        self.interventions += 1;
        Some(DoomIntervention {
            hit,
            suppress_ids,
            next_prefix: next_prefix.to_string(),
        })
    }

    /// Total number of anti-doom interventions performed so far.
    pub fn interventions(&self) -> usize {
        self.interventions
    }

    /// Cumulative detection time in nanoseconds (for logging).
    pub fn detection_time_ns(&self) -> u128 {
        self.detection_ns
    }
}

/// A doom-loop intervention — tokens to suppress in the upcoming sampler
/// pass.
#[derive(Debug)]
pub struct DoomIntervention {
    /// The detected repetition.
    pub hit: RepeatHit,
    /// Token ids whose surface starts with the next cycle continuation
    /// prefix.  These should have their logit set to -inf (or some very
    /// negative value) before sampling.
    pub suppress_ids: Vec<usize>,
    /// Decoded prefix that the model was about to continue with.
    pub next_prefix: String,
}

/// Suppress the given token ids in the logits vector by setting their
/// logit to `f32::NEG_INFINITY` (effectively removing them from the
/// sampling pool).  In-place.
pub fn suppress_in_logits(logits: &mut [f32], ids: &[usize]) {
    for &id in ids {
        if id < logits.len() {
            logits[id] = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_repetition_short_text() {
        assert!(find_inner_repetition("hello world").is_none());
    }

    #[test]
    fn test_simple_repetition() {
        // 8x "the quick " is 80 bytes — well above the 32 threshold
        let text = "the quick brown fox jumps. ".repeat(8);
        let hit = find_inner_repetition(&text).expect("should detect");
        assert!(hit.repeats >= 4);
        assert!(hit.period > 0);
        assert!(hit.snippet.len() > 0);
    }

    #[test]
    fn test_detect_interventions_state() {
        let mut state = AntiDoomState::new();
        // simulate generated text where token 0 is "ab" repeated to a loop
        for _ in 0..10 {
            state.record(0, "abcdefghij");
        }
        let vocab: Vec<String> = (0..10).map(|i| i.to_string()).collect();
        let res = state.detect(&vocab);
        // Some pattern detected (abcdefghij x10) but doesn't have a per-step continuation
        // that appears in vocab — should be None or Some depending on prefix mapping.
        // We just check the detector doesn't panic here.
        let _ = res;
    }

    #[test]
    fn test_suppress_in_logits() {
        let mut logits = vec![1.0_f32; 10];
        suppress_in_logits(&mut logits, &[2, 5, 8]);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[5], f32::NEG_INFINITY);
        assert_eq!(logits[8], f32::NEG_INFINITY);
        assert_eq!(logits[0], 1.0);
    }
}
