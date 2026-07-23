//! LfruCache — LFU + LRU hybrid (frequency-primary, recency-secondary)
//! with 25%+4-frequency hysteresis. Direct port of Colibri's `tier.h` LFRU.
//!
//! Why LFRU, not LRU or FIFO:
//! - **FIFO**: random eviction. Per-workload severe thrash on non-sequential reuse.
//! - **LRU**: exempts very-frequent items from eviction only by being accessed
//!   recently. A burst-of-cold traffic (e.g., model parallelism stage boundaries)
//!   evicts hot items even though their long-term frequency is high.
//! - **LFRU**: frequency dominates. Frequency×256 + (255 - age) means a merely
//!   recent item can only displace a truly hot one if its frequency is within
//!   the 25%+4 margin of the cold resident. Prevents ping-pong eviction.
//!
//! Heat is a u32 count that grows on each access and halves on `decay()`.
//! The caller decides when to call `decay()`; for a model with hot/cold
//! routing patterns we decay every N forward passes.

use std::collections::HashMap;
use crate::model::tensor::Tensor;

/// LfruCache — fixed-capacity cache keyed by layer index.
///
/// Stores `HashMap<String, Tensor>` payloads (the layer weights) keyed by
/// the layer's index in the model. Tracks frequency and recency per slot.
#[derive(Debug)]
pub struct LfruCache {
    /// residual slots, keyed implicitly by `idx`
    slots: HashMap<usize, HashMap<String, Tensor>>,
    /// per-layer insertion order (FIFO tiebreaker for ordering back into slots)
    slot_order: Vec<usize>,
    /// per-layer total access count
    heat: HashMap<usize, u32>,
    /// per-layer last-access tick
    last_access: HashMap<usize, u32>,
    /// global clock; incremented on each `touch`
    clock: u32,
    /// capacity
    max_slots: usize,
    /// hit / miss / evict counters (test-instrumented)
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl LfruCache {
    pub fn new(max_slots: usize) -> Self {
        Self {
            slots: HashMap::with_capacity(max_slots),
            slot_order: Vec::with_capacity(max_slots),
            heat: HashMap::new(),
            last_access: HashMap::new(),
            clock: 0,
            max_slots,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn max_slots(&self) -> usize {
        self.max_slots
    }

    /// LFRU score, port of `tier_lfru_score` from Colibri's tier.h.
    /// Frequency dominates: heat << 8 gives 256-unit weight per access.
    /// Recency contributes at most 255 units, breaking close calls.
    fn score(&self, idx: usize) -> u64 {
        let heat = *self.heat.get(&idx).unwrap_or(&0);
        let last = *self.last_access.get(&idx).unwrap_or(&0);
        let age = self.clock.saturating_sub(last);
        let recent = if age < 255 { 255 - age } else { 0 };
        ((heat as u64) << 8) | (recent as u64)
    }

    /// Pick the index of the slot to evict under LFRU policy.
    /// Returns the slot key (= layer idx) with the lowest score.
    fn pick_eviction(&self) -> Option<usize> {
        self.slot_order
            .iter()
            .copied()
            .min_by_key(|&i| self.score(i))
    }

    /// Pick the candidate (non-resident idx) most worth promoting into a slot.
    /// Combined with `pick_eviction`, lets us check the 25%+4 hysteresis
    /// before evicting cold for a hot candidate.
    fn pick_candidate(&self) -> Option<usize> {
        // Iterate tracked layers (could be residents or not). Non-residents
        // are higher-priority — those are newly-seen layers whose heat is
        // only acquired via misses. If we have a miss wave we want them.
        // For now, scan `heat.keys()`. (Rooms empty slots aren't tracked.)
        self.heat
            .keys()
            .copied()
            .filter(|i| !self.slots.contains_key(i))
            .max_by_key(|&i| self.score(i))
    }

    /// Time a call: bumps heat, updates last_access, advances clock.
    /// Always called on `get` AND `put` (so miss-then-fill counts as access).
    fn touch(&mut self, idx: usize) {
        self.clock = self.clock.wrapping_add(1);
        *self.heat.entry(idx).or_insert(0) += 1;
        self.last_access.insert(idx, self.clock);
    }

    /// Halve all heat counters. Periodic decay so old hot items don't stay
    /// hot forever.
    pub fn decay(&mut self) {
        for v in self.heat.values_mut() {
            *v >>= 1;
        }
    }

    pub fn get(&mut self, idx: usize) -> Option<HashMap<String, Tensor>> {
        if self.max_slots == 0 {
            // Even for a 0-slot cache, we still want frequency tracking
            // (the LfruCache is informed by access patterns even if it
            // doesn't store the data).
            self.touch(idx);
            self.misses += 1;
            return None;
        }

        let hit = self.slots.contains_key(&idx);
        self.touch(idx);
        if hit {
            self.hits += 1;
            self.slots.get(&idx).cloned()
        } else {
            self.misses += 1;
            None
        }
    }

    pub fn put(&mut self, idx: usize, weights: HashMap<String, Tensor>) {
        if self.max_slots == 0 {
            return;
        }
        self.touch(idx);

        // Already present — update payload, no eviction needed
        if self.slots.contains_key(&idx) {
            self.slots.insert(idx, weights);
            return;
        }

        // Slot available — insert directly
        if self.slots.len() < self.max_slots {
            self.slots.insert(idx, weights);
            self.slot_order.push(idx);
            return;
        }

        // At capacity — apply hysteresis: only evict if incoming candidate is
        // at least 25% + (4<<8) score units hotter than the coldest resident.
        let evictee = self.pick_eviction();
        let candidate_score = self.score(idx);
        if let Some(e) = evictee {
            let cold_score = self.score(e);
            // 25% margin: incoming_score > cold_score + (cold_score >> 2) + 4<<8
            // (4<<8) is in score units = 4 frequency-units.
            let margin = (cold_score >> 2) + (4u64 << 8);
            if candidate_score <= cold_score + margin {
                // incoming isn't worth displacing cold — drop incoming.
                // (This is what Colibri does; preserves hot resident layers.)
                return;
            }
        }

        // Replace the coldest resident with the incoming layer.
        if let Some(e) = evictee {
            self.slots.remove(&e);
            self.slot_order.retain(|&i| i != e);
            self.evictions += 1;
        }
        self.slots.insert(idx, weights);
        self.slot_order.push(idx);
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.slot_order.clear();
        // note: keep heat and last_access across clear() so the system
        // "remembers" the workload between context switches.
    }

    /// Snapshot of stats (test / benchmark helper).
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            resident: self.slots.len(),
            cap: self.max_slots,
            clock: self.clock,
        }
    }
}

/// Snapshot returned by `LfruCache::stats()` — useful for the test harness
/// and for `"LEAFCUTTER_PROFILE=1"` run summaries.
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub resident: usize,
    pub cap: usize,
    pub clock: u32,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(idx: usize) -> HashMap<String, Tensor> {
        let mut m = HashMap::new();
        m.insert(format!("x{}", idx), Tensor::zeros(vec![1]));
        m
    }

    #[test]
    fn test_lfru_zero_capacity_is_pure_miss() {
        let mut c = LfruCache::new(0);
        assert!(c.get(0).is_none()); // caps at 0 → always miss
        assert!(c.get(0).is_none());
        assert_eq!(c.stats().hits, 0);
        assert_eq!(c.stats().misses, 2);
        c.put(0, tiny(0));
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn test_lfru_basic_hit_miss() {
        let mut c = LfruCache::new(2);
        assert!(c.get(0).is_none()); // miss
        c.put(0, tiny(0));
        assert!(c.get(0).is_some()); // hit
        assert!(c.get(1).is_none()); // miss
        c.put(1, tiny(1));
        assert_eq!(c.stats().hits, 1);
        assert_eq!(c.stats().misses, 2);
    }

    #[test]
    fn test_lfru_eviction_picks_cold() {
        // Two slots. Touch 0 heavily (heat=10), touch 1 once (heat=1).
        // 2 is brand new. Eviction should drop 1 (coldest), keep 0.
        let mut c = LfruCache::new(2);
        for _ in 0..10 { c.touch(0); }
        c.touch(1);
        c.put(0, tiny(0));
        c.put(1, tiny(1));
        // Force 1's heat to be lower than 0's:
        // Now overwrite with decay so 0 stays hotter, then put 2.
        c.decay(); // heat halved: 5
        c.decay(); // heat halved: 2
        c.decay(); // heat halved: 1
        c.decay(); // heat halved: 0
        // Now 0 has heat=0 effective, 1 has heat=0. They tie on freq.
        // Recency: 1 set last so it has the higher recent. So 0 is colder.
        // Reset and try again with a clean test:
        c.clear();
        c.slots.clear();
        c.slot_order.clear();
        c.heat.clear();
        c.last_access.clear();
        c.clock = 0;
        for _ in 0..100 { c.touch(0); }
        c.touch(1);
        c.put(0, tiny(0)); // resident
        c.put(1, tiny(1)); // resident
        // Now both residents. Touch 2 a lot.
        for _ in 0..1_000 { c.touch(2); }
        c.put(2, tiny(2));
        // 2 has heat=1000+, 0 has heat=100, 1 has heat=1.
        // After hysteresis check: 2's score should beat 1's by far.
        assert!(c.slots.contains_key(&2));
        assert!(c.slots.contains_key(&0), "0 should be retained (hot)");
        assert!(!c.slots.contains_key(&1), "1 should have been evicted (coldest)");
    }

    #[test]
    fn test_lfru_hysteresis_protects_recent_cold_transient() {
        // Scenario: a recently-cold layer (heat=1) keeps getting picked up.
        // Hysteresis should prevent it from displacing a high-heat resident.
        let mut c = LfruCache::new(2);
        // Saturate residents with hot layers
        for _ in 0..1_000 { c.touch(100); }
        for _ in 0..1_000 { c.touch(200); }
        c.put(100, tiny(100));
        c.put(200, tiny(200));
        // Now try to push a fresh layer 300 with only 1 touch
        c.touch(300);
        c.put(300, tiny(300));
        // 300 has heat=1, resident scores: 100 and 200 both have heat=1000+.
        // Hysteresis check: 300's score < coldest's score + margin → rejected.
        assert!(!c.slots.contains_key(&300),
            "300 must be rejected by hysteresis — incoming is colder than margin allows");
        assert!(c.slots.contains_key(&100));
        assert!(c.slots.contains_key(&200));
    }

    #[test]
    fn test_lfru_decay_drops_old_heat() {
        let mut c = LfruCache::new(2);
        for _ in 0..256 { c.touch(5); }
        // heat=256 score = 256<<8 = 65536
        let s_pre = c.score(5);
        c.decay();
        let s_post = c.score(5);
        assert!(s_post < s_pre, "decay must reduce score");
        // heat should be 128
        let new_heat = c.heat.get(&5).copied().unwrap_or(0);
        assert!(new_heat < 256 && new_heat >= 128);
    }

    #[test]
    fn test_lfru_clock_wraps_at_u32() {
        // Saturating arithmetic must not panic. Wraps at 2^32.
        let mut c = LfruCache::new(1);
        c.clock = u32::MAX;
        c.touch(7);
        assert_eq!(c.clock, 0); // wrapped
        assert!(c.last_access.get(&7).copied().unwrap() == 0);
    }

    #[test]
    fn test_lfru_overwrite_existing() {
        let mut c = LfruCache::new(2);
        c.put(0, tiny(0));
        c.put(0, tiny(0)); // same idx, overwrites payload
        assert_eq!(c.len(), 1);
        assert_eq!(c.stats().evictions, 0, "no eviction on same-idx overwrite");
    }
}
