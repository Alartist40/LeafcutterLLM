//! Determinism contract (from the kimi-k3-in-c analysis).
//!
//! kimi-k3-in-c guarantees bit-identical output across machines by using
//! `-ffp-contract=off` (no FMA contraction in scalar code) and f64
//! accumulators for every reduction. Leafcutter's kernels use AVX2 FMA chains,
//! dual-accumulator splits, and an i32 integer-dot fast path — all of which
//! change rounding order and produce results that vary by build and machine.
//!
//! This module exposes a single env-controlled switch,
//! `LEAFCUTTER_DETERMINISTIC=1`, that kernels check before picking a fast
//! path. In deterministic mode every dot product falls back to a serial,
//! f64-accumulated reference reduction and the Q8_K integer-dot path is
//! disabled, so logits are reproducible run-to-run.

/// Whether deterministic mode is enabled (`LEAFCUTTER_DETERMINISTIC=1`).
#[inline]
pub fn enabled() -> bool {
    std::env::var("LEAFCUTTER_DETERMINISTIC")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Whether the current CPU supports AVX2 + FMA.
///
/// Safe to call on ANY architecture (x86, aarch64, riscv, …). On non-x86
/// targets this always returns `false`, so kernels can gate their SIMD fast
/// paths with a single portable check instead of sprinkling
/// `std::arch::is_x86_feature_detected!` behind `cfg!()` guards (the macro
/// itself cannot even be *parsed* on non-x86 targets).
#[inline]
pub fn cpu_has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Reference dot product: serial, f64-accumulated, no FMA contraction.
/// Bit-identical for a given input on every machine/thread count.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        sum += a[i] as f64 * b[i] as f64;
    }
    sum as f32
}

/// Reference f32 reduction (e.g. residual accumulation).
#[inline]
pub fn reduce(values: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    for v in values {
        sum += *v as f64;
    }
    sum as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_dot_matches_scalar_sum() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut expect = 0.0f64;
        for i in 0..4 {
            expect += a[i] as f64 * b[i] as f64;
        }
        assert_eq!(dot_product(&a, &b), expect as f32);
    }

    #[test]
    fn deterministic_dot_reproducible() {
        let a: Vec<f32> = (0..257).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..257).map(|i| (i as f32) * 0.3 + 0.5).collect();
        assert_eq!(dot_product(&a, &b), dot_product(&a, &b));
    }

    #[test]
    fn reduce_accumulates_in_order() {
        let vals = [1.0, 2.0, 3.0];
        assert_eq!(reduce(&vals), 6.0);
    }
}