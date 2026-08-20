//! Process-wide initialization helpers (thread pool sizing, etc.).
//!
//! LeafcutterLLM uses rayon's global thread pool for `into_par_iter()` calls
//! in the dequant kernels, attention, and LM-head projection. Without a cap,
//! that pool defaults to **all logical CPUs** (16 on a Ryzen 5800HS), which
//! overwhelms the system when a single inference workload is the only thing
//! happening — observed peaks up to 1585% CPU on a 5-token prefill+decode run.
//!
//! The fix is to size the pool explicitly to something the kernel-overhead
//! curve flattens out at (typically `physical_cores - 1`). See
//! `scripts/bench_one.sh` for the empirical study.
//!
//! Usage:
//! ```rust
//! leafcutter::init::configure_thread_pool(Some(7));   // hard cap at 7
//! leafcutter::init::configure_thread_pool(None);      // → default (= available parallelism - 1)
//! ```

/// Cap rayon's global pool to `n` worker threads (or auto-pick if `None`).
///
/// Must be called **before** any `par_iter()` is invoked — once rayon has
/// spawned its default pool it cannot be shrunk without calling
/// `shutdown_global` (which we don't expose here).
///
/// Returns whatever rayon reports; ignores errors (rayon returns a
/// string error when called twice).
pub fn configure_thread_pool(threads: Option<usize>) -> Result<usize, String> {
    let n = threads.unwrap_or_else(default_thread_count);
    // Pass RAYON_NUM_THREADS *before* the first par_iter; this is the
    // simplest hook. Process-level env override also works if user sets it.
    std::env::set_var("RAYON_NUM_THREADS", n.to_string());
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global(); // idempotent — second call fails harmlessly
    Ok(n)
}

/// Auto-pick a sensible default.
///
/// Empirical rule (see `scripts/bench_run.sh`):
///   - On a 16-vCPU Ryzen 5800HS, T7 (physical cores - 1) halved CPU
///     from ~1586% peak to ~706% with no measurable throughput cost.
///   - Below T4 throughput begins to drop noticeably; above T8 we pick
///     up only marginally faster matmul at disproportionately high CPU.
///
/// We approximate "physical cores" by guessing that SMT doublers in
/// /proc/cpuinfo (we read it if available) collapse logical CPUs into
/// physical pairs. When that file isn't available (Windows/Mac), we
/// fall back to `available_parallelism() / 2`.
pub fn default_thread_count() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|u| u.get())
        .unwrap_or(4);

    // On aarch64 / ARM (e.g. CIX Sky1 / RK3588), cores are physical (no SMT hyperthreading).
    // Use logical - 1 (or all logical cores if <= 4) for maximum parallel GEMV performance.
    if cfg!(target_arch = "aarch64") {
        return (logical.saturating_sub(1)).max(2);
    }

    // Try /proc/cpuinfo for x86 physical core detection.
    if let Ok(s) = std::fs::read_to_string("/proc/cpuinfo") {
        let mut phys = 0usize;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("cpu cores\t: ") {
                if let Ok(v) = rest.trim().parse::<usize>() {
                    phys = v;
                }
            }
        }
        if phys > 0 {
            return (phys.saturating_sub(1)).max(2);
        }
    }
    // Fallback for x86 SMT
    (logical / 2).max(2)
}

/// Read the effective rayon thread cap for this process.
/// Priority: explicit `Some(n)` parameter > RAYON_NUM_THREADS > available - 1.
pub fn effective_thread_count() -> usize {
    if let Ok(s) = std::env::var("RAYON_NUM_THREADS") {
        if let Ok(n) = s.parse() {
            return n;
        }
    }
    if let Ok(s) = std::env::var("LEAFCUTTER_THREADS") {
        if let Ok(n) = s.parse() {
            return n;
        }
    }
    default_thread_count()
}
