//! Background CPU/thermal/load monitor for LeafcutterLLM.
//!
//! User-facing safety layer.  We do NOT throttle, block, or otherwise
//! interfere with model execution.  We only **observe** and **warn** the
//! user when we see something they should know about.
//!
//! Killed eagerly on `/bye` (calls `stop()`) or when the host process
//! exits.  No dependencies on the inference engine.
//!
//! On Linux we sample:
//!   * CPU temperature from `/sys/class/thermal/thermal_zone*/temp` (°C)
//!   * Resident set size from `/proc/self/status` VmRSS (KB)
//!   * Per-second CPU usage from `/proc/stat` deltas
//!
//! Thresholds (sane defaults for a laptop, conservative — not panic-inducing):
//!   * TEMP_WARN=85°C  — Intel/AMD throttle threshold typical
//!   * TEMP_CRIT=95°C  — most CPUs throttle themselves here
//!   * RSS_WARN=100GB  — heavy but OK for 70B class
//!
//! Opt out entirely: `LEAFCUTTER_CPU_MONITOR=0`
//! Opt in (was the default in earlier versions; disabled to keep the REPL
//! output stream clean — safety warnings used to interleave with streamed
//! model text).  The monitor is still wired in for ops use; set the env var
//! to `1` to see the temperature/RSS warnings on stderr.

use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

static MONITORING: AtomicBool = AtomicBool::new(false);

/// Start the background monitor.  Idempotent.  No-op unless
/// `LEAFCUTTER_CPU_MONITOR=1` (default OFF so REPL output stays clean).
pub fn start() {
    if MONITORING.swap(true, Ordering::SeqCst) {
        return; // already running
    }
    if std::env::var("LEAFCUTTER_CPU_MONITOR")
        .map(|v| v != "1" && v != "true")
        .unwrap_or(true)
    {
        return;
    }
    thread::Builder::new()
        .name("leafcutter-cpu-monitor".to_string())
        .spawn(|| {
            run_loop();
        })
        .ok();
}

/// Stop the monitor (called on `/bye`).
pub fn stop() {
    MONITORING.store(false, Ordering::SeqCst);
}

fn run_loop() {
    let interval = Duration::from_secs(30);
    let mut last_warn_temp = 0u64;
    let mut last_warn_rss = 0u64;
    let mut prev_idle: Option<u64> = None;
    let mut prev_total: Option<u64> = None;

    while MONITORING.load(Ordering::SeqCst) {
        let start = Instant::now();

        let temp = read_cpu_temp();
        let rss_mb = read_rss_mb();
        let (cpu_pct, new_idle, new_total) = read_cpu_usage(prev_idle, prev_total);
        prev_idle = Some(new_idle);
        prev_total = Some(new_total);

        // Temperature warnings (sticky — only fire once per threshold)
        if let Some(t) = temp {
            if t >= 95 && last_warn_temp < 95 {
                eprintln!(
                    "\n[MONITOR] CPU temp °C= {}  — HOT.  Your CPU is likely throttling itself. \
                     Consider reducing layer count or pausing for a minute. \
                     Leafcutter is NOT throttling your CPU; this is purely informational.",
                    t
                );
                last_warn_temp = 95;
            } else if t >= 85 && last_warn_temp < 85 {
                eprintln!(
                    "\n[MONITOR] CPU temp °C= {}  — warm.  Modern CPUs are fine up to 100°C \
                     but persistent operation here may trigger thermal throttling. \
                     Leafcutter remains as fast as your hardware allows.",
                    t
                );
                last_warn_temp = 85;
            }
        }

        // RAM warning
        let rss_mb_val = rss_mb.unwrap_or(0);
        if rss_mb_val >= 100 * 1024 && last_warn_rss < 100 * 1024 {
            eprintln!(
                "\n[MONITOR] RSS: {} MB.  Heavy model loaded. \
                 `/bye` will release memory.  No action needed unless you \
                 want to free RAM for other apps.",
                rss_mb_val / 1024
            );
            last_warn_rss = 100 * 1024;
        }

        // Low-rate info log on first iteration only
        if last_warn_temp < 85 && last_warn_rss < 100 * 1024 {
            // Could emit an info log here; for now we stay silent.
        }

        let elapsed = start.elapsed();
        if elapsed < interval {
            thread::sleep(interval - elapsed);
        }
    }
}

/// Read the CPU temperature (°C).  We use thermal_zone0 by default; if it
/// reports 'x86_pkg' that's the package temp.  Otherwise we pick the first
/// zone that isn't empty.  Returns None if no zone works.
fn read_cpu_temp() -> Option<u64> {
    let entries = std::fs::read_dir("/sys/class/thermal").ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let n = name.to_string_lossy();
        if !n.starts_with("thermal_zone") {
            continue;
        }
        let path = entry.path();
        // Prefer zones that are actually CPU-related
        let ztype = std::fs::read_to_string(path.join("type"))
            .ok()?
            .trim()
            .to_lowercase();
        let ztemp: String = std::fs::read_to_string(path.join("temp")).ok()?;
        let Ok(v) = ztemp.trim().parse::<i64>() else { continue };
        // Many laptops report CPU at zone0 or 'x86_pkg'
        if ztype.contains("cpu")
            || ztype.contains("x86")
            || ztype.contains("pkg")
            || ztype.contains("acpi")
        {
            // Linux thermal zones report in millidegrees Celsius (e.g. 71000 = 71.0°C).
            return Some((v / 1000) as u64);
        }
    }
    None
}

/// Read resident set size in MB from /proc/self/status VmRSS.
fn read_rss_mb() -> Option<u64> {
    let mut f = std::fs::File::open("/proc/self/status").ok()?;
    let mut s = String::new();
    f.read_to_string(&mut s).ok()?;
    for line in s.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // ["VmRSS:", "1234", "kB"]  -> 1234 KB = 1.234 MB
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<u64>() {
                    return Some(kb / 1024);
                }
            }
        }
    }
    None
}

/// Read CPU usage percent (over the whole /proc/stat) by deltaing idle + total
/// between samples.  Returns `(percent, new_idle, new_total)`.
fn read_cpu_usage(prev_idle: Option<u64>, prev_total: Option<u64>) -> (f32, u64, u64) {
    let mut s = String::new();
    let Ok(mut f) = std::fs::File::open("/proc/stat") else {
        return (0.0, 0, 0);
    };
    let _ = f.read_to_string(&mut s);
    // Parse the first non-empty CPU line: "cpu  user nice system idle iowait..."
    let mut idle: u64 = 0;
    let mut total: u64 = 0;
    for line in s.lines() {
        if line.starts_with("cpu ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            for (i, p) in parts.iter().enumerate().skip(1) {
                let v: u64 = p.parse().unwrap_or(0);
                total += v;
                // idle = column 4, iowait = column 5 (Linux convention)
                if i == 4 || i == 5 {
                    idle += v;
                }
            }
            break;
        }
    }
    let pct = match (prev_idle, prev_total) {
        (Some(pi), Some(pt)) if total > pt => {
            let d_total = total - pt;
            let d_idle = idle.saturating_sub(pi);
            (1.0 - (d_idle as f32 / d_total as f32).clamp(0.0, 1.0)) * 100.0
        }
        _ => 0.0,
    };
    (pct, idle, total)
}
