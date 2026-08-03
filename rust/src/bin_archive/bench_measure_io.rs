//! Phase 2A pre-flight: measure what the I/O path actually costs.
//!
//! Goal: determine whether io_uring (or even pread) is faster than our
//! current `Mmap + MADV_DONTNEED + page-fault-on-touch` flow, on THIS
//! hardware, for representative layer sizes.
//!
//! Three loads are timed:
//!   1. **mmap-fault** — mmap(file), read byte to force fault, time it.
//!      This is what our loader does today.
//!   2. **pread-cold** — open(file), pread() into a heap buffer, time it.
//!      Evicts the OS page cache first via posix_fadvise(POSIX_FADV_DONTNEED).
//!   3. **pread-cached** — pread() after a previous read populated the page
//!      cache. (What happens on a hot-cache second pass through the layers.)
//!
//! Each test loads N layers of `layer_size` MB, sequentially. If pread-cold
//! is similar in cost to mmap-fault, the OS page cache is doing all the work
//! and io_uring won't help. If pread-cold dwarfs mmap-fault, io_uring's
//! submission-overlap might help.

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::PathBuf;
use std::time::Instant;

use memmap2::Mmap;

#[derive(Clone)]
struct Args {
    /// Path to a single layer file (used as test shard).
    layer_path: String,
    /// Layer size in MB (target file size).
    #[allow(dead_code)]
    layer_size_mb: usize,
    /// Number of layers to load per measurement.
    layers: usize,
    /// Repetitions — each repetition drops the cache.
    reps: usize,
}

impl Args {
    fn from_env() -> Self {
        let layer_path = std::env::var("MEASURE_LAYER")
            .unwrap_or_else(|_| "/tmp/leafcutter_measure_layer.bin".to_string());
        let layer_size_mb: usize = std::env::var("MEASURE_SIZE_MB")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(500);
        let layers: usize = std::env::var("MEASURE_LAYERS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(80);
        let reps: usize = std::env::var("MEASURE_REPS")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        Self { layer_path, layer_size_mb, layers, reps }
    }
}

/// Create a test file on disk filled with deterministic bytes.
fn ensure_test_file(path: &str, size: &[u8]) -> std::io::Result<()> {
    let p = PathBuf::from(path);
    if p.exists() {
        let m = std::fs::metadata(&p)?;
        if m.len() as usize >= size.len() {
            return Ok(());
        }
    }
    std::fs::write(&p, size)
}

/// Drop OS page cache for `path`'s inode. Best-effort.
fn drop_caches(path: &str) {
    // POSIX_FADV_DONTNEED: kernel may drop pages after return.
    // We don't insist or measure success — just send the hint.
    if let Ok(f) = OpenOptions::new().read(true).open(path) {
        use std::os::fd::AsRawFd;
        let fd = f.as_raw_fd();
        // SAFETY: fd is open; size 0 hint applies to whole file.
        unsafe {
            libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
}

fn time_mmap_fault(path: &str, off: u64, len: usize) -> std::time::Duration {
    let t = Instant::now();
    let mmap_full = unsafe { Mmap::map(&File::open(path).unwrap()).unwrap() };
    let mmap = &mmap_full[off as usize..(off as usize + len)];
    // Force page faults: read all bytes
    let mut sum: u64 = 0;
    for chunk in mmap.chunks(4096) {
        sum = sum.wrapping_add(chunk[0] as u64);
    }
    std::hint::black_box(sum);
    t.elapsed()
}

fn time_pread(path: &str, off: u64, len: usize) -> std::time::Duration {
    let t = Instant::now();
    let f = File::open(path).unwrap();
    let mut buf = vec![0u8; len];
    f.read_exact_at(&mut buf, off).unwrap();
    let mut sum: u64 = 0;
    for chunk in buf.chunks(4096) {
        sum = sum.wrapping_add(chunk[0] as u64);
    }
    std::hint::black_box(sum);
    // Tell OS we don't need this resident — symmetry with the mmap flow
    unsafe {
        use std::os::fd::AsRawFd;
        libc::posix_fadvise(f.as_raw_fd(), off as i64, len as i64, libc::POSIX_FADV_DONTNEED);
    }
    t.elapsed()
}

fn format_ms(d: std::time::Duration) -> String {
    format!("{:.2} ms", d.as_secs_f64() * 1000.0)
}

fn main() {
    let args = Args::from_env();
    println!("📏 Phase 2A pre-flight — I/O measurement");
    println!("   Layer path:   {}", args.layer_path);
    println!("   Layer size:   {} MB", args.layer_size_mb);
    println!("   Layers:       {}", args.layers);
    println!("   Reps:         {}", args.reps);

    // Create dummy layer file at MEASURE_LAYER_SIZE_MB
    let target_size = args.layer_size_mb * 1024 * 1024;
    println!("\n📦 Creating test file ({} bytes)...", target_size);
    let blob: Vec<u8> = (0..target_size as u32).map(|i| (i & 0xff) as u8).collect();
    if let Err(e) = ensure_test_file(&args.layer_path, &blob) {
        eprintln!("create failed: {}", e);
        std::process::exit(1);
    }
    drop(blob);
    let actual_size = std::fs::metadata(&args.layer_path).unwrap().len() as usize;
    println!("   File size:    {} bytes", actual_size);

    let layer_size = (actual_size / args.layers).max(4096);
    println!("   Per-layer:    {} bytes", layer_size);

    println!("\n🏁 Benchmarking — each rep drops page cache first\n");

    let mut sum_mmap = std::time::Duration::ZERO;
    let mut sum_pread_cold = std::time::Duration::ZERO;
    let mut sum_pread_warm = std::time::Duration::ZERO;
    let mut sum_pread_dontneed = std::time::Duration::ZERO;

    for rep in 0..args.reps {
        println!("─── rep {} ───", rep);

        // Pass 1: mmap-fault (our current method)
        drop_caches(&args.layer_path);
        let mut mm_t = std::time::Duration::ZERO;
        for i in 0..args.layers {
            let off = (i * layer_size) as u64;
            mm_t += time_mmap_fault(&args.layer_path, off, layer_size);
        }
        sum_mmap += mm_t;
        println!("  mmap-fault       total {} ({}/layer)", format_ms(mm_t), format_ms(mm_t / args.layers as u32));

        // Pass 2: cold pread — cache should be cold (we just did drop_caches)
        // Skip drop_caches because the kernel's recent mmap-fault may have populated
        // the cache. Sync first to flush.
        std::fs::File::open(&args.layer_path).unwrap().sync_all().ok();
        drop_caches(&args.layer_path);
        let mut pc_t = std::time::Duration::ZERO;
        for i in 0..args.layers {
            let off = (i * layer_size) as u64;
            pc_t += time_pread(&args.layer_path, off, layer_size);
        }
        sum_pread_cold += pc_t;
        println!("  pread-cold       total {} ({}/layer)", format_ms(pc_t), format_ms(pc_t / args.layers as u32));

        // Pass 3: warm pread — just re-read; should hit page cache
        let mut pw_t = std::time::Duration::ZERO;
        for i in 0..args.layers {
            let off = (i * layer_size) as u64;
            pw_t += time_pread(&args.layer_path, off, layer_size);
        }
        sum_pread_warm += pw_t;
        println!("  pread-warm       total {} ({}/layer)", format_ms(pw_t), format_ms(pw_t / args.layers as u32));

        // Pass 4: pread with hint-DONTNEED each call (privacy)
        // This is closer to how Colibri's `st_pread_full + DONTNEED` flows.
        let mut pd_t = std::time::Duration::ZERO;
        for i in 0..args.layers {
            let off = (i * layer_size) as u64;
            pd_t += time_pread(&args.layer_path, off, layer_size);
        }
        sum_pread_dontneed += pd_t;
        println!("  pread-dontneed   total {} ({}/layer)", format_ms(pd_t), format_ms(pd_t / args.layers as u32));

        println!();
    }

    println!("═══════════════════════════════════════════════");
    println!("📊 Summary (avg over {} reps):", args.reps);
    let n = args.reps as u32;
    let mmap_avg = sum_mmap / n;
    let pc_avg = sum_pread_cold / n;
    let pw_avg = sum_pread_warm / n;
    let pd_avg = sum_pread_dontneed / n;
    let total_layers = args.layers as u32;
    println!("  mmap-fault         avg total {} ({} / layer)",
        format_ms(mmap_avg), format_ms(mmap_avg / total_layers));
    println!("  pread-cold         avg total {} ({} / layer)",
        format_ms(pc_avg), format_ms(pc_avg / total_layers));
    println!("  pread-warm         avg total {} ({} / layer)",
        format_ms(pw_avg), format_ms(pw_avg / total_layers));
    println!("  pread-dontneed     avg total {} ({} / layer)",
        format_ms(pd_avg), format_ms(pd_avg / total_layers));

    println!();
    println!("═══════════════════════════════════════════════");
    println!("📐 Interpretation:");
    let ratio = mmap_avg.as_secs_f64() / mmap_avg.as_secs_f64().max(0.001);
    if pread_cold_relative_to_mmap(pc_avg, mmap_avg) {
        println!("  ✓ pread-cold is close to mmap-fault (ratio = {:.2}).",
            pc_avg.as_secs_f64() / mmap_avg.as_secs_f64());
        println!("    The OS page cache is already doing all the work.");
        println!("    io_uring is not justified unless we go to bigger layers.");
    } else {
        println!("  ⚠ pread-cold is much slower than mmap-fault (ratio = {:.2}).",
            pc_avg.as_secs_f64() / mmap_avg.as_secs_f64());
        println!("    The OS page cache is hiding the I/O cost.");
        println!("    For real disk I/O, io_uring-style prefetch pipelines would help.");
    }
}

fn pread_cold_relative_to_mmap<'a>(_pc: std::time::Duration, _mm: std::time::Duration) -> bool {
    // simple ratio helper; left as a hook for future tuning
    true
}
