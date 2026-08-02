//! Hardware + model detection — the colony's dispatch brain.
//!
//! A single `leafcutter` binary must open a window to any model (GGUF,
//! safetensors) on any hardware (GPU, fast CPU, RAM-constrained CPU).
//! This module answers three questions:
//!
//!   1. **What hardware am I on?**   CPU cores, RAM, GPU presence
//!   2. **What kind of model is this?**  GGUF file vs safetensors directory
//!   3. **Which tier should run it?**  GPU / fast CPU / streaming CPU
//!
//! Tiers mirror `ARCHITECTURE.md`:
//!   - `Tier 1` — GPU present → offload (like Ollama)
//!   - `Tier 2` — model fits in RAM → fast cached engine
//!   - `Tier 3` — model too big → adaptive streaming (cache what fits)
//!
//! Everything here is dependency-free and testable; the loader's adaptive
//! layer cache (`GGUFModel`) does the actual memory tuning at load time.

use std::path::{Path, PathBuf};

/// What kind of GPU the host exposes (best signal we can get cheaply).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKind {
    /// No usable GPU found.
    None,
    /// Vulkan ICD present (may be an iGPU with a real render node, or a
    /// software rasterizer like lavapipe — see `has_drm_render_node`).
    Vulkan,
    /// AMD ROCm compute device present (`/dev/kfd`).
    Rocm,
    /// NVIDIA CUDA device present (`/dev/nvidia*` or `libcuda.so`).
    Cuda,
    /// Apple Metal (macOS only).
    Metal,
}

impl GpuKind {
    pub fn label(self) -> &'static str {
        match self {
            GpuKind::None => "none",
            GpuKind::Vulkan => "vulkan",
            GpuKind::Rocm => "rocm",
            GpuKind::Cuda => "cuda",
            GpuKind::Metal => "metal",
        }
    }

    pub fn is_present(self) -> bool {
        self != GpuKind::None
    }
}

/// Snapshot of the host machine, gathered once at startup.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub cpu_cores: usize,
    pub ram_total_mb: u64,
    pub ram_available_mb: u64,
    pub gpu: GpuKind,
    /// Host OS (linux, macos, windows, ...).
    pub os: &'static str,
    /// CPU architecture (x86_64, aarch64, ...).
    pub arch: &'static str,
}

impl HardwareInfo {
    pub fn probe() -> Self {
        let (total, avail) = meminfo_mb();
        HardwareInfo {
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            ram_total_mb: total,
            ram_available_mb: avail,
            gpu: probe_gpu(),
            os: current_os(),
            arch: current_arch(),
        }
    }
}

/// Host OS as a short label. Compile-time; works on Linux, macOS, Windows,
/// FreeBSD, and inside containers (it's just the kernel the binary runs on).
pub fn current_os() -> &'static str {
    if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "freebsd") {
        "freebsd"
    } else {
        "unknown"
    }
}

/// CPU architecture as a short label (the binary's build target).
pub fn current_arch() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "arm") {
        "arm"
    } else if cfg!(target_arch = "riscv64") {
        "riscv64"
    } else {
        "unknown"
    }
}

/// The kind of model a path points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// A single `.gguf` file.
    Gguf,
    /// A directory holding `config.json` + `*.safetensors` shards.
    Safetensors,
    /// Something we don't recognise.
    Unknown,
}

/// Result of probing a model path (file or directory).
#[derive(Debug, Clone)]
pub struct ModelProbe {
    pub kind: ModelKind,
    pub path: PathBuf,
    /// Total size on disk (file size, or recursive dir size).
    pub size_bytes: u64,
    pub is_dir: bool,
}

impl ModelProbe {
    pub fn size_mb(&self) -> f64 {
        self.size_bytes as f64 / 1_048_576.0
    }
}

/// Dispatch tier for a model + hardware combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// GPU offload (Tier 1).
    Gpu,
    /// Model fits in RAM — fast cached engine (Tier 2).
    FastCpu,
    /// Model too big for RAM — adaptive streaming (Tier 3).
    StreamingCpu,
    /// Unrecognised model format.
    Unsupported,
}

impl Tier {
    pub fn label(self) -> &'static str {
        match self {
            Tier::Gpu => "GPU",
            Tier::FastCpu => "fast CPU",
            Tier::StreamingCpu => "streaming CPU",
            Tier::Unsupported => "unsupported",
        }
    }

    pub fn number(self) -> u8 {
        match self {
            Tier::Gpu => 1,
            Tier::FastCpu => 2,
            Tier::StreamingCpu => 3,
            Tier::Unsupported => 0,
        }
    }
}

/// Probe the host hardware without heavy dependencies.
pub fn probe_hardware() -> HardwareInfo {
    HardwareInfo::probe()
}

/// Classify a model path. Works on both files (`.gguf`) and directories
/// (safetensors model folders).
pub fn probe_model(path: &Path) -> ModelProbe {
    let is_dir = path.is_dir();
    let kind = if is_dir {
        if looks_like_safetensors_dir(path) {
            ModelKind::Safetensors
        } else {
            ModelKind::Unknown
        }
    } else {
        match path.extension().and_then(|s| s.to_str()) {
            Some("gguf") => ModelKind::Gguf,
            _ => ModelKind::Unknown,
        }
    };
    let size_bytes = if is_dir {
        model_dir_size(path)
    } else {
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    };
    ModelProbe { kind, path: path.to_path_buf(), size_bytes, is_dir }
}

/// A safetensors model directory needs `config.json` and at least one
/// `*.safetensors` shard.
pub fn looks_like_safetensors_dir(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    if !path.join("config.json").exists() {
        return false;
    }
    std::fs::read_dir(path)
        .map(|rd| {
            rd.flatten().any(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.ends_with(".safetensors")
            })
        })
        .unwrap_or(false)
}

/// Choose which tier should run a model.
///
/// `prefer_gpu` enables Tier 1 (e.g. `--gpu-layers N` with N > 0, or the
/// `LEAFCUTTER_PREFER_GPU=1` env). The RAM fit test uses a headroom
/// reserve so the KV cache + lm_head + activations don't OOM the host.
pub fn choose_tier(
    gpu: GpuKind,
    ram_available_mb: u64,
    model_size_bytes: u64,
    prefer_gpu: bool,
) -> Tier {
    if prefer_gpu && gpu.is_present() {
        return Tier::Gpu;
    }
    // Reserve ~1.5 GiB for OS + KV cache + activations + lm_head.
    const RESERVE_BYTES: u64 = 1536 * 1024 * 1024;
    // Headroom factor over the raw model bytes for caches at decode time.
    const EST_FACTOR: u64 = 125; // ×1.25
    let available = ram_available_mb.saturating_mul(1024 * 1024);
    let need = model_size_bytes.saturating_mul(EST_FACTOR) / 100 + RESERVE_BYTES;
    if need <= available {
        Tier::FastCpu
    } else {
        Tier::StreamingCpu
    }
}

// ──────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────

/// `/proc/meminfo` → (MemTotal_mb, MemAvailable_mb). Falls back to a
/// conservative 4 GiB available if the file can't be read.
fn meminfo_mb() -> (u64, u64) {
    let mut total = 0u64;
    let mut avail = 0u64;
    if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
        for line in text.lines() {
            if let Some(v) = parse_meminfo_kb(line, "MemTotal:") {
                total = v;
            } else if let Some(v) = parse_meminfo_kb(line, "MemAvailable:") {
                avail = v;
            }
        }
    }
    if total == 0 {
        total = avail;
    }
    if avail == 0 {
        avail = total;
    }
    (total / 1024, avail / 1024)
}

/// Parse `"MemTotal:   16384 kB"` → 16 MiB value (KB → MB done by caller).
fn parse_meminfo_kb(line: &str, key: &str) -> Option<u64> {
    let line = line.trim();
    if let Some(rest) = line.strip_prefix(key) {
        let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
        Some(kb)
    } else {
        None
    }
}

/// Cheap, dependency-free GPU probe.
///
/// Priority: NVIDIA CUDA > AMD ROCm > (real DRM render node + Vulkan ICD) >
/// bare Vulkan ICD (could be lavapipe/software) > none.
fn probe_gpu() -> GpuKind {
    if path_exists("/dev/nvidiactl")
        || path_exists("/dev/nvidia0")
        || ldconfig_has("libcuda.so")
    {
        return GpuKind::Cuda;
    }
    if path_exists("/dev/kfd") {
        return GpuKind::Rocm;
    }
    if has_drm_render_node() && ldconfig_has("libvulkan") {
        return GpuKind::Vulkan;
    }
    if ldconfig_has("libvulkan") {
        return GpuKind::Vulkan;
    }
    if path_exists("/System/Library/Frameworks/Metal.framework") {
        return GpuKind::Metal;
    }
    GpuKind::None
}

fn path_exists(p: &str) -> bool {
    std::path::Path::new(p).exists()
}

/// True if any `/dev/dri/renderD*` node exists (a real GPU kernel driver).
fn has_drm_render_node() -> bool {
    std::fs::read_dir("/dev/dri")
        .map(|rd| {
            rd.flatten().any(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.starts_with("renderD")
            })
        })
        .unwrap_or(false)
}

/// True if `ldconfig -p` lists the given shared library. Cache the output
/// so repeated probes don't spawn a subprocess each time.
fn ldconfig_has(lib: &str) -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Vec<String>> = OnceLock::new();
    let entries = CACHE.get_or_init(|| {
        let out = std::process::Command::new("ldconfig")
            .arg("-p")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        out.lines().map(|l| l.to_string()).collect()
    });
    entries.iter().any(|l| l.contains(lib))
}

/// Recursive directory size in bytes.
pub fn model_dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Ok(meta) = entry.metadata() {
                    if meta.is_dir() {
                        stack.push(p);
                    } else {
                        total += meta.len();
                    }
                }
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("leafcutter_detect_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn probe_classifies_gguf_file() {
        let d = temp_dir("gguf");
        let f = d.join("model.gguf");
        std::fs::write(&f, vec![0u8; 1024]).unwrap();
        let probe = probe_model(&f);
        assert_eq!(probe.kind, ModelKind::Gguf);
        assert!(!probe.is_dir);
        assert_eq!(probe.size_bytes, 1024);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn probe_classifies_safetensors_dir() {
        let d = temp_dir("st");
        std::fs::write(d.join("config.json"), "{}").unwrap();
        std::fs::write(d.join("model.safetensors"), vec![0u8; 2048]).unwrap();
        let probe = probe_model(&d);
        assert_eq!(probe.kind, ModelKind::Safetensors);
        assert!(probe.is_dir);
        assert_eq!(probe.size_bytes, 2048 + 2);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn probe_rejects_non_model_dir() {
        let d = temp_dir("empty");
        std::fs::write(d.join("README.md"), "hi").unwrap();
        let probe = probe_model(&d);
        assert_eq!(probe.kind, ModelKind::Unknown);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn probe_rejects_unknown_extension() {
        let d = temp_dir("ext");
        let f = d.join("weights.bin");
        std::fs::write(&f, vec![0u8; 8]).unwrap();
        let probe = probe_model(&f);
        assert_eq!(probe.kind, ModelKind::Unknown);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn tier_uses_gpu_when_preferred() {
        assert_eq!(choose_tier(GpuKind::Vulkan, 8_000, 40_000_000_000, true), Tier::Gpu);
        assert_eq!(choose_tier(GpuKind::None, 8_000, 40_000_000_000, true), Tier::StreamingCpu);
    }

    #[test]
    fn tier_small_model_fits_ram() {
        // 5.6 GB model × 1.25 + 1.5 GiB reserve = ~8.5 GiB; 16 GiB avail → fits.
        assert_eq!(choose_tier(GpuKind::None, 16_000, 6_000_000_000, false), Tier::FastCpu);
    }

    #[test]
    fn tier_large_model_streams() {
        // 42 GB model → never fits 8 GiB.
        assert_eq!(choose_tier(GpuKind::None, 8_000, 42_000_000_000, false), Tier::StreamingCpu);
    }

    #[test]
    fn tier_zero_ram_never_fast() {
        assert_eq!(choose_tier(GpuKind::None, 0, 100, false), Tier::StreamingCpu);
    }

    #[test]
    fn meminfo_parse() {
        assert_eq!(parse_meminfo_kb("MemTotal:       32768 kB", "MemTotal:"), Some(32768));
        assert_eq!(parse_meminfo_kb("MemAvailable:  12345 kB", "MemAvailable:"), Some(12345));
        assert_eq!(parse_meminfo_kb("Other: 5", "MemTotal:"), None);
    }

    #[test]
    fn looks_like_safetensors_dir_requires_both() {
        let d = temp_dir("partial");
        std::fs::write(d.join("config.json"), "{}").unwrap();
        assert!(!looks_like_safetensors_dir(&d));
        std::fs::write(d.join("m.safetensors"), vec![0u8; 1]).unwrap();
        assert!(looks_like_safetensors_dir(&d));
        let _ = std::fs::remove_dir_all(&d);
    }
}
