# LeafcutterLLM Strategy — 2026 July

> **Date:** 2026-07-03
> **Audience:** Core team + stakeholders
> **Status:** Draft for brainstorm. No code edits yet.
> **Scope:** Two directions: (A) CPU thermal control, (B) GPU-accelerated image model expansion

---

## Context

LeafcutterLLM now runs Ornith 1.0 9B end-to-end on native Rust, peaking at **1,216 MB RSS** (vs 8,155 MB in llama.cpp). That is a 6.7× RAM reduction that makes multi-billion-parameter models viable on edge devices. The trade-off is throughput: **0.55 tok/s on CPU** for a 9B model, saturating all 4 cores on a Raspberry Pi 5 and pushing **300 % CPU utilisation**.

This strategy addresses two natural next steps without altering existing inference code.

---

## A. CPU Thermal & Power Management

### A.1 The Problem

Running matrix multiplications at full utilisation on a Raspberry Pi 5 (ARM Cortex-A76, no active cooling in stock config) drives the SoC into thermal throttling within seconds. Throttling begins at ~80 °C and hard-limits at ~85 °C. The user reports 300 % CPU use — essentially 3 cores at 100 %, one idle. This suggests one of:

- The current matmul kernels are single-thread bounded and the runtime uses 3 workers.
- Thread-pool size equals number of physical cores minus one (leaving one for OS).
- The build is not yet using all 4 cores because of: (a) a thread-pool cap, or (b) a workload that cannot split across all cores.

Either way, the result is a hot SoC, audible fan ramp-up (or none), and potential long-term hardware degradation.

### A.2 Available Mitigations (No Code Changes Required)

These are all runtime / OS / BIOS / user-side levers. They can be documented, scripted, and adopted immediately.

#### A.2.1 Linux CPU Governor & Frequency Scaling

```bash
# Check current governor
$ cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Swap to conservative / powersave (throttles down at low utilisation)
sudo cpupower frequency-set -g powersave   # or userspace + set limit

# Hard-limit max frequency on all cores
for i in /sys/devices/system/cpu/cpu*/cpufreq; do
    echo 1200000 | sudo tee $i/scaling_max_freq  # 1.2 GHz cap (stock = 2.4 GHz)
done
```

Trade-off: ~50 % throughput loss, significant temperature drop.

#### A.2.2 User-space Throttle (Sleep Between Layers)

The existing layer-streaming loader in `rust/` already runs one transformer block at a time. We can insert a configurable `std::thread::sleep(Duration::from_millis(N))` between each `forward` call. This exists as a compile-time or runtime flag today. Documentation only is needed to expose it as a product feature: "Eco Mode".

#### A.2.3 Batch-Size Limiting

For the HTTP API (`/v1/chat/completions`), incoming prompts are batched. Capping the batch size to 1 concurrent request prevents sustained 100 % utilisation across all cores. This is already configurable in the Axum server's startup parameters. Document the tunable.

#### A.2.4 Hardware-Level Cooling (External)

This is a procurement / ops decision, not a code change:

- **Passive heatsink (Aluminium case)** — drops SoC by 10–15 °C under load.
- **Active fan (Pimoroni Fan Shim)** — drops SoC by 20–25 °C under load.
- **Thermal pad + heat spreader** — necessary if enclosure prevents airflow.

### A.3 Architecture Decision Record

| Option | Effort | Throughput Impact | Thermal Impact | Recommended For |
|--------|--------|-------------------|---------------|-----------------|
| Governor = powersave | 1 min | −30 % | −15 °C | Quick fix, ops script |
| Frequency cap to 1.2 GHz | 1 min | −50 % | −25 °C | Emergency cooling |
| Inter-layer sleep (Eco Mode) | 0 (already wired) | −varies | −10 °C | User-facing product toggle |
| Batch cap = 1 | 0 (already wired) | −75 % | −20 °C | API server default on Pi |
| Hardware active cooling | $15–30 | None | −25 °C | Permanent solution |

**Recommended team action:**
1. Document the three existing tunables (Eco Mode sleep, batch cap, governor setting) in `README.md` under a "Running on Raspberry Pi" section.
2. Provide a `pi-thermal-profile.sh` script that applies governor + frequency limits, queries `vcgencmd measure_temp`, and reports safe/unsafe.
3. Add a `--eco` CLI flag to `test_generation` and the server binary that enables inter-layer sleep and single-request batching. This is a **5-line wrapper change**, not a logic change.

### A.4 Governance — Who Does What

| Task | Owner | Time |
|------|-------|------|
| Document governor / frequency / Eco Mode levers | Docs/PM | 2 h |
| Write `pi-thermal-profile.sh` | DevOps | 4 h |
| Add `--eco` flag to CLI / server | Backend dev | 2 h |
| Benchmark thermal delta on Pi 5 | QA | 4 h |
| Procure & test active cooling | Hardware | Parallel |

---

## B. GPU-Accelerated Image Model Expansion

### B.1 The Problem

LeafcutterLLM is text-only. The user wants to add image generation. State-of-the-art diffusion models (Stable Diffusion 3, FLUX, SDXL) run on GPUs. CPU inference is possible (via `ggml_diffusion.cpp` or `onnxruntime` CPU provider) but is 20–50× slower, making it impractical for interactive use.

The strategy is: **Detect GPU at runtime; if present, offload diffusion to it. If absent, degrade gracefully (reject, queue, or CPU-fallback depending on SLA).** Text models remain on the CPU path unchanged.

### B.2 Existing GPU Infrastructure to Leverage

Leafcutter already builds against `llama.cpp`, which ships with:

- **CUDA** (NVIDIA, Linux/Windows)
- **Metal** (Apple Silicon, macOS)
- **Vulkan** (cross-platform, Intel/AMD/ARM Mali/Adreno)
- **OpenCL** (legacy, broad support but slower)

The build system (`rust/build.rs`, `CMakeLists.txt`) already has feature flags for each backend (`-DLLAMA_CUDA=ON`, etc.). We can reuse this detection and infrastructure.

Additionally, the Rust ecosystem has:

- **`candle-core`** — Hugging Face's Rust ML framework. Supports Metal and CUDA.
- **`tract-core`** — Rust-native ONNX inference. Supports GPU via CUDA / Metal delegates.
- **`gguf`** format for diffusion — `stable-diffusion.cpp` (Georgi Gerganov) and `diffusers-rs` provide GGUF diffusion.
- **Stable Diffusion GGUF models** — already published by TheBloke and community.

### B.3 Proposed Architecture

#### B.3.1 Runtime GPU Detection

```rust
// Pseudocode — no code edits yet
enum Gpu acceleration {
    None,
    Cuda { device_id: i32 },
    Metal { device_id: i32 },
    Vulkan { device_id: i32 },
    OpenCl { platform_id: i32, device_id: i32 },
}

fn detect_gpu() -> GpuAcceleration {
    // Priority: CUDA > Metal > Vulkan > OpenCL > None
    // Check env var LEAFCUTTER_GPU_BACKEND to force selection
    // Return None if no GPU or env var disables GPU
}
```

This is a **new module**, not touching existing text-inference paths.

#### B.3.2 Diffusion Backend Abstraction

```rust
// New trait, new implementations
trait DiffusionBackend {
    fn generate(&self, prompt: &str, negative_prompt: &str,
              width: u32, height: u32,
              steps: u32, cfg_scale: f32,
              seed: Option<u64>) -> Result<Image, Error>;
}

// Implementations:
// - CpuDiffusionBackend -> candle / tract
// - CudaDiffusionBackend -> candle with CUDA
// - MetalDiffusionBackend -> candle with Metal
// - VulkanDiffusionBackend -> via Vulkan compute shaders (advanced)
```

#### B.3.3 Integration Point

The existing HTTP server (`/v1/chat/completions`) remains untouched. A **new endpoint** is added:

```
POST /v1/images/generations   # OpenAI-compatible
POST /v1/images/edits         # OpenAI-compatible
```

This is a **new Axum route**, not modifying existing handlers.

#### B.3.4 Model Format & Storage

For diffusion models, the recommended path is **GGUF** (reuse existing GGUF loader) or **ONNX** (use `tract`).

| Format | Pros | Cons | Status |
|--------|------|------|--------|
| GGUF (diffusion) | Reuse loader, mmap, layer streaming | Limited model availability | Growing community support |
| ONNX (Diffusers export) | Broad model availability, standard format | Larger file sizes (~2× GGUF) | Mature ecosystem |
| Safetensors (Hugging Face) | Native `candle` support | No layer streaming | Good for GPU |

Decision point for team:
1. **Short term (PoC):** ONNX via `tract-core` with CPU/CUDA fallback. Fastest path to a working `/v1/images/generations` endpoint.
2. **Medium term:** GGUF diffusion via `stable-diffusion.cpp` Rust bindings. Aligns withLeafcutter's existing GGUF + mmap philosophy.
3. **Long term:** Native Rust diffusion kernels (quantised GEMM for UNet, VAE decode). Reuse the SIMD matmul work already done for text models.

### B.4 Security & Sandboxing

Running diffusion models on GPU opens a new attack surface:

- **Prompt injection / NSFW generation** — content filter required before reaching the model.
- **Resource exhaustion** — GPU memory is finite. A single SDXL model needs ~6–8 GB VRAM. The scheduler must queue or reject requests when VRAM is exhausted.
- **Thermal** — GPU workloads generate even more heat than CPU. The thermal levers in Section A apply doubly here.

Mitigation: implement a **request pre-filter** (text safety classifier, black/white list) and a **GPU memory monitor** (query VRAM before accepting diffusion request, return 503 if unavailable).

### B.5 Governance — Who Does What

| Phase | Duration | Owner | Deliverable |
|-------|----------|-------|-------------|
| 1. GPU Detection module | 1 day | Backend | `gpu_detect.rs` with env-var override |
| 2. ONNX diffusion PoC | 3 days | Backend | Working `/v1/images/generations` on CPU |
| 3. CUDA backend wiring | 2 days | Backend | GPU offloading via tract/CUDA delegate |
| 4. Content safety filter | 2 days | ML / Infra | Prompt classifier + NSFW rejection |
| 5. Integration & benchmark | 2 days | QA / Dev | Latency/VRAM/thermal numbers on target hardware |
| 6. Documentation | 1 day | Docs | API docs, deployment guide, model zoo |

**Total estimate:** 2 weeks for a minimal viable diffusion endpoint.

---

## C. Unified CLI / Server Interface

Both directions (thermal, image) should be exposed through a unified, non-breaking CLI and configuration system.

### C.1 CLI Flags (Backward-Compatible)

```bash
# Thermal control (Section A)
leafcutter generate --model ... --eco                # Enable inter-layer sleep + single-batch
leafcutter server --thermal-limit 80                 # Throttle when SoC > 80 °C
leafcutter server --max-concurrent 1                  # Already exists; document for Pi

# Diffusion control (Section B)
leafcutter generate-image --prompt "a cat" \
                           --size 512x512 \
                           --steps 20 \
                           --gpu auto              # auto / cuda / metal / vulkan / cpu

# Unified server start
leafcutter server --model-text ... \
                   --model-image ... \
                   --gpu auto \
                   --eco
```

### C.2 Configuration File

```yaml
# leafcutter.yaml
server:
  port: 8081
  max_concurrent: 4
  gpu_backend: auto   # none / cuda / metal / vulkan / opencl

text_model:
  path: ~/models/llama-3.2-3b.gguf
  eco_mode: false     # inter-layer sleep
  thermal_limit_c: 80

image_model:
  path: ~/models/sdxl-base.gguf   # or .onnx
  backend: auto                   # overrides server.gpu_backend
  max_queue: 10
  safety_filter: true
```

These are **new top-level keys**, not conflicting with existing Rust struct layouts.

---

## D. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| GPU detection false-positives on headless servers | Medium | High (crash) | Add `--gpu none` override; test on CI |
| Diffusion model VRAM > available GPU memory | High | High (OOM) | Pre-flight VRAM check; queue or CPU fallback |
| Thermal throttling on Pi makes text model unusable | Medium | High (UX) | Default `--eco` on ARM; hardware cooling rec |
| ONNX dependency bloats binary (>20 MB) | Medium | Medium | Feature-gate (`--features diffusion-onnx`); CI builds slim + fat variants |
| Content moderation misses NSFW | Low | Critical | Layered defence: keyword list + cheap classifier + human review queue |
| Team bandwidth — two directions at once | High | Medium | Sequence A before B; thermal is quick win, diffusion is strategic bet |

---

## E. Decision Checklist for Team Brainstorm

| # | Question | Options | My Recommendation |
|---|----------|---------|-------------------|
| 1 | Should thermal control be a compile-time or runtime toggle? | Compile = simpler binary; Runtime = more flexible | **Runtime** (`--eco`, `--thermal-limit`) — one-liners in Rust, huge UX gain |
| 2 | Hardware cooling — stock or aftermarket? | Stock = cheaper; Aftermarket = quieter, cooler | **Aftermarket fan** ($15) as default in Pi kit; document in README |
| 3 | Diffusion format — ONNX or GGUF? | ONNX = faster PoC; GGUF = aligned with project philosophy | **ONNX for PoC**, **GGUF for production** (2-phase) |
| 4 | GPU backend priority? | CUDA > Metal > Vulkan > None | **CUDA > Metal > Vulkan > None** — match llama.cpp priority, well-tested |
| 5 | Should image generation share the same server port? | Same = simpler; Separate = isolation, independent scaling | **Same server, separate routes** (`/v1/images/*`) — standard OpenAI compat |
| 6 | Should we expose GPU metrics (VRAM, temp) in `/health`? | Yes = transparency; No = less surface area | **Yes** — add `gpu_vram_mb`, `gpu_temp_c` to `/health` JSON; helps ops |
| 7 | Timebox for thermal MVP? | 1 day / 3 days / 1 week | **1 day** — script + docs + `--eco` flag |
| 8 | Timebox for diffusion PoC? | 1 week / 2 weeks / 1 month | **2 weeks** — ONNX CPU path first, GPU second week |
| 9 | Should we maintain a model zoo (download scripts)? | Yes = onboarding friction drops; No = less maintenance | **Yes** — extend existing `scripts/download_models.sh` to include diffusion GGUFs / ONNX exports |
| 10 | Priority: thermal or diffusion first? | Thermal = user pain today; Diffusion = strategic growth | **Thermal first** (1 day),公众的 thermal first; diffusion parallel after PoC |

---

## F. Appendix: Target Hardware Specifications

| Device | CPU | Cores | RAM | GPU | Use Case |
|--------|-----|-------|-----|-----|----------|
| Raspberry Pi 5 | ARM Cortex-A76 | 4 | 8 GB | None (VideoCore VII, no CUDA) | Edge text inference, thermal-constrained |
| NVIDIA Jetson Orin Nano | ARM Cortex-A78AE | 6 | 8 GB | 1024-core Ampere | Edge GPU diffusion + text |
| Apple MacBook Air M3 | Apple M3 | 8 | 16 GB | Core (Metal) | Dev, Metal diffusion |
| Generic x86_64 | Intel/AMD | 8+ | 32 GB | NVIDIA RTX 4060 / 3060 | Primary dev, CUDA diffusion |
| Cloud (AWS EC2 g4dn) | Intel Xeon | 4 | 16 GB | NVIDIA T4 (16 GB) | Production GPU inference |

**Observation:** Pi 5 has no CUDA. So for Pi 5, diffusion must either (a) run on CPU (slow), or (b) not run at all (graceful degradation). The GPU detection logic must handle this — "GPU auto" on Pi 5 returns `None`, and the image endpoint returns `501 Not Implemented` or `503 GPU Required`.

However, the **Jetson Orin Nano** ($499) has both ARM CPU and Ampere GPU. This is the natural next hardware target for GPU diffusion on edge. The architecture above (CUDA detection + offload) works here.

---

## G. Success Criteria

| Milestone | Metric | Target |
|-----------|--------|--------|
| Thermal MVP | SoC temp under sustained load | < 75 °C on Pi 5 with Ornith 9B |
| Thermal MVP | Throughput in eco mode | > 0.3 tok/s (acceptable for chat) |
| Diffusion PoC | Image generation latency (512×512, 20 steps) | < 30 s on RTX 4060 |
| Diffusion PoC | VRAM usage (SDXL) | < 8 GB |
| Diffusion PoC | CPU fallback latency | < 120 s (functional, not fast) |
| Diffusion Production | GPU backend coverage | CUDA + Metal + Vulkan |
| General | Binary size increase | < 20 MB with diffusion feature gate |

---

## H. Call to Action

1. **Read this document** in the next team meeting.
2. **Vote on decision checklist items** (Section E).
3. **Assign owners** for thermal MVP (estimated 1 day) and diffusion PoC (estimated 2 weeks).
4. **Approve hardware spend** if active cooling or Jetson Orin Nano is desired.
5. **Archive this document** once decisions are made; move accepted items to GitHub issues / project board.

---

*End of strategy. No code was harmed in the making of this document.*
