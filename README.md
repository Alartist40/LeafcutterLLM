# 🌿 LeafcutterLLM

**One binary. Any GGUF. Adaptive local AI on strong or weak hardware.**

LeafcutterLLM is a memory-first adaptive LLM runtime written entirely in Rust. It runs massive models — from small 1.5B/3B models to 27B and 70B parameters — on hardware you already own, whether it's a 12-core ARMv9 board, an SBC, a laptop, or a desktop server.

```bash
curl -fsSL https://raw.githubusercontent.com/Alartist40/LeafcutterLLM/main/install.sh | bash
```

[![Rust 1.86](https://img.shields.io/badge/Rust-1.86-000000?logo=rust)](https://rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green)]()
[![Tests: 202 pass](https://img.shields.io/badge/Tests-202%20pass%2F%200%20fail-brightgreen)]()

![LeafCutter logo](https://github.com/Alartist40/LeafcutterLLM/blob/e95fe79a9628a2c165ffe46ebd1350d7f4dead6f/LeafCutter_logo_light.png)

---

## 📋 Table of Contents

- [Why Leafcutter?](#-why-leafcutter)
- [Install & Quick Start](#-install--quick-start)
- [Command Cheat Sheet](#-command-cheat-sheet)
- [Validated Model Roster & Benchmarks](#-validated-model-roster--benchmarks)
- [Hardware Auto-Detection & Silicon Speccing](#-hardware-auto-detection--silicon-speccing)
- [Architecture & Technical Design](#-architecture--technical-design)
- [Interactive Terminal UI (Gold & Purple Palette)](#-interactive-terminal-ui)
- [Build from Source](#-build-from-source)
- [Container Deployment](#-container-deployment)
- [HTTP API](#-http-api)
- [Roadmap](#-roadmap)

---

## 🚀 Install & Quick Start

**Requires:** Linux or macOS, 2 GB+ RAM, any 64-bit CPU (x86_64 or aarch64). No GPU, no Python, no CUDA required.

```bash
curl -fsSL https://raw.githubusercontent.com/Alartist40/LeafcutterLLM/main/install.sh | bash
```

The installer installs the binary to your PATH and sets up both `leafcutter` and the `leaf` shortcut command. 

Update to the latest version at any time:
```bash
leafcutter update          # Or shortcut: leaf update
leafcutter update --from-source  # Rebuild from source automatically
```

---

## 💡 Command Cheat Sheet

| Command | Shortcut | Description |
|---|---|---|
| `leafcutter list` | `leaf list` | Auto-detect and list available GGUF models in `./models` and `~/Downloads/models` |
| `leafcutter source add <dir>` | `leaf source add <dir>` | Add a persistent folder of GGUF models (stored in `~/.config/leafcutter`) |
| `leafcutter source list` | `leaf source list` | View all configured model directories |
| `leafcutter run <model>` | `leaf run <model>` | Launch interactive chat REPL with gold & purple complementary terminal UI |
| `leafcutter run <path> --max-tokens 32` | `leaf run ...` | Run inference with explicit token limits |
| `leafcutter generate --model <path> --prompt "..."` | `leaf generate ...` | Run one-shot generation from command line |
| `leafcutter serve --host 0.0.0.0 --port 8081` | `leaf serve ...` | Launch OpenAI-compatible REST API server (`/v1/chat/completions`) |
| `leafcutter update` | `leaf update` | Download or build the latest release |

### In-Session Slash Commands (inside `leafcutter run` REPL):
- `/help`, `/?` — Display available commands and active settings (temperature, top_p, max_tokens, profile).
- `/set <key> <val>` — Set parameters mid-session (`/set temp 0.7`, `/set top_p 0.95`, `/set system ...`).
- `/temp <val>` — Quick alias to set temperature (`/temp 0.5`).
- `/show <target>` — Show loaded model info, profile, system prompt, or session stats.
- `/info` — View loaded model's tensor layers, architecture, and memory footprint.
- `/stats` — Show rolling session token speed (tok/s), latency, and peak RAM consumption.
- `/clear` — Reset conversation history and flush state/KV caches.
- `/source` — List or modify model source directories mid-session.
- `/bye`, `/quit` — Exit the interactive chat cleanly.

---

## 📊 Validated Model Roster & Benchmarks

Every entry below has been tested and verified on real hardware:

| Model | Size | Architecture | Status | Measured Performance & RAM |
|---|---|---|---|---|
| **Ornith 1.0 9B** (Flagship) | 5.24 GiB | Qwen3.5 hybrid (DeltaNet) | ✅ Coherent Reasoning | **1.5–2.4 tok/s**, ~3.3 GB peak RAM (14 GiB ARM SBC, ARM sdot kernels) |
| **Qwen3.8-27B** Q4_K_M | 15.93 GiB | Qwen3.5 MoE (64 layers, 5120 hidden) | ✅ Verified Coherent | Loaded & generated coherent persona chat |
| **Ornith 1.0 35B** Q4_K_M | 19.71 GiB | Qwen3.6 MoE (256 experts) | ✅ Native Layer Streaming | **3,963 MB** peak RAM in low-RAM mode |
| **Ministral-3-3B** Q4_K_M | 2.00 GiB | Mistral / YaRN context scaling | ✅ Coherent Chat & Math | `2+2=4.`, ~504 MB peak RAM (Native/FFI) |
| **Meta-Llama-3.1-70B** Q4_K_S | 40.3 GB | Llama | ✅ 1-token forward | **1,145 MB** peak RAM on 4 GB machine |
| **Qwen2.5-1.5B** Q4_K_M | 1.04 GB | Qwen2 | ✅ Native | **4.07 tok/s** (21.6× faster than AirLLM) |

---

## 🔬 Hardware Auto-Detection & Silicon Speccing

LeafcutterLLM automatically probes system hardware at startup to choose memory limits, thread pools, and execution tiers:

```
  ╔══════════════════════════════════════════════╗
  ║  🌿 LeafcutterLLM — Native Engine            ║
  ╠══════════════════════════════════════════════╣
  ║  Model   : Qwen3.8-27B-Q4_K_M.gguf           ║
  ║  Arch    : Qwen3.5                           ║
  ║  Layers  : 64 layers, 5120 hidden            ║
  ║  Size    : 16314.3 MB                        ║
  ║  Hardware: linux · 12 cores · 14 GiB free    ║
  ║  Tier    : 3 — streaming CPU                 ║
  ║  Profile : ornith                            ║
  ║  Temp    : 0.60  (top_p=0.95)                ║
  ║  Max tok : 32                                ║
  ╚══════════════════════════════════════════════╝
```

- **Processor Probing**: Probes logical CPU count, big.LITTLE core configurations, ARM NEON / AVX2 availability, and thread concurrency ceilings.
- **NPU Detection**: Probes fixed-function NPUs (e.g. Arm China Zhouyi AIPU `/dev/aipu`, 28.8 TOPS INT8). Reports `supports_dynamic_offload() == false` honestly since embedded NPUs require precompiled offline graphs.
- **Memory Tiers**:
  - **Tier 1 (Full RAM)**: Fits entire model in RAM/VRAM for max throughput.
  - **Tier 2 (Cached Streaming)**: Pins dense weights, streams layer tensors with prefetch.
  - **Tier 3 (Low-RAM Streaming)**: Evicts layer weights with `madvise(DONTNEED)` for machines with tight RAM.

---

## 🎨 Interactive Terminal UI

LeafcutterLLM features a custom terminal interface inspired by **Paraclea**:
- **Seamless Complementary Palette**: Reasoning and thinking tokens stream softly in **dimmed purple (`dim_purple`)** so thinking blends subtly into the background. When reasoning completes (`</think>`), the assistant response streams immediately in **bright Gold (`#FFD700`)** without cluttering text labels (`Thinking...` / `Leafcutter >`).
- **Token Speed & Peak RSS HUD**: Each turn displays real-time execution statistics (output tokens, elapsed time, tok/s, current RAM, and peak RSS memory).

---

## ⚡ ARM Hardware Acceleration & Memory Layer Prefetching

- **Automatic Thread Pool Sizing ([`init.rs`](file:///home/orangepi/Documents/portfolio/LeafcutterLLM/rust/src/init.rs))**: Auto-detects physical ARMv9 cores (e.g. 12-core CIX Sky1 / RK3588) and scales Rayon worker threads across all physical cores rather than halving SMT threads.
- **ARM dot-product (sdot) Quantized Kernels ([`q8_k.rs`](file:///home/orangepi/Documents/portfolio/LeafcutterLLM/rust/src/kernels/q8_k.rs))**: On ARMv8.2+ (`asimddp`), Q4_K/Q6_K × Q8_K dots use the `sdot` instruction (bit-exact vs scalar). Q4_K decode uses a single-column sdot layout; Q6_K stays NEON for the bandwidth-bound lm_head.
- **Batched Prefill GEMM**: Quantized Q8 GEMV/Gemm quantizes each activation row once and builds each weight column's block buffers once, reusing them across rows (63 s → 7.5 s prefill at m=77).
- **Layer Prefetching ([`engine.rs`](file:///home/orangepi/Documents/portfolio/LeafcutterLLM/rust/src/inference/engine.rs))**: Automatically spawns background layer-load threads to overlap layer $l+1$ parsing with layer $l$ GEMV compute.
- **RAM Layer Caching ([`loader.rs`](file:///home/orangepi/Documents/portfolio/LeafcutterLLM/rust/src/model/loader.rs))**: When total system RAM fits the model, layer weights remain resident in memory after pass 1, enabling 100% in-RAM CPU execution.

---

## 🛠 Build from Source

### Pure Native Build (No external dependencies)
```bash
git clone https://github.com/Alartist40/LeafcutterLLM.git
cd LeafcutterLLM/rust
cargo build --release --no-default-features --bin leafcutter
```

### With llama.cpp FFI (Fallback path for exotic quants)
```bash
./scripts/build_llama_cpp.sh
cd LeafcutterLLM/rust
cargo build --release --features llama-ffi
```

---

## 🐳 Container Deployment

Pre-built container images run at full native speed without virtual machine overhead:

```bash
docker run --rm -it \
  -p 8081:8081 \
  -v /path/to/models:/models \
  -e LEAF_MODELS_DIR=/models \
  ghcr.io/alartist40/leafcutterllm:latest serve --host 0.0.0.0 --port 8081
```

---

## 🌐 HTTP API

### OpenAI-Compatible Chat (`POST /v1/chat/completions`)
```bash
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 64
  }'
```

---

## ⚙️ Settings & Environment Variables

Sampling and behavior defaults are per-profile (see `rust/src/profiles.rs`) and can
be changed mid-session in the REPL (`/set temp 0.7`, `/set top_p 0.95`). The CLI
flags mirror them (`--temp`, `--top-k`, `--top-p`, `--max-tokens`).

Persistent model search paths live in `~/.config/leafcutter/config.json` (Linux),
managed with `leafcutter source add/list/remove` or `LEAF_MODELS_DIR`.

| Variable | Default | Effect |
|---|---|---|
| `LEAF_MODELS_DIR` | `./models`, `~/Downloads/models` | Colon-separated extra model search dirs |
| `LEAFCUTTER_THREADS` | physical cores (ARM) / cores−1 (x86) | Rayon worker thread count |
| `LEAFCUTTER_NO_CACHE` | off | Tier 3 low-RAM mode: stream + evict layer weights (`madvise(DONTNEED)`) |
| `LEAFCUTTER_PREFETCH` | on when RAM fits | Background layer $l+1$ prefetch during layer $l$ compute |
| `LEAFCUTTER_CACHE_MB` | auto | Cache budget ceiling for resident layers |
| `LEAFCUTTER_CTX_KB` | auto | KV-cache context budget |
| `LEAFCUTTER_DETERMINISTIC` | off | Force bit-identical serial reductions (`LEAFCUTTER_Q8_GEMV=0` implied) |
| `LEAFCUTTER_Q8_GEMV` | on | `0` disables the ARM sdot Q8 GEMV path (NEON fallback) |
| `LEAFCUTTER_PREFER_GPU` | off | Prefer a detected Vulkan/GPU path when available |
| `LEAFCUTTER_TOP_K` | profile (2048) | Sampling top-k cap |
| `LEAFCUTTER_API_KEY` / `LEAFCUTTER_BASE_URL` | — | Remote model API credentials (for API-backed models) |
| `LEAFCUTTER_PROFILE` | off | Per-component layer timing (`pre_norm`, `ffn_forward`, per-matmul, lm_head) to stderr |
| `LEAFCUTTER_MODEL` | — | Model path override for scripts |
| Debug | off | `LEAFCUTTER_DEBUG`, `LEAFCUTTER_DEBUG_LAYERS`, `LEAFCUTTER_DEBUG_NORMS`, `LEAFCUTTER_DEBUG_PROMPT`, `LEAFCUTTER_ROPE_DEBUG`, `LEAFCUTTER_TOKENIZER_DEBUG`, `LEAFCUTTER_DELTANET_DEBUG`, `LEAFCUTTER_CHUNK_DEBUG`, `LEAFCUTTER_OLLAMA_DEBUG`, `LEAFCUTTER_CPU_MONITOR`, `LEAFCUTTER_PROFILE_BLOCKS` |

## 📜 License

MIT License. See [LICENSE](LICENSE).

---

**Made with 🌿 for running intelligence everywhere, on any hardware.**
