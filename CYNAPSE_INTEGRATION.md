# Cynapse + Leafcutter Integration Guide

**Cynapse** is the orchestration layer — it downloads models, manages conversations, and provides the TUI/API. **Leafcutter** is the inference engine — it runs those models efficiently via direct llama.cpp FFI.

Together they form a complete local AI system:

```
┌─────────────────────────────────────────────────────────────┐
│                        Cynapse                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Model Hub  │  │  TUI / API  │  │  Memory (Dendrite)  │  │
│  │  (download) │  │  (chat)     │  │  (context tracking) │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────┘  │
│         │                │                                    │
│         └────────────────┼────────────────────────────────────┘
│                          │
│              "Run this model"                                  │
│                          │                                     │
│         ┌────────────────┘                                     │
│         ▼                                                      │
│  ┌─────────────────────────────────────────────┐              │
│  │           Leafcutter (inference)             │              │
│  │  ┌─────────────┐      ┌──────────────────┐  │              │
│  │  │  Native Rust │  or  │  llama.cpp FFI   │  │              │
│  │  │  (kernels)   │      │  (universal)     │  │              │
│  │  └─────────────┘      └──────────────────┘  │              │
│  └─────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture Philosophy

**Why two projects instead of one?**

- **Cynapse** focuses on UX: downloading, organizing, chatting, tool use, memory
- **Leafcutter** focuses on performance: quantization, kernels, inference speed
- They communicate via simple CLI / HTTP interfaces
- Users can use either independently, but together they're greater than the sum

**Key advantage over Cynapse's current llama-server approach:**

| Aspect | Cynapse + llama-server | Cynapse + Leafcutter FFI |
|--------|------------------------|--------------------------|
| Startup time | 2–5 seconds (process spawn) | **Instant** (shared library) |
| Memory overhead | ~200MB extra (server process) | **Zero** (same process) |
| Context switching | Save/restore KV cache to disk | **Keep in RAM** |
| Throughput | HTTP JSON overhead per token | **Direct memory** |
| Reliability | Process can crash independently | **Single process** |

---

## Installation

### Option A: Single-Line Install (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/Alartist40/LeafcutterLLM/main/install.sh | bash
```

This will:
1. Install Rust (if missing)
2. Clone Leafcutter + llama.cpp
3. Build shared libraries
4. Build the `leafcutter` CLI binary
5. Add `leafcutter` to your `PATH`
6. Configure `LD_LIBRARY_PATH` in your shell profile

### Option B: Manual Install

```bash
# 1. Clone repositories
git clone https://github.com/Alartist40/LeafcutterLLM.git ~/.leafcutter/LeafcutterLLM
git clone https://github.com/ggml-org/llama.cpp.git ~/.leafcutter/llama.cpp

# 2. Build llama.cpp shared libraries
cd ~/.leafcutter/llama.cpp
mkdir -p build && cd build
cmake .. -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON
cmake --build . --parallel $(nproc)

# 3. Build Leafcutter
cd ~/.leafcutter/LeafcutterLLM/rust
export LD_LIBRARY_PATH="$HOME/.leafcutter/llama.cpp/build/bin:$LD_LIBRARY_PATH"
cargo build --release --bin leafcutter

# 4. Install binary
cp target/release/leafcutter ~/.local/bin/
```

---

## Usage

### Standalone (without Cynapse)

```bash
# List downloaded models
leafcutter list-models --dir ~/models

# One-shot generation
leafcutter generate \
  --model ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --prompt "Explain quantum computing:" \
  --max-tokens 128 \
  --temperature 0.7

# Interactive chat
leafcutter chat \
  --model ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --system "You are a helpful coding assistant."

# HTTP API server (OpenAI-compatible)
leafcutter server \
  --model ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --port 8081
```

### With Cynapse

Cynapse downloads models to its workspace. Point Leafcutter at those models:

```bash
# Cynapse typically stores models here:
export MODELS_DIR="$HOME/.cynapse/workspace/models"

# List Cynapse-downloaded models
leafcutter list-models --dir "$MODELS_DIR"

# Chat with a Cynapse-downloaded model
leafcutter chat --model "$MODELS_DIR/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

**Future integration:** Cynapse's TUI will detect Leafcutter installation and offer a "Run with Leafcutter" option that bypasses llama-server entirely.

---

## Model Recommendations

For the best experience with Leafcutter + Cynapse:

| Use Case | Model | Size | Quant | RAM Needed |
|----------|-------|------|-------|------------|
| General chat | Llama-3.2-3B-Instruct | 3B | Q4_K_M | ~2.5 GB |
| Coding | Qwen2.5-Coder-3B | 3B | Q4_K_M | ~2.5 GB |
| Reasoning | DeepSeek-R1-Distill-Qwen-7B | 7B | Q4_K_M | ~5 GB |
| Fast/edge | Llama-3.2-1B-Instruct | 1B | Q4_0 | ~1 GB |
| Multilingual | Qwen2.5-7B-Instruct | 7B | Q4_K_M | ~5 GB |

Download any of these via Cynapse:
```bash
cynapse model download meta-llama/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

---

## Performance Tuning

### CPU Threads
```bash
# Auto-detect optimal thread count
leafcutter chat --model model.gguf --threads $(nproc)
```

### GPU Offloading (if you have a GPU)
```bash
# Offload all layers to GPU
leafcutter chat --model model.gguf --gpu-layers 99

# Offload just 10 layers (hybrid CPU/GPU)
leafcutter chat --model model.gguf --gpu-layers 10
```

### Context Size
```bash
# Large context for long conversations
leafcutter chat --model model.gguf --ctx-size 8192

# Small context for faster loading
leafcutter generate --model model.gguf --ctx-size 512 --prompt "Hi"
```

---

## Troubleshooting

### `error while loading shared libraries: libllama.so`
```bash
# Reload your shell profile
source ~/.bashrc  # or ~/.zshrc

# Or manually set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$HOME/.leafcutter/llama.cpp/build/bin:$LD_LIBRARY_PATH"
```

### Model fails to load
- Ensure the GGUF file is complete (not a partial download)
- Try a smaller context size: `--ctx-size 512`
- Check available RAM: `free -h`

### Slow generation
- Increase threads: `--threads $(nproc)`
- Try a smaller model (1B–3B)
- Enable GPU layers if you have a GPU: `--gpu-layers 99`

---

## Development

To update Leafcutter after initial install:

```bash
cd ~/.leafcutter/LeafcutterLLM
git pull origin main
cd rust
cargo build --release --bin leafcutter
cp target/release/leafcutter ~/.local/bin/
```

To update llama.cpp:
```bash
cd ~/.leafcutter/llama.cpp
git pull origin master
cd build && cmake --build . --parallel $(nproc)
```

---

*Leafcutter + Cynapse: Download once, run anywhere, own your AI.*
