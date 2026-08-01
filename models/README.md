# 📁 Models Directory

**Place your LLM models here for LeafcutterLLM to detect automatically.**

## Supported Formats

### 1. GGUF Single File (recommended — native engine)
```
models/
└── ornith-1.0-9b-Q4_K_M.gguf
```

Run with the native streaming chat REPL:
```bash
leafcutter run ornith            # fuzzy name match
leafcutter run models/ornith-1.0-9b-Q4_K_M.gguf   # direct path
```

### 2. HuggingFace Safetensors (Directory — safetensor backend)
```
models/
└── llama-7b/
    ├── config.json
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    └── model-00003-of-00003.safetensors
```

## Quick Start

1. Download a model from HuggingFace or llama.cpp
2. Place it in this directory
3. Run: `leafcutter list` (auto-detects `./models` or `~/Downloads/models`)
4. Run: `leafcutter run <name>`

## Recommended Models for Different Hardware

| Hardware | Model | Size |
|----------|-------|------|
| Pi Zero 2W (512MB) | TinyLlama-1.1B-Q4 | ~600MB |
| Pi 5 (4GB) | Qwen2-1.5B-Q4 | ~1GB |
| Pi 5 (8GB) | LLaMA-7B-Q4 | ~4GB |
| Laptop (16GB) | Ornith-9B-Q4_K_M (native, ~8 GB peak) | 5.3GB |

## Hardware Compatibility

LeafcutterLLM will automatically check if your system can run the model:
- ✅ Green: Model fits comfortably
- ⚠️ Yellow: Model fits but tight on memory
- ❌ Red: Model too large, reduce quantization or use smaller model

Run `leafcutter list` to see which models are detected and their sizes.
