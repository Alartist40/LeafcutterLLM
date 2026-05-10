# 📁 Models Directory

**Place your LLM models here for LeafcutterLLM to detect automatically.**

## Supported Formats

### 1. HuggingFace Safetensors (Directory)
```
models/
└── llama-7b/
    ├── config.json
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    └── model-00003-of-00003.safetensors
```

### 2. GGUF Single File
```
models/
└── llama-7b-q4.gguf
```

## Quick Start

1. Download a model from HuggingFace or llama.cpp
2. Place it in this directory
3. Run: `leafcutter-server` (auto-detects)
4. Or specify: `leafcutter-server --model models/your-model`

## Recommended Models for Different Hardware

| Hardware | Model | Size |
|----------|-------|------|
| Pi Zero 2W (512MB) | TinyLlama-1.1B-Q4 | ~600MB |
| Pi 5 (4GB) | Qwen2-1.5B-Q4 | ~1GB |
| Pi 5 (8GB) | LLaMA-7B-Q4 | ~4GB |
| Laptop (16GB) | LLaMA-13B-Q4 | ~8GB |

## Hardware Compatibility

LeafcutterLLM will automatically check if your system can run the model:
- ✅ Green: Model fits comfortably
- ⚠️ Yellow: Model fits but tight on memory
- ❌ Red: Model too large, reduce quantization or use smaller model

Run `leafcutter-server --check-only` to test without loading.
