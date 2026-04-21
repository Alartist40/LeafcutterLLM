# AirLLM-Go Server

A high-performance Go LLM inference server featuring **Speculative Decoding** and **Continuous Batching**, leveraging CGO for native 4-bit quantization math.

![Status](https://img.shields.io/badge/Status-Beta-orange.svg)
![Go Version](https://img.shields.io/badge/Go-1.21+-00ADD8.svg)

## Overview

AirLLM-Go has been completely architected for extreme concurrency and throughput. Standard LLM servers suffer from the Python GIL and sequential processing bottlenecks. This engine solves that by combining Go's goroutines with a custom C-based 4-bit math kernel to achieve native hardware speeds without heavy dependencies like PyTorch or llama.cpp.

### Key Architectural Features

1. **Speculative Decoding (Parallel Generation & Verification)**
   - Runs a fast, small "draft" model to guess future tokens.
   - Concurrently runs a massive "target" model to verify batches of draft tokens in a single forward pass.
   - Go Channels seamlessly pipeline the data between the models, providing true multi-core parallelization impossible in Python.
2. **Continuous Batching (In-Flight Batching)**
   - Incoming user requests enter a priority FIFO queue.
   - A background dispatcher groups ready requests into a single hardware batch.
   - Prevents the "stop-the-world" effect of sequential request processing.
4. **KV Caching (O(1) Generation)**
   - Tracks `pastK` and `pastV` tensors across network layers for sequential token generation.
   - Reduces transformer computational complexity from O(T) to O(1) by avoiding redundant history calculation.
5. **CGO Native 4-bit Kernel (`qkernel`)**
   - 4-bit Quantized Matrix Multiplication (GEMM) written in pure C.
   - Avoids the overhead of pure Go math loops for tensor operations.
   - Implements in-register nibble unpacking to prevent memory bandwidth bottlenecks.
   - Strict memory safety boundaries utilizing Go's `runtime.KeepAlive` and C NULL guards.

## Installation & Setup

Because this project utilizes CGO for high-performance math, it is recommended to run it via Docker to isolate the C runtime, or ensure you have `gcc` installed.

### Option 1: Docker (Recommended)

A multi-stage `Dockerfile` is included to build and sandbox the engine safely.

```bash
git clone https://github.com/xander/airllm-go.git
cd airllm-go

# Build the container
docker build -t airllm-server .

# Run the container (mounting your local models directory)
docker run -p 8080:8080 -v /path/to/local/models:/models airllm-server \
    --model /models/target-70b \
    --draft /models/draft-300m \
    --speculative \
    --batch-size 8 \
    --port 8080
```

### Option 2: Build From Source

You must have `gcc` and Go 1.21+ installed.

```bash
cd airllm-go

# Build the HTTP inference server
CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o airllm-server ./cmd/server

# Build the CLI debugging tool
CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o airllm ./cmd/airllm
```

## Running the Server

Start the API server with Speculative Decoding enabled:

```bash
./airllm-server \
    --model ./models/llama-2-70b \
    --draft ./models/llama-2-small \
    --speculative \
    --batch-size 16 \
    --queue-depth 512 \
    --port 8080
```

### Server Endpoints

**1. Generate Text**
`POST /generate`

```bash
curl -X POST http://localhost:8080/generate \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "The capital of France is",
       "max_tokens": 50,
       "temperature": 0.7
     }'
```

**2. Health & Metrics**
`GET /health`

```bash
curl http://localhost:8080/health
# Returns queue depth, dropped requests, batch stats
```

## Code Structure

```text
airllm-go/
├── cmd/
│   ├── airllm/          # Interactive CLI mode
│   └── server/          # Production HTTP server (Scheduler & API)
├── pkg/
│   ├── qkernel/         # CGO 4-bit Matrix Math (qkernel.c, qkernel.go)
│   ├── inference/       # Tensor layers and SpeculativeEngine (speculative.go)
│   ├── server/          # Continuous Batching Scheduler (scheduler.go)
│   ├── tensor/          # Go tensor structs and memory layout
│   └── model/           # Safetensors weight loading
└── Dockerfile           # Secure runtime sandbox
```

## Safety Guarantees

Working with C memory from Go requires strict safety boundaries. AirLLM-Go implements:
- Slice capacity validation in Go before any pointer crosses the CGO boundary.
- `runtime.KeepAlive` calls following every C invocation to protect against aggressive Go Garbage Collection.
- Internal pointer `NULL` guards inside `qkernel.c`.

## Disclaimer

This is a high-performance inference engine built for research and low-level hardware utilization. It expects safetensor formats. While it integrates a real HuggingFace BPE tokenizer, the current version is highly optimized for performance testing rather than robust edge-case validation.
