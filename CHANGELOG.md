# Changelog

All notable changes to AirLLM-Go will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Architectural Overhaul (Speculative Decoding + CGO Kernel)
*Date: 2026-04-21*

This release completely rearchitects `airllm-go` from a naive port of the Python library into a highly concurrent, C-accelerated production inference server. 

#### Added
- **KV Caching (`pkg/inference/engine.go`, `pkg/inference/layers.go`)**:
  - Implemented full KV Caching to reduce generation complexity from O(T) to O(1) for subsequent tokens.
  - Added `pastK` and `pastV` caching mechanisms within the attention layer (`scaledDotProductAttention`).
  - Added cache management in the Engine to clear context between entirely new prompts.
- **BPE Tokenizer (`pkg/tokenizer/tokenizer.go`)**:
  - Replaced the simple ASCII tokenizer stub with a full Byte-Pair Encoding (BPE) implementation.
  - Parses real HuggingFace `tokenizer.json` formats to decode actual subword tokens properly.
- **Speculative Decoding Engine (`pkg/inference/speculative.go`)**: 
  - Implemented parallel generation and verification pipelines utilizing Go channels and goroutines.
  - Supports running a fast "draft" model alongside a massive "target" model to dramatically increase tokens-per-second.
  - Implements rigorous acceptance/rejection sampling with temperature scaling.
- **Continuous Batching Scheduler (`pkg/server/scheduler.go`)**:
  - Replaced single-request processing with in-flight continuous batching.
  - Features a FIFO priority queue with configurable `MaxBatchSize` and `MaxWaitDuration`.
  - Automatically flushes partial batches to optimize hardware utilization without causing latency spikes for users.
- **CGO Math Kernels (`pkg/qkernel`)**:
  - Implemented pure C kernel (`qkernel.c`) for 4-bit quantized matrix multiplication (`q4_gemm`).
  - Added OpenBLAS bindings (`blas.go`) to accelerate single-precision (FP32/FP16) tensor operations using `cblas_sgemm`.
  - Added safe CGO bindings in `qkernel.go` that enforce memory boundaries using `runtime.KeepAlive`.
  - Solves the pure-Go O(N^3) math bottleneck, increasing tensor operations speed by multiple orders of magnitude.
- **HTTP Server (`cmd/server/main.go`)**:
  - Added an HTTP API server wrapper over the Continuous Batching scheduler.
  - Endpoints for generation (`/generate`) and metrics/monitoring (`/health`).
- **Containerization (`Dockerfile`)**:
  - Added a multi-stage Dockerfile to compile CGO binaries and sandbox execution away from the host OS.

#### Fixed (Pre-Architecture Overhaul)
- Fixed critical logic stubs in `embedLookup`, `scaledDotProductAttention`, and `layerNorm` which prevented the original engine from running.
- Patched severe goroutine leaks in the layer-prefetching pipeline.
- Fixed matrix dimensions for `matmul` to correctly support HuggingFace's `[out, in]` weight matrix layout.
- Fixed IEEE 754 bit-math calculations for float16 to float32 subnormal conversions.
- Fixed general N-D array transpose math.

#### Removed
- Removed the strict "Zero External Dependency" constraint explicitly for C standard libraries to support the new `qkernel`.

---

## [1.0.0-alpha] - 2025-01-05

*Initial alpha release prior to architectural overhaul.*

### Added
- Layer-by-layer inference engine.
- Concurrent prefetching.
- Safetensors support.
- 8-bit block-wise quantization and 4-bit NF4 quantization algorithms (Pure Go).
- Zero-copy tensor views.
- CLI application tool (`cmd/airllm`).