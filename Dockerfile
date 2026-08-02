# Dockerfile for LeafcutterLLM — safe sandboxed testing environment.
#
# Multi-stage build: builder compiles the `leafcutter` CLI, runtime is a
# minimal image with just the binary + models volume.
#
# Build:
#   docker build -t leafcutter .
# Run (mount your models dir):
#   docker run -it -v ~/Downloads/models:/models leafcutter list
#   docker run -it -v ~/Downloads/models:/models leafcutter run Ministral-3-3B
#
# The container ships WITHOUT models — mount them at runtime. The binary is
# pure native (no GPU/wgpu requirement at runtime), so it runs in containers
# and falls back to the CPU tier automatically.

# ── stage 1: builder ─────────────────────────────────────────────
FROM rust:1.97-slim AS builder

WORKDIR /build

# Install git (needed if Cargo pulls git deps) and pkg-config
RUN apt-get update && apt-get install -y --no-install-recommends \
    git pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy the Rust project
COPY rust/ ./rust/

# Build the leafcutter CLI in release mode (no FFI — pure native)
WORKDIR /build/rust
RUN cargo build --release --no-default-features --bin leafcutter

# ── stage 2: runtime ─────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

# Minimal runtime deps: ca-certificates for any HTTPS model downloads,
# libgcc for the Rust binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the leafcutter binary
COPY --from=builder /build/rust/target/release/leafcutter /usr/local/bin/leafcutter

# Default models directory inside the container — mount your host
# models dir here at runtime: -v ~/Downloads/models:/models
RUN mkdir -p /models
ENV LEAF_MODELS_DIR=/models

# Default entrypoint is the leafcutter CLI
ENTRYPOINT ["leafcutter"]
CMD ["help"]
