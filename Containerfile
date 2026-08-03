# ─── Stage 1: Rust Builder ─────────────────────────────────────────────────────
FROM rust:1.97-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY rust/Cargo.toml rust/Cargo.lock ./
COPY rust/src ./src
COPY rust/.cargo ./.cargo

# Build native-only server (no llama.cpp FFI required)
RUN cargo build --release --no-default-features --bin leafcutter

# ─── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -u 1001 -s /sbin/nologin leafcutter
USER leafcutter

COPY --from=builder /src/target/release/leafcutter /usr/local/bin/leafcutter

VOLUME ["/models"]
EXPOSE 8081

ENV LEAF_MODELS_DIR=/models

ENTRYPOINT ["leafcutter"]
CMD ["serve", "--port", "8081", "--host", "0.0.0.0"]
