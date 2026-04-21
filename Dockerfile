# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# Uses the official Go image which already has gcc (for CGO).
# ──────────────────────────────────────────────────────────────────────────────
FROM golang:1.21-bookworm AS builder

# Install gcc and standard C library headers (needed for CGO / qkernel.c).
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        pkg-config \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy dependency files first to leverage Docker layer caching.
COPY go.mod ./
RUN go mod download || true

# Copy full source tree.
COPY . .

# Build the inference server binary.
# CGO_ENABLED=1 is required for the qkernel package.
# -trimpath removes local filesystem paths from the binary.
RUN CGO_ENABLED=1 GOOS=linux go build \
        -trimpath \
        -ldflags="-s -w" \
        -o /bin/airllm-server \
        ./cmd/server

# Build the CLI binary as well.
RUN CGO_ENABLED=1 GOOS=linux go build \
        -trimpath \
        -ldflags="-s -w" \
        -o /bin/airllm \
        ./cmd/airllm

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime
# Minimal Debian image — only what the CGO binary needs at runtime.
# ──────────────────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

# CGO binaries need the C standard library at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libc6 \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security — C crashes are sandboxed within the container.
RUN useradd -r -u 1001 -s /sbin/nologin airllm
USER airllm

# Copy compiled binaries from the builder stage.
COPY --from=builder /bin/airllm-server /usr/local/bin/airllm-server
COPY --from=builder /bin/airllm        /usr/local/bin/airllm

# Model weights are mounted at runtime — not baked into the image.
VOLUME ["/models"]

EXPOSE 8080

# Default: run the inference server.
# Override with `docker run ... airllm --model /models/...` for CLI mode.
ENTRYPOINT ["airllm-server"]
CMD ["--port", "8080", "--batch-size", "8", "--model", "/models/target", "--help"]
