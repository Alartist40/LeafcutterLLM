# ─── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM golang:1.22-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        pkg-config \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY go.mod ./
RUN go mod download || true
COPY . .

RUN CGO_ENABLED=1 GOOS=linux go build \
        -trimpath \
        -ldflags="-s -w" \
        -o /bin/leafcutter-server \
        ./cmd/server

RUN CGO_ENABLED=1 GOOS=linux go build \
        -trimpath \
        -ldflags="-s -w" \
        -o /bin/airllm \
        ./cmd/airllm

RUN CGO_ENABLED=1 GOOS=linux go build \
        -trimpath \
        -ldflags="-s -w" \
        -o /bin/leafcutter-tui \
        ./cmd/tui

RUN CGO_ENABLED=1 GOOS=linux go build \
        -trimpath \
        -ldflags="-s -w" \
        -o /bin/leafcutter-bench \
        ./cmd/benchmark

# ─── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libc6 \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -u 1001 -s /sbin/nologin leafcutter
USER leafcutter

COPY --from=builder /bin/leafcutter-server /usr/local/bin/leafcutter-server
COPY --from=builder /bin/airllm            /usr/local/bin/airllm
COPY --from=builder /bin/leafcutter-tui    /usr/local/bin/leafcutter-tui
COPY --from=builder /bin/leafcutter-bench  /usr/local/bin/leafcutter-bench

VOLUME ["/models"]
EXPOSE 8080

ENTRYPOINT ["leafcutter-server"]
CMD ["--port", "8080", "--batch-size", "8", "--model", "/models/target"]
