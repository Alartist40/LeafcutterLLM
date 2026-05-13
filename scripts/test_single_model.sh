#!/bin/bash
# test_single_model.sh - Test one model

MODEL_PATH=$1
MODEL_NAME=$(basename "$MODEL_PATH")
PORT=${2:-8081}

if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 <path/to/model> [port]"
  echo "Example: $0 ./models/tinyllama-1.1b 8081"
  exit 1
fi

# Ensure server is built
if [ ! -f "./leafcutter-server" ]; then
    echo "🔨 Building leafcutter-server..."
    CGO_ENABLED=1 go build -o leafcutter-server ./cmd/server
fi

echo "🧪 Testing $MODEL_NAME on port $PORT..."
echo ""

# Start Leafcutter server
echo "Starting Leafcutter server..."
./leafcutter-server \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  > /tmp/leafcutter.log 2>&1 &

SERVER_PID=$!

# Wait for server to start
MAX_RETRIES=30
RETRY_COUNT=0
while ! curl -s http://localhost:"$PORT"/health > /dev/null; do
    sleep 1
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
        echo "❌ Server failed to start. Check /tmp/leafcutter.log"
        kill $SERVER_PID
        exit 1
    fi
done

echo "✅ Server started."

# Test single request
echo "Testing single request..."
curl -s -X POST http://localhost:"$PORT"/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning? Answer briefly.",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq .

# Run benchmark
echo ""
echo "Running performance benchmark..."
curl -s -X POST http://localhost:"$PORT"/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "num_requests": 10,
    "context_tokens": 128,
    "batch_size": 4
  }' | jq .

# Cleanup
kill $SERVER_PID
echo ""
echo "✅ Test complete"
