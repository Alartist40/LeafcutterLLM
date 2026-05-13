#!/bin/bash
# benchmark_all_models.sh - Test all models in order

MODELS=(
  "tinyllama-1.1b:TinyLlama-1.1B"
  "qwen2-0.5b:Qwen2-0.5B"
  "mistral-7b-q4.gguf:Mistral-7B-Q4"
)

RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"

# Ensure server is built
if [ ! -f "./leafcutter-server" ]; then
    echo "🔨 Building leafcutter-server..."
    CGO_ENABLED=1 go build -o leafcutter-server ./cmd/server
fi

echo "🧪 Starting comprehensive Leafcutter testing..."
echo ""

PORT=8082

for MODEL_SPEC in "${MODELS[@]}"; do
  IFS=':' read -r MODEL_PATH MODEL_NAME <<< "$MODEL_SPEC"
  FULL_PATH="./models/$MODEL_PATH"
  
  if [ ! -e "$FULL_PATH" ]; then
    echo "⚠️  Skipping $MODEL_NAME (not found at $FULL_PATH)"
    continue
  fi
  
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Testing: $MODEL_NAME"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RESULT_FILE="$RESULTS_DIR/test_${MODEL_NAME}_${TIMESTAMP}.json"
  
  # Start server
  echo "Starting Leafcutter server..."
  ./leafcutter-server \
    --model "$FULL_PATH" \
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
          continue 2
      fi
  done
  
  # Run tests and collect metrics
  echo "Running benchmark..."
  curl -s -X POST http://localhost:"$PORT"/benchmark \
    -H "Content-Type: application/json" \
    -d '{
      "num_requests": 10,
      "context_tokens": 128,
      "batch_size": 4
    }' > "$RESULT_FILE"
  
  # Inject model name into JSON
  # This is a bit hacky, but works for the python script later
  sed -i "s|{|{\"model_name\": \"$MODEL_NAME\", \"hardware\": \"$(uname -a)\", |" "$RESULT_FILE"
  
  echo "Results saved to: $RESULT_FILE"
  jq . "$RESULT_FILE"
  
  # Cleanup
  kill $SERVER_PID
  sleep 2
  
  echo "✅ $MODEL_NAME complete"
  echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 All tests complete!"
echo "Results saved to: $RESULTS_DIR/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
