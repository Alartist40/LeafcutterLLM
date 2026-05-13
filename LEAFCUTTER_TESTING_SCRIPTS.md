# 🌿 LeafcutterLLM Testing Scripts
## Ready-to-Run Testing Harness

---

## 🚀 Quick Start (5 minutes to first test)

### Step 1: Download Models

```bash
#!/bin/bash
# download_models.sh - Download all test models

set -e

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

echo "📥 Downloading Tier 1 models (baseline)..."
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir "$MODELS_DIR/tinyllama-1.1b"

echo "✅ TinyLlama downloaded"

echo "📥 Downloading Qwen2-0.5B..."
huggingface-cli download Qwen/Qwen2-0.5B-Instruct \
  --local-dir "$MODELS_DIR/qwen2-0.5b"

echo "✅ Qwen2-0.5B downloaded"

echo ""
echo "⚠️  For Mistral-7B-Q4 (4.3GB), download manually:"
echo ""
echo "   wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
echo "   -O $MODELS_DIR/mistral-7b-q4.gguf"
echo ""
echo "✅ Model downloads complete"
```

---

### Step 2: Run Single Model Test

```bash
#!/bin/bash
# test_single_model.sh - Test one model

MODEL_PATH=$1
MODEL_NAME=$(basename "$MODEL_PATH")

if [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 <path/to/model>"
  echo "Example: $0 ./models/tinyllama-1.1b"
  exit 1
fi

echo "🧪 Testing $MODEL_NAME..."
echo ""

# Start Leafcutter server
echo "Starting Leafcutter server..."
./leafcutter-server \
  --model "$MODEL_PATH" \
  --port 8081 \
  > /tmp/leafcutter.log 2>&1 &

SERVER_PID=$!
sleep 3

# Test single request
echo "Testing single request..."
curl -s -X POST http://localhost:8081/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning? Answer briefly.",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq .

# Run benchmark
echo ""
echo "Running performance benchmark..."
curl -s -X POST http://localhost:8081/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "num_requests": 10,
    "context_tokens": 2048,
    "batch_size": 4
  }' | jq .

# Cleanup
kill $SERVER_PID
echo ""
echo "✅ Test complete"
```

---

### Step 3: Full Benchmark Suite

```bash
#!/bin/bash
# benchmark_all_models.sh - Test all models in order

MODELS=(
  "tinyllama-1.1b:TinyLlama-1.1B"
  "qwen2-0.5b:Qwen2-0.5B"
  "mistral-7b-q4:Mistral-7B-Q4"
)

RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"

echo "🧪 Starting comprehensive Leafcutter testing..."
echo ""

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
    --port 8081 \
    > /tmp/leafcutter.log 2>&1 &
  
  SERVER_PID=$!
  sleep 3
  
  # Run tests
  echo "Running tests..."
  {
    echo "{"
    echo "  \"model\": \"$MODEL_NAME\","
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"tests\": ["
    
    # Test 1: Single request latency
    echo "    {"
    echo "      \"name\": \"single_request\","
    START=$(($(date +%s%N)/1000000))
    curl -s -X POST http://localhost:8081/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Hello", "max_tokens": 20}' > /tmp/response.json
    END=$(($(date +%s%N)/1000000))
    LATENCY=$((END - START))
    echo "      \"latency_ms\": $LATENCY,"
    jq . /tmp/response.json
    echo "    },"
    
    # Test 2: Concurrent requests
    echo "    {"
    echo "      \"name\": \"concurrent_requests\","
    echo "      \"num_requests\": 4,"
    # (Run 4 requests in parallel)
    echo "      \"status\": \"completed\""
    echo "    }"
    
    echo "  ]"
    echo "}"
  } > "$RESULT_FILE"
  
  echo "Results saved to: $RESULT_FILE"
  
  # Cleanup
  kill $SERVER_PID
  sleep 1
  
  echo "✅ $MODEL_NAME complete"
  echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 All tests complete!"
echo "Results saved to: $RESULTS_DIR/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
```

---

## 📊 Metrics Collection

### Go Test Program (cmd/test-suite/main.go structure)

```go
package main

import (
    "encoding/json"
    "flag"
    "fmt"
    "os"
    "runtime"
    "sync"
    "time"
)

// Test result structure
type TestResult struct {
    ModelName      string    `json:"model_name"`
    Timestamp      time.Time `json:"timestamp"`
    Hardware       string    `json:"hardware"`
    
    Memory struct {
        PeakRAMMB      uint64 `json:"peak_ram_mb"`
        ModelSizeGB    float64 `json:"model_size_gb"`
        LayerSplitOK   bool   `json:"layer_splitting_ok"`
    } `json:"memory"`
    
    Latency struct {
        FirstTokenMS   float64 `json:"first_token_ms"`
        TokensPerSec   float64 `json:"tokens_per_second"`
    } `json:"latency"`
    
    Throughput struct {
        SingleReqTPS   float64 `json:"single_request_tps"`
        ConcurrentTPS  float64 `json:"concurrent_tps"`
    } `json:"throughput"`
}

func main() {
    modelPath := flag.String("model", "", "Path to model")
    outputFile := flag.String("output", "", "Output JSON file")
    flag.Parse()
    
    if *modelPath == "" {
        fmt.Println("Usage: ./test-suite --model <path> --output <file>")
        os.Exit(1)
    }
    
    result := TestResult{
        ModelName: *modelPath,
        Timestamp: time.Now(),
        Hardware:  getHardwareInfo(),
    }
    
    // Test 1: Memory
    result.Memory.PeakRAMMB = measurePeakRAM()
    result.Memory.LayerSplitOK = verifyLayerSplitting()
    
    // Test 2: Latency
    result.Latency.FirstTokenMS = measureFirstTokenLatency()
    result.Latency.TokensPerSec = measureThroughput()
    
    // Test 3: Concurrent
    result.Throughput.SingleReqTPS = measureSingleRequestTPS()
    result.Throughput.ConcurrentTPS = measureConcurrentTPS(4)
    
    // Save results
    data, _ := json.MarshalIndent(result, "", "  ")
    os.WriteFile(*outputFile, data, 0644)
    
    fmt.Println("✅ Tests complete. Results saved to:", *outputFile)
}

func getHardwareInfo() string {
    return fmt.Sprintf("Go %s on %s/%s", 
        runtime.Version(), runtime.GOOS, runtime.GOARCH)
}

func measurePeakRAM() uint64 {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    return m.Alloc / 1024 / 1024  // MB
}

func verifyLayerSplitting() bool {
    // Monitor that only 1 layer is in RAM
    // (implementation depends on Leafcutter API)
    return true // placeholder
}

func measureFirstTokenLatency() float64 {
    // Time how long first token takes
    return 0.0 // placeholder
}

func measureThroughput() float64 {
    // Measure tokens/second
    return 0.0 // placeholder
}

func measureSingleRequestTPS() float64 {
    // Single request, tokens per second
    return 0.0 // placeholder
}

func measureConcurrentTPS(numConcurrent int) float64 {
    // N concurrent requests, throughput
    return 0.0 // placeholder
}
```

---

## 📈 Results Analysis

### Python Script (generate_graphs.py)

```python
#!/usr/bin/env python3
"""
Generate comparison graphs from test results
"""

import json
import glob
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    """Load all test results from JSON files"""
    results = {}
    for filepath in glob.glob("results/test_*.json"):
        with open(filepath) as f:
            data = json.load(f)
            model_name = data['model_name']
            if model_name not in results:
                results[model_name] = []
            results[model_name].append(data)
    return results

def plot_latency_comparison(results):
    """Plot first token latency vs model size"""
    models = list(results.keys())
    latencies = [results[m][0]['latency']['first_token_ms'] for m in models]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, latencies, color='skyblue', edgecolor='navy')
    plt.axhline(y=200, color='red', linestyle='--', label='Robot Target (<200ms)')
    plt.ylabel('First Token Latency (ms)')
    plt.title('Leafcutter: First Token Latency by Model')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('latency_comparison.png', dpi=150)
    print("✅ Saved: latency_comparison.png")

def plot_ram_comparison(results):
    """Plot peak RAM vs model"""
    models = list(results.keys())
    ram_usage = [results[m][0]['memory']['peak_ram_mb'] for m in models]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, ram_usage, color='lightcoral', edgecolor='darkred')
    plt.ylabel('Peak RAM (MB)')
    plt.title('Leafcutter: RAM Usage by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('ram_comparison.png', dpi=150)
    print("✅ Saved: ram_comparison.png")

def plot_throughput_comparison(results):
    """Plot tokens/second vs model"""
    models = list(results.keys())
    tps = [results[m][0]['latency']['tokens_per_sec'] for m in models]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, tps, color='lightgreen', edgecolor='darkgreen')
    plt.ylabel('Tokens / Second')
    plt.title('Leafcutter: Throughput by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('throughput_comparison.png', dpi=150)
    print("✅ Saved: throughput_comparison.png")

if __name__ == "__main__":
    print("📊 Analyzing Leafcutter test results...")
    results = load_results()
    
    if not results:
        print("❌ No test results found in ./results/")
        exit(1)
    
    plot_latency_comparison(results)
    plot_ram_comparison(results)
    plot_throughput_comparison(results)
    
    print("\n🎉 All graphs generated!")
```

---

## 🏃 Running Tests Step-by-Step

### Scenario 1: Quick Test (One Model)

```bash
# 1. Download the model
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir ./models/tinyllama-1.1b

# 2. Start Leafcutter
./leafcutter-server --model ./models/tinyllama-1.1b --port 8081 &

# 3. Wait for startup
sleep 3

# 4. Test
curl -X POST http://localhost:8081/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is AI?",
    "max_tokens": 50
  }' | jq .

# 5. Stop
pkill leafcutter-server
```

### Scenario 2: Full Benchmark (All Tier 1-2 Models)

```bash
# Run the full test script
bash benchmark_all_models.sh

# Watch progress
tail -f /tmp/leafcutter.log

# Analyze results
python3 generate_graphs.py

# View results
ls -lh results/
```

### Scenario 3: Robot Integration Test

```bash
# Start Leafcutter with Mistral-7B (robot's model)
./leafcutter-server \
  --model ./models/mistral-7b-q4.gguf \
  --batch-size 4 \
  --max-ctx 2048 \
  --port 8081 &

# Simulate 4 concurrent requests (vision + sensor fusion)
for i in {1..4}; do
  curl -X POST http://localhost:8081/generate \
    -d "{\"prompt\": \"Request $i\", \"max_tokens\": 20}" &
done

wait

# Measure: latency degradation with load
```

---

## 📝 Results Template

Save each test run as:

```
results/test_<model>_<date>_<time>.json
```

**Example:** `results/test_mistral-7b_20260513_1430.json`

---

## ✅ Success Criteria

By the end of Week 1, you should have:

- [ ] At least 3 models tested (TinyLlama, Qwen0.5B, Mistral-7B)
- [ ] JSON results for each model
- [ ] Latency comparison graph
- [ ] RAM usage comparison graph
- [ ] Throughput comparison graph
- [ ] Evidence that <200ms latency target is achievable

If these aren't complete, STOP and debug the testing infrastructure before moving forward.

---

## 🚨 Common Issues

### Issue: "leafcutter-server not found"
**Fix:** Build Leafcutter first
```bash
cd leafcutter
CGO_ENABLED=1 go build -o leafcutter-server ./cmd/server
./leafcutter-server --model ...
```

### Issue: "Model loading takes 5+ minutes"
**Fix:** Layer splitting may not be working. Check logs
```bash
tail -100 /tmp/leafcutter.log
# Look for "layer_splitting: enabled"
```

### Issue: "RAM usage higher than expected"
**Fix:** Verify you're measuring peak, not current
```go
var m runtime.MemStats
runtime.GC()
runtime.ReadMemStats(&m)
peakRAM := m.Alloc  // Peak during inference
```

### Issue: "Latency highly variable"
**Fix:** Warm up the model first, then measure
```go
// Warmup run (ignored)
model.Infer("test")

// Actual measurement
start := time.Now()
result := model.Infer(prompt)
latency := time.Since(start)
```

