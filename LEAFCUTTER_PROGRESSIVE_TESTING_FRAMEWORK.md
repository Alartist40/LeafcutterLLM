# 🌿 LeafcutterLLM Model Testing Framework
## Progressive Testing Strategy for THE-PATHFINDER-EYE Robot

**Document Version:** 1.0  
**Date:** 2026-05-13  
**Target Audience:** Leafcutter Development Team  
**Purpose:** Define a systematic approach to test Leafcutter with progressively larger models, measure performance metrics, and validate robot integration requirements.

---

## 🎯 Objective

Test Leafcutter's core claims (layer splitting, BLAS acceleration, continuous batching) against real-world robotics constraints:

- **RAM constraint:** Pi Zero 2W (512MB), Pi 5 (8GB)
- **Latency constraint:** <200ms first token (robot responsiveness)
- **Throughput:** Multiple concurrent requests (sensor fusion + AI reasoning)
- **Reliability:** Graceful degradation under memory pressure

---

## 📊 Model Testing Progression

### Tier 1: Ultra-Lightweight (Testing Infrastructure)
**Goal:** Verify testing tools work, establish baseline metrics

| Model | Size | Params | Purpose | Pi Zero | Pi 5 | Comment |
|-------|------|--------|---------|---------|------|---------|
| **TinyLlama** | 500MB | 160M | Verify tools | ✅ | ✅ | Test runner validation |
| **Phi-2** | 1.4GB | 2.7B | Baseline speed | ✅ | ✅ | Fast inference reference |

**What We Measure:**
- ✅ Benchmark tool accuracy
- ✅ Metric collection pipeline
- ✅ CSV/JSON output format
- ✅ Graph generation working

---

### Tier 2: Small Models (Robot Feasibility)
**Goal:** Prove Leafcutter works on actual robot hardware

| Model | Size | Params | Purpose | Pi Zero | Pi 5 | RAM (4-bit) | Latency Target |
|-------|------|--------|---------|---------|------|-------------|-----------------|
| **TinyLlama-1.1B** | 2.2GB | 1.1B | Lightweight baseline | ✅ (+swap) | ✅ | ~500MB | <500ms |
| **Phi-3-mini** | 2GB | 3.8B | Accuracy boost | ⚠️ | ✅ | ~1GB | <800ms |
| **Qwen2-0.5B** | 400MB | 500M | Ultra-light vision prep | ✅ | ✅ | ~200MB | <200ms |

**What We Test:**
- Layer splitting effectiveness (verify 1-layer loading works)
- RAM usage vs claimed reduction
- First-token latency (<200ms goal for robot)
- Context window handling (2K tokens)
- Concurrent request handling (vision → brain → actions)

---

### Tier 3: Medium Models (Sweet Spot)
**Goal:** Find optimal size for robot + quality trade-off

| Model | Size | Params | Purpose | Pi Zero | Pi 5 | RAM (4-bit) | Latency Target |
|-------|------|--------|---------|---------|------|-------------|-----------------|
| **Mistral-7B-Instruct-Q4** | 4.3GB | 7B | Current robot brain | ❌ | ✅ | ~1.4GB | <200ms |
| **Qwen2-1.5B** | 900MB | 1.5B | Better reasoning | ✅ (+swap) | ✅ | ~600MB | <400ms |
| **Neural-Chat-7B** | 4GB | 7B | Conversation focus | ❌ | ✅ | ~1.4GB | <200ms |

**What We Test:**
- Full inference pipeline (vision input → LLM → action output)
- Multi-turn conversation memory
- Token budgeting under load
- Batch processing 4+ concurrent requests
- BLAS kernel performance

---

### Tier 4: Large Models (Stretch Goal)
**Goal:** Demonstrate Leafcutter's 11x RAM advantage

| Model | Size | Params | Purpose | Pi Zero | Pi 5 | RAM (4-bit) | Latency Target |
|-------|------|--------|---------|---------|------|-------------|-----------------|
| **Llama-2-13B-Q4** | 7GB | 13B | Advanced reasoning | ❌ | ⚠️ | ~2.5GB | <500ms |
| **Mistral-8x7B** | 26GB | 46B (MoE) | Expert model | ❌ | ❌ | ~12GB | Experimental |

**What We Test:**
- Layer-splitting proof of concept
- MoE routing efficiency
- Context window limits
- Fallback mechanisms

---

## 🔧 Testing Infrastructure

### Phase 1: Model Download Registry

**File:** `models/REGISTRY.md`

Create a structured list with download links and checksums:

```markdown
# LeafcutterLLM Model Registry

## Small Models (Testing)
- TinyLlama-1.1B-Chat-v1.0 (2.2GB)
  - HF Link: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
  - Format: Safetensors
  - Quantization: Not needed (already small)

- Qwen2-0.5B-Instruct (400MB)
  - HF Link: https://huggingface.co/Qwen/Qwen2-0.5B-Instruct
  - Format: Safetensors
  - GGUF: https://huggingface.co/bartowski/Qwen2-0.5B-Instruct-GGUF

## Medium Models (Robot Ready)
- Mistral-7B-Instruct-v0.2-Q4
  - HF Link: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
  - Format: GGUF (Q4_K_M recommended)
  - Size: 4.3GB
  - Recommended for Pi 5 (8GB RAM)
```

**Download Script:** `scripts/download_models.sh`

```bash
#!/bin/bash
# Download models for testing in progressive order

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

# Tier 1: Baseline
echo "📥 Downloading Tier 1 (Baseline)..."
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --cache-dir "$MODELS_DIR" \
  --local-dir "$MODELS_DIR/tinyllama-1.1b"

# Tier 2: Small
echo "📥 Downloading Tier 2 (Small)..."
huggingface-cli download Qwen/Qwen2-0.5B-Instruct \
  --cache-dir "$MODELS_DIR" \
  --local-dir "$MODELS_DIR/qwen2-0.5b"

# Tier 3: Medium
echo "📥 Downloading Tier 3 (Medium)..."
# (User manually downloads Mistral-7B-Q4 due to size)
echo "⚠️  Tier 3 models are large. Download manually:"
echo "   https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
```

---

### Phase 2: Benchmark Test Suite

**File:** `cmd/test-suite/main.go`

Extend existing benchmark with:

1. **Model-Agnostic Testing**
   - Load any model from `models/` directory
   - Auto-detect size, parameter count, quantization
   - Run standardized tests

2. **Metrics Collected:**
   - Peak RAM (during inference)
   - First-token latency (ms)
   - Throughput (tokens/sec)
   - Per-layer load time (ms)
   - Batch efficiency (%)
   - Concurrent request handling

3. **Test Scenarios:**

```go
// Scenario 1: Single Request
func TestSingleRequest(model string, prompt string) {
    start := time.Now()
    response, tokens := Infer(model, prompt)
    latency := time.Since(start)
    peakRAM := ReadMemStats()
    
    Report("Single Request", map[string]interface{}{
        "latency_ms": latency.Milliseconds(),
        "tokens_generated": tokens,
        "tokens_per_second": float64(tokens) / latency.Seconds(),
        "peak_ram_mb": peakRAM / 1024 / 1024,
    })
}

// Scenario 2: Concurrent Requests
func TestConcurrentRequests(model string, numRequests int) {
    var wg sync.WaitGroup
    results := make(chan Result, numRequests)
    startRAM := ReadMemStats()
    
    for i := 0; i < numRequests; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            prompt := fmt.Sprintf("Request %d: Analyze this sensor data", id)
            latency, tokens := TimeInference(model, prompt)
            results <- Result{id, latency, tokens}
        }(i)
    }
    
    wg.Wait()
    peakRAM := ReadMemStats() - startRAM
    
    Report("Concurrent Requests", map[string]interface{}{
        "num_requests": numRequests,
        "total_tokens": sumTokens(results),
        "avg_latency_ms": averageLatency(results),
        "peak_ram_increase_mb": peakRAM / 1024 / 1024,
    })
}

// Scenario 3: Long Context (Robot Memory)
func TestLongContext(model string, historyTokens int) {
    // Simulate robot remembering past sensor readings
    prompt := BuildContextPrompt(historyTokens)
    
    latency, tokens := TimeInference(model, prompt)
    peakRAM := ReadMemStats()
    
    Report("Long Context", map[string]interface{}{
        "context_tokens": historyTokens,
        "latency_ms": latency.Milliseconds(),
        "memory_degradation": CalculateDegradation(),
    })
}

// Scenario 4: Layer-Split Verification
func TestLayerSplitting(model string) {
    // Verify that only one layer is in RAM at a time
    maxSimultaneousLayers := MonitorLayerLoads()
    
    Report("Layer Splitting", map[string]interface{}{
        "max_concurrent_layers": maxSimultaneousLayers,
        "layer_splitting_working": maxSimultaneousLayers == 1,
    })
}
```

---

### Phase 3: Output & Analysis

**File:** `results/test_report.json`

```json
{
  "test_run": "2026-05-13_mistral-7b",
  "model": "Mistral-7B-Instruct-v0.2-Q4",
  "hardware": "Raspberry Pi 5 (8GB RAM)",
  "timestamp": "2026-05-13T14:30:00Z",
  
  "results": {
    "single_request": {
      "latency_ms": 185,
      "tokens_generated": 47,
      "tokens_per_second": 254,
      "peak_ram_mb": 1430
    },
    "concurrent_requests": {
      "num_requests": 4,
      "avg_latency_ms": 320,
      "batch_efficiency": 85,
      "peak_ram_increase_mb": 200
    },
    "long_context": {
      "context_tokens": 2048,
      "latency_ms": 450,
      "memory_degradation_percent": 12
    },
    "layer_splitting": {
      "max_concurrent_layers": 1,
      "layer_splitting_working": true
    }
  },
  
  "performance_vs_claim": {
    "latency_target_met": true,      // <200ms for first token
    "ram_reduction_actual": "75%",
    "concurrent_requests_supported": 4
  }
}
```

**Graph Generation:** `scripts/generate_graphs.py`

```python
import json
import matplotlib.pyplot as plt

# Load all test results
results = []
for file in glob("results/test_report_*.json"):
    with open(file) as f:
        results.append(json.load(f))

# Plot 1: Latency vs Model Size
models = [r['model'] for r in results]
latencies = [r['results']['single_request']['latency_ms'] for r in results]
plt.plot(models, latencies, marker='o')
plt.ylabel('First Token Latency (ms)')
plt.title('Leafcutter: Latency Across Models')
plt.axhline(y=200, color='r', linestyle='--', label='Robot Target (<200ms)')
plt.legend()
plt.savefig('latency_comparison.png')

# Plot 2: RAM Usage vs Model Size
model_sizes = [r['model'].size_gb for r in results]
ram_usage = [r['results']['single_request']['peak_ram_mb'] for r in results]
plt.plot(model_sizes, ram_usage, marker='s')
plt.ylabel('Peak RAM (MB)')
plt.xlabel('Model Size (GB)')
plt.title('Leafcutter: RAM Efficiency')
plt.savefig('ram_comparison.png')

# Plot 3: Throughput vs Concurrency
num_requests = range(1, 9)
throughputs = []
for n in num_requests:
    avg_tps = calculate_throughput_at_concurrency(n)
    throughputs.append(avg_tps)
plt.bar(num_requests, throughputs)
plt.ylabel('Tokens/Second')
plt.xlabel('Concurrent Requests')
plt.title('Leafcutter: Batching Efficiency')
plt.savefig('throughput_comparison.png')
```

---

## 🤖 Robot Integration Constraints

### Critical Considerations for Leafcutter Optimization

**THE-PATHFINDER-EYE runs on limited hardware with strict latency requirements. Leafcutter must be optimized for:**

---

### 1. LATENCY CONSTRAINT: <200ms First Token

**Why it matters:**
- Robot must respond to sensor input within 200ms (human perception threshold)
- If latency > 1s, robot feels sluggish and unresponsive
- Camera runs at 30 FPS = 33ms per frame. Response must come before next frame arrives

**Optimization required:**
- ✅ Speculative decoding ENABLED by default (3-4x speedup)
- ✅ Context cache MUST persist between requests (don't reload model)
- ✅ Pre-warm model on startup (don't load on first inference)
- ✅ Reduce context window if needed (2K tokens max, not 4K)

**Test:** Run with worst-case prompt:
```
"I see [sensor data]. Previous observations: [1K tokens of history]. What should I do?"
```
Measure time to first token. MUST be <200ms.

---

### 2. RAM CONSTRAINT: Pi Zero 2W (512MB)

**Why it matters:**
- Pi Zero 2W is the minimum target
- Running vision + Leafcutter + OS = ~512MB total
- Layer splitting is THE selling point

**Optimization required:**
- ✅ Verify layer splitting actually works (only 1 layer in RAM)
- ✅ Measure peak RAM with swap disabled (true constraint)
- ✅ Test on Pi Zero 2W in CI/CD pipeline
- ✅ Support context window override (`--max-ctx 1024`)

**Test:** On Pi Zero 2W with 256MB swap:
```bash
./leafcutter-server --model mistral-7b-q4.gguf --max-ctx 1024
# Must not exceed 512MB peak RAM (even with swap)
```

**Current issue:** Leafcutter claims layer splitting but hasn't been tested on Pi Zero 2W.

---

### 3. CONCURRENCY CONSTRAINT: Multiple Sensor Streams

**Why it matters:**
- Robot has:
  - Vision pipeline (30 FPS = 33ms between frames)
  - Voice input (concurrent STT)
  - Sensor fusion (IMU, encoders, ultrasonic)
  - Action execution (motor commands)

- All may want AI reasoning at the same time
- Continuous batching MUST handle 4-8 concurrent requests

**Optimization required:**
- ✅ Batch scheduler handles queue depth >= 64
- ✅ No request starvation (fairness algorithm)
- ✅ Graceful degradation if queue fills
- ✅ Measure latency degradation with load

**Test:** Simulate robot workload:
```bash
# 4 vision frames/sec requesting inference
# + 1 speech input
# + 2 sensor fusion queries
# = 7 concurrent requests

$ leafcutter-bench --concurrent-requests 7 --duration 60s
# Measure: avg latency, p99 latency, throughput
```

---

### 4. THROUGHPUT CONSTRAINT: Tokens/Second

**Why it matters:**
- Robot needs to generate action quickly
- ~20 tokens = typical action command ("move forward 30cm")
- At 100 tokens/sec = 200ms response (acceptable)
- At 500 tokens/sec = 40ms response (excellent)

**Optimization required:**
- ✅ BLAS kernels MUST be compiled with OpenBLAS (not fallback Go)
- ✅ Measure tokens/sec per model
- ✅ Report in benchmark output

**Test:**
```bash
./leafcutter-bench --model mistral-7b-q4.gguf
# Output should show: "Tokens/second: 254 (via OpenBLAS)"
# If it says "<50", fallback is being used (bad)
```

---

### 5. GRACEFUL DEGRADATION: Out-of-Memory Handling

**Why it matters:**
- Robot may run out of RAM under peak load
- System must not crash
- Should degrade gracefully (slower responses, not error)

**Optimization required:**
- ✅ Implement context window auto-reduction
- ✅ Add memory pressure warning
- ✅ Queue requests instead of rejecting them
- ✅ Emit metrics for monitoring

**Test:**
```bash
# Simulate memory pressure
./leafcutter-server --model mistral-7b-q4.gguf --memory-limit 800MB
# Send 10 concurrent requests with 2K context each
# Expected: latency increases, no crashes, no dropped requests
```

---

## 📋 Testing Checklist

### Before Each Tier Release:

- [ ] **Model Downloads Working**
  - [ ] All models in registry downloadable
  - [ ] Checksums verified
  - [ ] Formats detected correctly

- [ ] **Benchmark Tool Working**
  - [ ] Compiles without errors
  - [ ] Runs all test scenarios
  - [ ] JSON output valid
  - [ ] Metrics within expected range

- [ ] **Single Request Tests**
  - [ ] Latency <200ms for Tier 2-3 models
  - [ ] Tokens generated > 0
  - [ ] Peak RAM < (model_size * 1.2)
  - [ ] Deterministic (same input = same output)

- [ ] **Concurrent Request Tests**
  - [ ] Batch scheduler handles 4+ requests
  - [ ] No request drops
  - [ ] Latency scales sub-linearly
  - [ ] RAM increase < (additional_layers * layer_size)

- [ ] **Layer Splitting Verification**
  - [ ] Monitor shows only 1 layer in RAM
  - [ ] Peak RAM matches claimed reduction
  - [ ] No hidden buffers eating RAM
  - [ ] Works on Pi Zero 2W (if testing there)

- [ ] **Robot Integration Ready**
  - [ ] HTTP API responds <100ms
  - [ ] Tokenizer integration works
  - [ ] Error handling doesn't crash
  - [ ] Graceful degradation under load

---

## 🚀 Execution Timeline

### Week 1: Infrastructure
- [ ] Set up model registry
- [ ] Implement benchmark test suite
- [ ] Create automated testing pipeline
- [ ] Generate baseline metrics

### Week 2: Tier 1-2 Testing
- [ ] Test TinyLlama, Phi-2, Qwen2-0.5B
- [ ] Verify metrics collection working
- [ ] Generate graphs
- [ ] Identify any benchmark bugs

### Week 3: Tier 3 Testing
- [ ] Test Mistral-7B, Qwen2-1.5B on Pi 5
- [ ] Verify <200ms latency target
- [ ] Measure concurrent request performance
- [ ] Test on actual Pi Zero 2W (if available)

### Week 4: Optimization & Documentation
- [ ] Fix any identified issues
- [ ] Optimize for robot constraints
- [ ] Write performance report
- [ ] Integrate with THE-PATHFINDER-EYE

---

## 📊 Success Criteria

### Leafcutter Must Prove:

| Claim | Test | Pass Criteria |
|-------|------|---------------|
| Layer splitting saves 75% RAM | Measure peak during inference | Actual ≥ 70% reduction |
| Latency <200ms on 7B | Time first token on Pi 5 | Actual <200ms |
| Handles 4+ concurrent | Batch 4 requests | No drops, avg latency <400ms |
| Works on Pi Zero 2W | Run mistral-7b on Pi Zero | Peak RAM <512MB |
| BLAS 13x speedup | Compare vs Go matmul | Actual ≥10x speedup |
| Throughput 100+ tokens/sec | Measure tokens/sec | Actual ≥100 tokens/sec |

---

## 📝 Deliverables

1. **model_registry.md** — List of all tested models with download links
2. **benchmark_results.json** — Raw metrics from all test runs
3. **performance_graphs.png** — Latency vs Model Size, RAM vs Model Size, Throughput vs Concurrency
4. **robot_integration_report.md** — Recommendations for THE-PATHFINDER-EYE
5. **optimization_guide.md** — How Leafcutter can better serve robotics use cases

---

## 🎯 Recommendation to Team

**Leafcutter is being tested in a REAL robot with hard constraints.**

This is not academic benchmarking. This is:
- Latency critical (robot responsiveness)
- Memory critical (embedded hardware)
- Throughput critical (multi-sensor fusion)

**Every optimization matters.** Focus on:
1. **Speculative decoding** — Most impactful for latency
2. **Layer splitting verification** — Core selling point
3. **Concurrent batching** — Required for robot workload
4. **Graceful degradation** — Must not crash under memory pressure

The robot will expose every weakness in Leafcutter's architecture. Make sure it's battle-tested before THE-PATHFINDER-EYE ships. 🤖

