# 🌿 LeafcutterLLM Model Testing Lineup
## Small to Large: Progressive Testing Strategy

**Document Version:** 1.0  
**Created:** 2026-05-13  
**Purpose:** Curated list of models for progressive Leafcutter testing  

---

## ⚡ Quick Reference Table

| Tier | Model Name | Size | Params | Type | Latency* | RAM** | Pi Zero | Pi 5 | Notes |
|------|-----------|------|--------|------|----------|-------|---------|------|-------|
| **1A** | TinyLlama-1.1B | 2.2GB | 1.1B | Chat | ~200ms | 500MB | ✅ | ✅ | Baseline - verify tools work |
| **1B** | Phi-2 | 1.4GB | 2.7B | Instruct | ~150ms | 600MB | ✅ | ✅ | Fast inference reference |
| **2A** | Qwen2-0.5B | 400MB | 500M | Chat | ~100ms | 200MB | ✅ | ✅ | Lightest option |
| **2B** | Phi-3-mini | 2GB | 3.8B | Instruct | ~250ms | 1GB | ⚠️ | ✅ | Better quality than tiny |
| **2C** | Neural-Chat-3B | 1.6GB | 3B | Chat | ~180ms | 800MB | ✅ | ✅ | Conversation optimized |
| **3A** | Qwen2-1.5B | 900MB | 1.5B | Chat | ~350ms | 600MB | ✅ | ✅ | Good speed/quality ratio |
| **3B** | Mistral-7B-Q4 | 4.3GB | 7B | Instruct | ~185ms | 1.4GB | ❌ | ✅ | **Robot current** |
| **3C** | Neural-Chat-7B | 4GB | 7B | Chat | ~200ms | 1.4GB | ❌ | ✅ | Chat focused 7B |
| **4A** | Llama-2-13B-Q4 | 7GB | 13B | Instruct | ~400ms | 2.5GB | ❌ | ⚠️ | Stretch goal |
| **4B** | Mixtral-8x7B-Q4 | 26GB | 46B | Instruct | ~800ms | 12GB | ❌ | ❌ | Very large, MoE |

**Latency* = First token time on Pi 5 (estimated)  
**RAM** = Peak RAM during inference (4-bit quantization)

---

## 📥 TIER 1: Baseline Testing (Ultra-Lightweight)

### 1A. TinyLlama-1.1B-Chat

**Primary Purpose:** Verify testing infrastructure works

```bash
# Download
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir ./models/tinyllama-1.1b

# Run
./leafcutter-server --model ./models/tinyllama-1.1b

# Test
curl -X POST http://localhost:8081/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Specifications:**
- **Size:** 2.2GB (Safetensors)
- **Parameters:** 1.1B
- **Context Window:** 2K tokens
- **Architecture:** Llama-based
- **Quantization:** None (native BF16)
- **Expected Latency:** ~100-150ms (first token)
- **Expected RAM:** ~500MB peak

**Why this model:**
- ✅ Small enough for any hardware
- ✅ Actual transformer architecture (representative)
- ✅ Chat-tuned (useful for robot)
- ✅ Good baseline for comparison
- ✅ Real quality (usable responses)

**Test Scenarios:**
1. Single inference (verify output is sensible)
2. Memory usage during load
3. Token generation rate
4. Multiple sequential requests
5. Check layer splitting (monitor RAM)

---

### 1B. Phi-2

**Purpose:** Fast inference reference point

```bash
# Download (GGUF format for consistency)
wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf

# Run
./leafcutter-server --model ./models/phi-2-q4.gguf
```

**Specifications:**
- **Size:** 1.4GB (Q4_K_M)
- **Parameters:** 2.7B
- **Context Window:** 2K tokens
- **Architecture:** Custom (PhiAttention)
- **Quantization:** 4-bit (Q4_K_M)
- **Expected Latency:** ~120-160ms
- **Expected RAM:** ~600MB peak

**Why this model:**
- ✅ Very fast (2.7B, well-optimized)
- ✅ Small quantized size
- ✅ Good benchmark reference
- ✅ Instruction-following (good for tasks)

**Key Measurement:**
- Tokens/second (should be >200)
- Compare to TinyLlama (should be ~same speed due to size)

---

## 📱 TIER 2: Small Models (Robot Feasibility Proof)

### 2A. Qwen2-0.5B-Instruct

**Purpose:** Ultra-lightweight robotics brain

**Download:**
```bash
# HuggingFace Safetensors (full precision)
huggingface-cli download Qwen/Qwen2-0.5B-Instruct \
  --local-dir ./models/qwen2-0.5b

# Or GGUF quantized (preferred)
wget https://huggingface.co/bartowski/Qwen2-0.5B-Instruct-GGUF/resolve/main/Qwen2-0.5B-Instruct-Q4_K_M.gguf
```

**Specifications:**
- **Size:** 400MB (Q4_K_M)
- **Parameters:** 500M
- **Context Window:** 32K (!)
- **Architecture:** Qwen (efficient)
- **Quantization:** 4-bit (Q4_K_M)
- **Expected Latency:** ~80-120ms
- **Expected RAM:** ~200-300MB peak

**Why this model:**
- ✅ SMALLEST option (Pi Zero 2W friendly)
- ✅ Massive context window (good for memory)
- ✅ Modern Qwen architecture
- ✅ Instruction-optimized
- ✅ Very fast tokens/second

**Robot Application:**
```
Camera input → "I see: [object detection]"
Qwen0.5B → "move forward 30cm" (instant, 100ms)
```

**Critical Test:**
- Verify it runs on Pi Zero 2W without swap
- Measure context scaling (2K, 4K, 8K tokens)
- Test instruction following quality

---

### 2B. Phi-3-mini

**Purpose:** Accuracy improvement over ultra-small

**Download:**
```bash
# Safetensors (recommended)
huggingface-cli download microsoft/Phi-3-mini-4k-instruct \
  --local-dir ./models/phi-3-mini
```

**Specifications:**
- **Size:** 2GB (float16)
- **Parameters:** 3.8B
- **Context Window:** 4K tokens
- **Architecture:** Microsoft Phi (compact)
- **Quantization:** None (FP16)
- **Expected Latency:** ~200-250ms
- **Expected RAM:** ~1GB peak

**Why this model:**
- ✅ Better reasoning than 0.5B
- ✅ Still very efficient (3.8B is small)
- ✅ Known for high quality
- ✅ Good middle ground
- ✅ Microsoft research-backed

**Robot Application:**
```
Complex query + sensor history →
Phi-3-mini → Better decision making (still <300ms)
```

**Critical Test:**
- Does it run on Pi Zero 2W with swap?
- Compare output quality vs Qwen0.5B
- Measure latency impact of size increase

---

### 2C. Neural-Chat-3B

**Purpose:** Chat-optimized alternative

**Download:**
```bash
huggingface-cli download Intel/neural-chat-7b-v3-1 \
  --local-dir ./models/neural-chat-3b
```

**Specifications:**
- **Size:** 1.6GB (GGUF)
- **Parameters:** 3B
- **Context Window:** 2K
- **Architecture:** Llama-based
- **Quantization:** 4-bit
- **Expected Latency:** ~150-180ms
- **Expected RAM:** ~800MB peak

**Why this model:**
- ✅ Conversation-optimized
- ✅ Smaller than Neural-Chat-7B
- ✅ Good dialogue coherence
- ✅ Fast for 3B params

**Robot Application:**
```
User: "What obstacles do you see?"
Neural-Chat-3B → "I detected a tree 2m away and..."
(Natural conversation, not robotic)
```

---

## 🎯 TIER 3: Medium Models (Sweet Spot)

### 3A. Qwen2-1.5B

**Purpose:** Best efficiency/quality ratio

**Download:**
```bash
huggingface-cli download Qwen/Qwen2-1.5B-Instruct \
  --local-dir ./models/qwen2-1.5b
```

**Specifications:**
- **Size:** 900MB (Q4_K_M)
- **Parameters:** 1.5B
- **Context Window:** 32K tokens (!!)
- **Architecture:** Qwen2 (efficient)
- **Quantization:** 4-bit
- **Expected Latency:** ~250-350ms
- **Expected RAM:** ~600-800MB peak

**Why this model:**
- ✅ Sweet spot for speed + quality
- ✅ Huge context window (32K!)
- ✅ Good instruction following
- ✅ Still small enough for Pi Zero 2W + swap
- ✅ Qwen2 is very well optimized

**Robot Application:**
```
Long-term memory: 8K tokens of history
Current sensor: 500 tokens
Qwen2-1.5B: Process both, respond in <400ms
```

**Critical Test:**
- Test with 32K context (won't fit all at once)
- Verify layer splitting handles long context
- Measure latency degradation with history

---

### 3B. Mistral-7B-Instruct-Q4 ⭐ **CURRENT ROBOT MODEL**

**Purpose:** Production robotics brain

**Download:**
```bash
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  -O ./models/mistral-7b-q4.gguf
```

**Specifications:**
- **Size:** 4.3GB (Q4_K_M)
- **Parameters:** 7B
- **Context Window:** 32K tokens
- **Architecture:** Mistral (efficient for size)
- **Quantization:** 4-bit (Q4_K_M)
- **Expected Latency:** ~180-220ms
- **Expected RAM:** ~1.4GB peak (Leafcutter's 75% reduction from 5.5GB)

**Why this model:**
- ✅ Proven on THE-PATHFINDER-EYE
- ✅ Excellent instruction following
- ✅ Good reasoning capability
- ✅ ~200ms latency (acceptable for robot)
- ✅ Layer splitting brings RAM down to 1.4GB

**Robot Application:**
```
"I see a person 3 meters away. 
Previous visits here: [2K history].
What should I do?"

Mistral-7B → "Approach and ask if they need help."
(200ms response time)
```

**Critical Test:**
- Verify <200ms latency on Pi 5
- Test layer splitting (peak RAM ~1.4GB)
- Concurrent requests with vision pipeline
- Speculative decoding benefit (if enabled)

---

### 3C. Neural-Chat-7B

**Purpose:** Chat-focused 7B alternative

**Download:**
```bash
wget https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q4_K_M.gguf \
  -O ./models/neural-chat-7b-q4.gguf
```

**Specifications:**
- **Size:** 4GB (Q4_K_M)
- **Parameters:** 7B
- **Context Window:** 4K tokens
- **Architecture:** Llama-based
- **Quantization:** 4-bit
- **Expected Latency:** ~200-250ms
- **Expected RAM:** ~1.4GB peak

**Comparison to Mistral-7B:**
- ❌ Shorter context (4K vs 32K)
- ✅ Better dialogue quality
- ❌ Slightly larger quantized size
- ✅ More conversational tone

**When to use:**
- If robot focuses on natural conversation over instruction tasks
- Compare output quality with Mistral

---

## 🚀 TIER 4: Large Models (Stretch Goal)

### 4A. Llama-2-13B-Q4

**Purpose:** Prove layer splitting on large model

**Download:**
```bash
wget https://huggingface.co/TheBloke/Llama-2-13B-Instruct-GGUF/resolve/main/llama-2-13b-instruct.Q4_K_M.gguf \
  -O ./models/llama-2-13b-q4.gguf
```

**Specifications:**
- **Size:** 7GB (Q4_K_M)
- **Parameters:** 13B
- **Context Window:** 4K tokens
- **Architecture:** Llama v2
- **Quantization:** 4-bit
- **Expected Latency:** ~400-600ms
- **Expected RAM:** ~2.5GB peak (WITHOUT layer splitting)
- **Expected RAM:** ~700MB peak (WITH layer splitting) ⭐

**Why this model:**
- ✅ Demonstrates Leafcutter's 11x RAM advantage
- ✅ Massive leap from 7B to 13B
- ✅ Very sophisticated reasoning
- ❌ Slower on Pi 5 (may hit latency limit)

**Critical Test:**
- Verify peak RAM is ~700MB (layer splitting proof!)
- Measure actual latency on Pi 5
- Compare to running without Leafcutter (need 16GB+)

**Note:** This is the PROOF that Leafcutter solves the memory problem.

---

### 4B. Mixtral-8x7B-Q4

**Purpose:** MoE (Mixture of Experts) testing

**Download:**
```bash
wget https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
  -O ./models/mixtral-8x7b-q4.gguf
```

**Specifications:**
- **Size:** 26GB (Q4_K_M)
- **Parameters:** 46.7B (MoE)
- **Active Parameters:** 12.9B (only 2/8 experts active)
- **Context Window:** 32K tokens
- **Architecture:** Mistral MoE
- **Quantization:** 4-bit
- **Expected Latency:** ~800ms-2s
- **Expected RAM:** ~3-5GB (with layer splitting)

**Why this model:**
- ✅ Tests MoE routing in Leafcutter
- ✅ Massive model (proves layer splitting value)
- ✅ Very capable reasoning
- ❌ Likely TOO slow for robot (>500ms)
- ❌ May not fit even on Pi 5

**Critical Test:**
- Can layer splitting handle 46B model?
- Does MoE routing work efficiently?
- What's the actual latency? (probably unacceptable)
- This is the LIMIT test

---

## 🎬 Execution Order

### Week 1: Start Here
1. **TinyLlama-1.1B** ← Easiest, verify tools
2. **Phi-2** ← Speed reference
3. **Qwen2-0.5B** ← Smallest option

**What you'll learn:** Testing infrastructure works, baseline metrics established

---

### Week 2: Robot Feasible Models
1. **Phi-3-mini** ← Can it run on Pi Zero 2W?
2. **Neural-Chat-3B** ← Alternative small model
3. **Qwen2-1.5B** ← Sweet spot

**What you'll learn:** Best balance of speed/quality for robot

---

### Week 3: Production Ready
1. **Mistral-7B-Q4** ← Current robot model, detailed testing
2. **Neural-Chat-7B** ← Compare dialogue quality

**What you'll learn:** Production latency, RAM efficiency, concurrent request handling

---

### Week 4: Stretch Goals
1. **Llama-2-13B-Q4** ← Prove layer splitting
2. **Mixtral-8x7B-Q4** ← Limit test

**What you'll learn:** Maximum capabilities, MoE handling, when to stop scaling

---

## 📊 Measurement Template

For EACH model, measure:

```json
{
  "model_name": "mistral-7b-q4",
  "hardware": "Raspberry Pi 5",
  "test_date": "2026-05-13",
  
  "memory": {
    "model_size_gb": 4.3,
    "peak_ram_mb": 1430,
    "peak_ram_with_context_2k_mb": 1550,
    "peak_ram_with_context_4k_mb": 1680,
    "layer_splitting_working": true,
    "max_concurrent_layers": 1
  },
  
  "latency": {
    "first_token_ms": 185,
    "tokens_per_second": 254,
    "avg_token_time_ms": 3.9,
    "latency_p99_ms": 250
  },
  
  "throughput": {
    "single_request_tokens_per_sec": 254,
    "concurrent_4_requests_tokens_per_sec": 850,
    "batch_efficiency_percent": 85
  },
  
  "quality": {
    "response_quality": "Excellent",
    "instruction_following": "Perfect",
    "coherence": "High",
    "hallucination_rate_percent": 5
  },
  
  "robot_suitability": {
    "meets_latency_target_200ms": true,
    "fits_pi_zero_2w": false,
    "fits_pi_5": true,
    "concurrent_request_handling": "Good"
  }
}
```

---

## ✅ Recommended Starting Point

**For immediate testing:**

1. Start with **TinyLlama-1.1B** (2-3 hours)
   - Download: 2 minutes
   - Setup: 5 minutes
   - Testing: 2-3 hours
   - Result: Baseline established

2. Then **Qwen2-0.5B** (2-3 hours)
   - Verify Pi Zero 2W compatibility
   - Measure speed difference

3. Then **Mistral-7B-Q4** (4 hours)
   - Current robot model
   - Most important for THE-PATHFINDER-EYE

**Total first-pass:** ~8-10 hours

This gives you solid data on what Leafcutter can do across a 1B → 7B range.

---

## 🎯 Success Indicators

By end of Week 1, you should have:
- ✅ Benchmark tool running
- ✅ CSV data from 3 models
- ✅ Graphs showing latency vs size
- ✅ Graphs showing RAM vs model
- ✅ Confidence in measurement accuracy

If these aren't true, fix the testing infrastructure FIRST before moving to bigger models.

