# 🌿 LeafcutterLLM Testing Checklist
## Quick Reference for Team Implementation

---

## ✅ PRE-TESTING CHECKLIST

### Environment Setup
- [ ] Go 1.22+ installed (`go version`)
- [ ] CGO enabled (`CGO_ENABLED=1`)
- [ ] OpenBLAS dev installed (`apt install libopenblas-dev`)
- [ ] HuggingFace CLI installed (`pip install huggingface-hub`)
- [ ] Python 3.8+ with matplotlib (`pip install matplotlib`)

### Repository Setup
- [ ] Clone Leafcutter repo
- [ ] Build server: `CGO_ENABLED=1 go build -o leafcutter-server ./cmd/server`
- [ ] Verify build: `./leafcutter-server --version`
- [ ] Create directories:
  ```bash
  mkdir -p models results logs
  ```

### Hardware Confirmation
- [ ] What hardware are we using?
  - [ ] Pi 5 (8GB)?
  - [ ] Pi Zero 2W (512MB)?
  - [ ] Laptop?
  - [ ] Multiple?
- [ ] RAM verified: `free -h`
- [ ] CPU info: `lscpu` or `sysctl hw.ncpu`

---

## 📥 WEEK 1: MODEL DOWNLOADS

### Tier 1 Models (Must Have)
- [ ] TinyLlama-1.1B (2.2GB)
  ```bash
  huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --local-dir ./models/tinyllama-1.1b
  ```
  - Expected size: 2.2GB
  - Verify checksum: ✓

- [ ] Qwen2-0.5B (400MB)
  ```bash
  huggingface-cli download Qwen/Qwen2-0.5B-Instruct \
    --local-dir ./models/qwen2-0.5b
  ```
  - Expected size: 400MB
  - Verify checksum: ✓

### Tier 2 Models (Should Have)
- [ ] Phi-3-mini (2GB)
  - [ ] Download verified
  - [ ] Checksum validated

- [ ] Qwen2-1.5B (900MB)
  - [ ] Download verified
  - [ ] Checksum validated

- [ ] Mistral-7B-Q4 (4.3GB) ⭐ **Current robot model**
  ```bash
  wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    -O ./models/mistral-7b-q4.gguf
  ```
  - Expected size: 4.3GB
  - SHA256 verified: ✓

### Tier 3 Models (Optional - Space Permitting)
- [ ] Neural-Chat-7B (4GB)
- [ ] Llama-2-13B-Q4 (7GB)

### Total Disk Space Needed
- Tier 1 only: ~2.6GB
- Tier 1-2: ~12GB
- Tier 1-3: ~23GB
- **Recommendation:** Start with Tier 1, expand as needed

---

## 🧪 WEEK 1: INFRASTRUCTURE TESTING

### Benchmark Tool Validation
- [ ] Build benchmark: `go build -o leafcutter-bench ./cmd/benchmark`
- [ ] Run basic benchmark:
  ```bash
  ./leafcutter-bench --hidden-size 4096 --num-layers 32 --mat-m 4096 --mat-n 4096 --mat-k 4096
  ```
- [ ] Output format correct (text, not errors)
- [ ] All metrics reported (RAM, BLAS speed, throughput)

### Single Model Test
- [ ] Start server with TinyLlama:
  ```bash
  ./leafcutter-server --model ./models/tinyllama-1.1b --port 8081
  ```
- [ ] Server starts without errors: ✓
- [ ] Health check passes:
  ```bash
  curl http://localhost:8081/health
  ```
- [ ] Response is `{"status": "ok"}`

### API Test
- [ ] Basic inference works:
  ```bash
  curl -X POST http://localhost:8081/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "max_tokens": 10}'
  ```
- [ ] Returns valid JSON: ✓
- [ ] Contains "tokens" or "text" field: ✓
- [ ] No errors in response: ✓

### Results Collection
- [ ] Create results directory: `mkdir -p results`
- [ ] Run test script: `bash test_single_model.sh ./models/tinyllama-1.1b`
- [ ] JSON output generated: ✓
- [ ] Metrics are numeric: ✓

---

## 📊 WEEK 1: METRIC COLLECTION (TinyLlama Test)

### Memory Metrics
- [ ] Peak RAM recorded (MB)
  - Expected: ~500MB for TinyLlama
  - Actual: _____ MB
  - Within expected range? [ ] Yes [ ] No

- [ ] Layer splitting verified
  - [ ] Monitor shows only 1 layer in RAM
  - [ ] Peak doesn't exceed single-layer footprint

### Latency Metrics
- [ ] First token latency (ms)
  - Expected: ~100-150ms
  - Actual: _____ ms
  - Note any anomalies: _________

- [ ] Tokens per second
  - Expected: >100
  - Actual: _____ TPS
  - Consistent across runs? [ ] Yes [ ] No

### Quality Check
- [ ] Response is sensible
  - Input: "What is machine learning?"
  - Output looks reasonable: [ ] Yes [ ] No
  - No gibberish/hallucinations: [ ] Yes [ ] No

### Save Results
- [ ] JSON file created: `results/test_tinyllama-1.1b_[DATE].json`
- [ ] All metrics populated
- [ ] Timestamp recorded
- [ ] Hardware info included

---

## 📊 WEEK 2: QWEN2-0.5B TESTING

- [ ] Download verified
- [ ] Server starts: `./leafcutter-server --model ./models/qwen2-0.5b`
- [ ] Health check passes
- [ ] Inference works: ✓
- [ ] Metrics collected:
  - Peak RAM: _____ MB (Expected: ~200-300MB)
  - Latency: _____ ms (Expected: ~80-120ms)
  - TPS: _____ (Expected: >150)
- [ ] Results saved to JSON
- [ ] Quality check: [ ] Good [ ] Acceptable [ ] Poor

### Comparison to TinyLlama
- [ ] Faster? [ ] Yes [ ] No (by _____ ms)
- [ ] Less RAM? [ ] Yes [ ] No (by _____ MB)
- [ ] Higher TPS? [ ] Yes [ ] No (by _____ TPS)
- [ ] Quality comparable? [ ] Yes [ ] No

---

## 📊 WEEK 2-3: MISTRAL-7B-Q4 TESTING ⭐ CRITICAL

### Pre-Test
- [ ] Model downloaded and verified (4.3GB)
- [ ] Sufficient disk space: [ ] Yes [ ] No
- [ ] Sufficient RAM: [ ] 8GB+ [ ] Using swap

### Server Startup
- [ ] Start with appropriate context:
  ```bash
  ./leafcutter-server \
    --model ./models/mistral-7b-q4.gguf \
    --max-ctx 2048 \
    --port 8081
  ```
- [ ] Server starts without OOM error: ✓
- [ ] Loads in reasonable time: < 30 seconds

### Critical Metrics (Robot Requirements)
- [ ] **Latency <200ms?**
  - Actual: _____ ms
  - Requirement: <200ms
  - Status: [ ] PASS [ ] FAIL
  - If FAIL, note issue: _________

- [ ] **Peak RAM <1.5GB?** (Layer splitting proof)
  - Actual: _____ MB
  - Expected: ~1.4GB (75% reduction)
  - Status: [ ] PASS [ ] FAIL
  - If FAIL, check: [ ] Layer splitting enabled [ ] GC happening

- [ ] **Throughput >100 TPS?**
  - Actual: _____ TPS
  - Expected: 200-300 TPS
  - Status: [ ] PASS [ ] FAIL
  - If FAIL, check: [ ] BLAS working [ ] Fallback to Go

### Concurrent Request Test
- [ ] Single request: ✓
  - Latency: _____ ms
  
- [ ] 4 concurrent requests: ✓
  - Avg latency: _____ ms
  - Max latency: _____ ms
  - Throughput: _____ TPS
  - Status: [ ] Good [ ] Acceptable [ ] Degraded

### Layer Splitting Verification
- [ ] Monitor layer loads during inference
- [ ] Only 1 layer in RAM at a time: [ ] Yes [ ] No
- [ ] Peak RAM matches single-layer size: [ ] Yes [ ] No
- [ ] Comments: _________

### Save Comprehensive Results
- [ ] All metrics in JSON
- [ ] Timestamp recorded
- [ ] Hardware info (Pi 5, RAM config)
- [ ] Notes about any issues
- [ ] Quality assessment

### ROBOT INTEGRATION DECISION
- [ ] Is this ready for robot? [ ] YES [ ] NO [ ] MAYBE
- [ ] If NO, why? _________
- [ ] If MAYBE, what needs fixing? _________

---

## 📈 WEEK 3: ANALYSIS & GRAPHS

### Data Preparation
- [ ] All test results collected (minimum 3 models)
- [ ] Results formatted as JSON (check format)
- [ ] No missing fields
- [ ] Timestamps consistent format

### Graph Generation
- [ ] Run: `python3 generate_graphs.py`
- [ ] latency_comparison.png generated: [ ] Yes [ ] No
- [ ] ram_comparison.png generated: [ ] Yes [ ] No
- [ ] throughput_comparison.png generated: [ ] Yes [ ] No

### Graph Analysis
- [ ] Latency increases linearly with model size? [ ] Yes [ ] No
- [ ] RAM increases linearly? [ ] Yes [ ] No
- [ ] Mistral-7B crosses <200ms threshold? [ ] Yes [ ] No
- [ ] Layer splitting visible in RAM graph? [ ] Yes [ ] No
- [ ] Any anomalies? _________

---

## 📋 WEEK 4: FINAL VALIDATION

### Stretch Goals (If Time)
- [ ] Llama-2-13B-Q4 tested (optional)
- [ ] Mixtral-8x7B tested (optional)
- [ ] Speculative decoding benchmarked (optional)

### Report Generation
- [ ] Create performance_report.md
  - [ ] Summary of all tests
  - [ ] Key findings
  - [ ] Recommendations
  - [ ] Graphs embedded

- [ ] Create optimization_guide.md
  - [ ] Bottleneck analysis
  - [ ] Recommendations for Leafcutter team
  - [ ] Priority ranking

- [ ] Create robot_integration_plan.md
  - [ ] Which models work for robot
  - [ ] Expected performance on Pi 5
  - [ ] Expected performance on Pi Zero 2W
  - [ ] Deployment recommendations

### Final Deliverables
- [ ] results/ directory with all JSON files
- [ ] graphs/ directory with all PNG files
- [ ] reports/ directory with markdown files
- [ ] All accessible and documented

---

## 🎯 SUCCESS CRITERIA

### By End of Week 1
- [ ] Infrastructure working (tools, scripts, API)
- [ ] 3 models tested (TinyLlama, Qwen0.5B, Qwen1.5B)
- [ ] Metrics being collected correctly
- [ ] No framework blockers

### By End of Week 2
- [ ] Mistral-7B tested thoroughly
- [ ] Latency/RAM/Throughput verified
- [ ] Decision made: Ready for robot? YES/NO/MAYBE
- [ ] Any optimization gaps identified

### By End of Week 3
- [ ] All graphs generated and analyzed
- [ ] Performance report written
- [ ] Optimization recommendations documented
- [ ] Ready to present to team

### By End of Week 4
- [ ] All testing complete
- [ ] Final report delivered
- [ ] Leafcutter team has actionable insights
- [ ] Robot integration plan ready

---

## 🚨 TROUBLESHOOTING QUICK REFERENCE

| Issue | Solution | Verify |
|-------|----------|--------|
| Server won't start | Check RAM free: `free -h` | [ ] 2GB+ free |
| Latency very high | Warm up first, then measure | [ ] 3+ dummy requests |
| RAM usage wrong | Use `runtime.MemStats`, not `ps` | [ ] MemStats used |
| Graphs empty | Check results/ directory has JSON files | [ ] Files exist |
| BLAS not working | Rebuild with `CGO_ENABLED=1` | [ ] CGO enabled |
| Model won't load | Check file path and format | [ ] Path correct |

---

## 📞 Status Check-In Points

**Every Monday:**
- [ ] Report completed tests
- [ ] Any blockers identified
- [ ] Next week's plan
- [ ] Resource needs (storage, compute time)

**Every Friday:**
- [ ] Summary of week's findings
- [ ] Preliminary graphs
- [ ] Any surprises
- [ ] Adjusted timeline if needed

---

## 🎉 COMPLETION SIGN-OFF

- [ ] All tests executed
- [ ] All metrics collected
- [ ] All graphs generated
- [ ] All reports written
- [ ] Team briefed on findings
- [ ] Next steps identified

**Completed by:** _____________  
**Date:** _____________  
**Sign-off:** _____________  

---

**This checklist is your guide to Leafcutter validation.**  
**Print it. Check items off. Present results.**

Good luck! 🚀

