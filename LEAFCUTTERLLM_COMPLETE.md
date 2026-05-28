# LeafcutterLLM — Complete Update & Expansion Strategy

> Last updated: 2026-05-27  
> Status: 113 tests passing, 0 failures | 12 architectures | 8 models validated

---

# PART 1: Updated Social Media Posts

Use these directly. Copy-paste into each platform. All numbers are verified from your test runs.

---

## LINKEDIN — The Journey Post

*(Pair with your "70B in 1.2GB" poster)*

A year ago, a 70-billion-parameter AI model refused to load on my 16 GB ThinkPad. The error might as well have read: *"Intelligence is reserved for those who can afford it."* That broke something in me. So I built LeafcutterLLM.

**The journey:** AirLLM inspired me → Go prototype failed (GC pauses) → burned it down → Rust rewrite → 42 days chasing a signedness bug in a 16-entry lookup table → NaN propagation through f16 KV cache → the breakthrough: `VmRSS: 1,155,208 kB` while a 70B model generated text. I stood up from my chair.

**The numbers (validated, not benchmarketed):**

| Metric | Result |
|--------|--------|
| **Peak RAM (70B)** | **1,155 MB** — layer streaming + `madvise` eviction |
| **Speedup** | **4.4x** — matrixmultiply + Rayon + AVX2 + thread-local caches |
| **Tests** | **113 passing** — SIMD, round-trip quant, end-to-end |
| **Models** | **8 validated** — Llama, Mistral, Ministral, Qwen2/2.5/3.5/3.6, 70B |
| **Quants** | **Q4_0 through IQ5_0** — 9 formats |
| **Chat templates** | **5 families** — Llama-3, Mistral, ChatML, Gemma, Ministral |

The architecture: layer streaming (one layer resident, rest on disk), quantized GEMM directly on Q4_K blocks, three-path backend (native SIMD → FFI bridge → auto-fallback), and config correction that verifies metadata against actual tensor shapes because metadata lies.

There is a global intelligence divide. The best models live behind API keys and 80GB GPUs. A student in Lagos with a used ThinkPad gets autocomplete. Same internet, different access. That's a software failure. I'm writing different software.

Local AI is a public good. Not a luxury product.

#LocalAI #RustLang #LLM #OpenSourceAI #InferenceEngine #Quantization #AIAccessibility #70BillionParameters #MemoryOptimization

---

## LINKEDIN — Technical Deep-Dive

*(Pair with your "Accessibility & Innovation" poster)*

How I fit 70B parameters into 1.1 GB of RAM — a technical teardown.

The core loop is stupidly simple:

```rust
for layer in 0..num_layers {
    let weights = mmap_layer(file, layer);     // ~150 MB
    hidden_state = forward(weights, hidden_state);
    madvise(weights.ptr, weights.len(), MADV_DONTNEED); // evict immediately
}
```

`madvise(MADV_DONTNEED)` is the killer — without it, Linux's page cache hoards pages and RSS grows linearly. With it, pages are reclaimed immediately. The difference between "fits in 1 GB" and "needs 8 GB."

Quantized GEMM runs directly on Q4_K/Q5_K/Q6_K/IQ4_NL blocks. No dequantization to f32 first. Thread-local caches, AVX2 SIMD via `matrixmultiply`, 256-entry lookup tables for IQ formats. Max layer error vs. reference: **0.003**.

Config correction: Ministral claims `hidden_size=4096, num_layers=32`. Actual tensors: `3072, 26`. RoPE base: metadata says 10,000, actual: **1,000,000**. Trust the tensors, not the headers.

Three-path backend: Native Rust SIMD for standard transformers. FFI bridge for Qwen3.5's hybrid SSM+Attention. Auto-fallback for unsupported quants (IQ1_M detected and routed automatically). The model **always** runs.

113 tests. Zero failures. Zero assumptions about your hardware.

#RustLang #LLM #Quantization #AIEngineering #Mamba #SSM #MemoryMapping #SIMD

---

## INSTAGRAM — Post 1: The Hook

*(Pair with Poster 1 — "70B in 1.2GB")*

Your laptop isn't too slow for AI. The software just gave up on you.

I built software that streams one layer at a time — keeping only what's needed in RAM, evicting the rest.

**Result: 70B parameters. 1,155 MB peak RAM. On a ThinkPad.**

No cloud. No subscription. No "out of memory."

113 tests passing. 8 models validated. 9 quantization formats. 5 chat template families.

The goal isn't to be the fastest. It's to make the best open-weight models accessible to anyone with a used laptop and an internet connection.

Local AI is a public good. Not a luxury product.

#localai #rustlang #llm #aiaccessibility #quantization #machinelearning #opensource #consumerhardware #thinkpad #70billionparameters #memoryoptimization #aiengineering

---

## INSTAGRAM — Post 2: The Mission

*(Pair with Poster 2 — "Accessibility & Innovation")*

There is a global intelligence divide.

The best reasoning models live in data centers behind API keys and 80GB GPUs. A student in Lagos with a used ThinkPad gets a dumb autocomplete. Same internet, different access.

I built LeafcutterLLM because that's a software problem, not a hardware law.

**The architecture:**
⚡ Layer streaming — one layer resident, rest on disk
⚡ Quantized GEMM — matmul directly on Q4_K blocks
⚡ madvise eviction — OS pages returned immediately
⚡ Three-path backend — native SIMD → FFI bridge → auto-fallback
⚡ Config correction — metadata lies, tensors tell the truth
⚡ Chat templates — 5 families auto-detected from GGUF

**The numbers:**
🔢 70B model → 1,155 MB peak RSS
🔢 4.4x speedup over naive implementation
🔢 0.003 max layer error vs. reference
🔢 113 tests passing
🔢 12 architectures detected

Local AI won't be niche forever. The models are getting better. The hardware is getting cheaper. The software just needs to stop assuming everyone has a data center.

#aiaccessibility #localai #rustlang #llm #machinelearning #globaldivide #opensourceai #quantization #inferenceengine #consumerhardware #publicgood

---

## INSTAGRAM — Post 3: The Human Story

*(Pair with Poster 3 — Japanese poster)*

One year. One project. Almost gave up three times.

Started in Go because I loved the syntax. Learned the hard way that garbage collectors and AI inference don't mix. Burned it down, rewrote in Rust.

Six weeks chasing a bug that turned 10 tokens of perfect English into Unicode chaos. The cause? A 16-entry lookup table with the wrong sign bit. IQ4_NL uses signed 4-bit values [-8..7], not unsigned [0..15]. Two conflicting tables in the codebase. Every 4th token hit the wrong one.

42 days to find it. 15 minutes to fix.

Another month fighting NaN propagation through KV cache f16 accumulation. Forward pass perfect on call 1, NaN on call 2. f16 round-trip error accumulating past threshold. Switched to f32 storage. Gone.

Then a config bug: RoPE base for Ministral was 1,000,000 but we were reading 10,000. One line fix. Logits jumped from 5.6 to 15+.

The breakthrough: `cat /proc/[pid]/status` → `VmRSS: 1155208 kB` while a 70B model generated coherent text. I stood up from my chair.

This isn't about being the fastest framework. It's about proving that the best open-weight AI models belong to everyone.

Local AI is a public good.

#rustlang #aiengineering #buildinpublic #localai #llm #machinelearning #debugginglife #indiedev #systemsprogramming #aiaccessibility #nevergiveup

---

## TWITTER/X — 8-Tweet Thread

**Tweet 1:** I fit a 70-billion-parameter AI model into 1.1 GB of RAM and ran it on a ThinkPad. No GPU. No cloud. No subscription. Here's how I built LeafcutterLLM 🧵

**Tweet 2:** It started with frustration. State-of-the-art models refused to load because my laptop "only" had 16 GB RAM. The unspoken rule: serious AI requires serious hardware. I refused to accept that.

**Tweet 3:** First attempt: Go. Beautiful language. GC paused during inference. CGO overhead worse than staying in C. The runtime fought me for every byte. Burned it down. Rewrote in Rust.

**Tweet 4:** The core trick is stupidly simple:
```
for each layer:
    mmap layer weights from disk
    run forward pass
    madvise(MADV_DONTNEED)  // evict immediately
```
At most ONE layer resident in RAM. Peak RSS for 70B: 1,155 MB.

**Tweet 5:** The debugging war stories:
• 42 days on a signedness bug in a 16-entry IQ4_NL lookup table
• NaN propagation through f16 KV cache (perfect on call 1, NaN on call 2)
• RoPE base 1,000,000 vs 10,000 — one line fix, logits jumped 3x

**Tweet 6:** 113 tests passing. 12 architectures. 9 quantization formats. 5 chat template families. 4.4x speedup. 0.003 max layer error. Config correction that verifies metadata against actual tensor shapes because metadata lies.

**Tweet 7:** Three-path backend: Native Rust SIMD for standard transformers. FFI bridge for bleeding-edge architectures. Auto-fallback so every model runs, even if not yet fully optimized. The model ALWAYS runs.

**Tweet 8:** This matters because there's a global intelligence divide. The best reasoning models live behind API keys and 80GB GPUs. A student in Lagos with a used ThinkPad gets autocomplete. Same internet. Different access. That's a software failure. I'm trying to write different software. Local AI is a public good, not a luxury product. Follow for the open-source release.

---

# PART 2: Model Expansion Strategy

## Current Verified State

| Architecture | Backend | Status | Notes |
|-------------|---------|--------|-------|
| Llama 2/3/3.1 | Native | ✅ Working | Full attention, all quants |
| Mistral 7B | Native | ✅ Working | Sliding Window Attention |
| Ministral 3B/8B | Native | ✅ Working | RoPE 1M fix, reasoning template |
| Qwen2/2.5 | Native | ✅ Working | Standard attention |
| Qwen3.5/3.6 | FFI | ✅ Working | Hybrid SSM+Attention via llama.cpp |
| 70B Llama | Native | ✅ Working | 1,155 MB peak RSS |

## Expansion Target Matrix

### TIER 1: Quick Wins — Same Architecture, Just Test

These models use the **standard Llama-family architecture** that your native path already supports. They should "just work" with minimal or no code changes.

| Model | Size | Download | Why | Effort |
|-------|------|----------|-----|--------|
| **Yi-1.5-6B** | ~3.6 GB | `MaziyarPanahi/Yi-6B-GGUF` | Apache 2.0, Chinese+English, standard Llama arch | Zero code, just test |
| **Yi-1.5-9B** | ~5.4 GB | `itlwas/Yi-1.5-9B-Q4_K_M-GGUF` | Same as above, larger | Zero code, just test |
| **Nemotron-4-4B** | ~2.3 GB | `nvidia/Nemotron-Mini-4B-Instruct-GGUF` | NVIDIA's model, optimized for inference | Zero code, just test |

**What to do:** Download any of these, run `test_generation`. If it produces coherent text, add to the validated list. No code changes needed.

---

### TIER 2: Small Code Changes — New Features Needed

These models need **minor architectural additions** that your codebase can accommodate.

| Model | Size | Download | What's Needed | Effort |
|-------|------|----------|---------------|--------|
| **Gemma-2B** | ~1.3 GB | `ggml-org/gemma-3-2b-it-GGUF` | Logit soft-capping: `soft_cap * tanh(logits/soft_cap)` | 4-6 hours |
| **Phi-4** | ~8.4 GB | `bartowski/Phi-4-GGUF` | ChatML template, RoPE base 250K | 2-3 hours |
| **Qwen3** | Varies | `unsloth/Qwen3-8B-GGUF` | New architecture tag, likely standard attention | 1-2 hours |

**Gemma details:** Google models apply **logit soft-capping** after the final layer — a nonlinearity that prevents extreme logits. Formula: `output = soft_cap * tanh(output / soft_cap)`. The `soft_cap` value is in GGUF metadata (`gemma.logit_cap` or similar). Without it, generated text can be erratic. This is a single post-processing step after `lm_head`.

**Phi-4 details:** Standard dense transformer (same as Llama), but uses **ChatML** template (`<|im_start|>user
{msg}<|im_end|>
<|im_start|>assistant
`) and **RoPE base 250,000** (your `get_meta_f32` already handles this). Your chat template infrastructure already detects ChatML. Should work with just architecture detection.

---

### TIER 3: Major Features — Significant Engineering

These models require **fundamentally new code paths**. Recommend FFI-only for now.

| Model | Size | Architecture Challenge | Native Effort |
|-------|------|----------------------|---------------|
| **DeepSeek-V3** | 671B (37B active) | **MLA attention** — KV cache compressed to latent vector via low-rank projections. Not standard MHA/GQA. | 40-60 hours |
| **DeepSeek-V3** | 671B (37B active) | **MoE routing** — 256 experts, 8 activated per token via learned router | 20-30 hours |
| **MiniMax-M2.5** | 230B (10B active) | **Lightning Attention** — custom attention mechanism + Top-2 MoE routing | 50+ hours |
| **Falcon-H1** | 1B-34B | **Hybrid Transformer + Mamba SSM** — similar to Qwen3.5 but different SSM variant | 15-20 hours |
| **MPT** | 7B-30B | **ALiBi attention** — replaces RoPE with linear biases added to attention scores. No positional embeddings at all. | 15-20 hours |

**DeepSeek-V3 MLA explained:** Standard attention caches full Key and Value tensors per layer per token. MLA compresses them through low-rank projections:
1. **Down-project:** `c_t = W_DKV * h_t` (compress to latent vector)
2. **Cache only `c_t`** (massive memory savings)
3. **Up-project when needed:** `k_t = W_UK * c_t`, `v_t = W_UV * c_t`
4. **Decoupled RoPE:** Separate content and positional components

This is fundamentally different from your current KV cache. The good news: llama.cpp supports it via FFI, so you can validate DeepSeek immediately.

**MiniMax-M2.5 explained:** 230B total, only 10B active per token — the most sparse frontier model. Uses "Lightning Attention" (their own custom mechanism) plus Top-2 MoE routing. Requires `ik_llama.cpp` fork, not main llama.cpp. MIT licensed but very hard to implement natively.

**MPT ALiBi explained:** Instead of RoPE (rotary position encoding), MPT adds **linear bias terms** to attention scores based on token distance: `score(q, k) = q·k / sqrt(d) + m * (i - j)` where `m` is a head-specific slope and `(i-j)` is the distance. No positional embeddings at all. This is elegant but requires a completely different attention scoring path.

---

### TIER 4: Research — Status Unknown

| Model | Notes |
|-------|-------|
| **ChatGLM / Zhipu AI (GLM)** | Chinese model by Zhipu AI. Architecture is **GLM** (General Language Model) — uses autoregressive blank infilling, not standard causal LM. Limited GGUF support. **Recommend: wait for better GGUF ecosystem.** |
| **Command R+ (Cohere)** | 104B, RAG-optimized, uses ALiBi + alternating dense/MoE layers. Non-commercial license (CC-BY-NC). **Skip due to license.** |
| **GPT-OSS (NVIDIA)** | Very new, MXFP4 format. Experimental. **Wait for stabilization.** |

---

## Recommended Download & Test Order

When you're ready to download, do it in this order (smallest first):

```bash
# === TIER 1: Zero-code validation (2-6 GB each) ===

# 1. Yi-1.5-6B — smallest, should "just work" (~3.6 GB)
huggingface-cli download MaziyarPanahi/Yi-6B-GGUF \
    --include "*Q4_K_M.gguf" --local-dir ./models

# 2. Gemma-2B-IT — tests logit soft-capping (~1.3 GB)  
huggingface-cli download ggml-org/gemma-3-2b-it-GGUF \
    --include "*Q4_K_M.gguf" --local-dir ./models

# 3. Llama-3.1-8B-Instruct — replace broken Q4_0_4_4 (~4.7 GB)
huggingface-cli download TheBloke/Meta-Llama-3.1-8B-Instruct-GGUF \
    --include "*Q4_K_M.gguf" --local-dir ./models

# === TIER 2: Validate via FFI first (8-15 GB) ===

# 4. DeepSeek-V3-Q4_K_M — validate MoE+MLA via FFI (~15 GB)
huggingface-cli download unsloth/DeepSeek-V3-GGUF \
    --include "*Q4_K_M.gguf" --local-dir ./models

# 5. Phi-4-Q4_K_M — validate ChatML + high RoPE base (~8.4 GB)
huggingface-cli download bartowski/Phi-4-GGUF \
    --include "*Q4_K_M.gguf" --local-dir ./models

# === TIER 3: Exotic architectures (large) ===

# 6. Falcon-H1-1B-Instruct — hybrid SSM test (~1 GB)
huggingface-cli download tiiuae/Falcon-H1-1B-Instruct-GGUF \
    --include "*Q4_K_M.gguf" --local-dir ./models
```

---

## Code Changes Needed Per Tier

### Tier 1 (Yi, Nemotron): Architecture Detection Only

Add to `src/model/arch.rs`:

```rust
// In the architecture detection match:
"yi" | "yi1" | "yi1.5" => ModelArchitecture::Yi,
"nemotron" | "nvidia_nemotron" => ModelArchitecture::Nemotron,

// Yi uses llama.* GGUF keys and Llama layer mappings
// Nemotron likely does too — just needs testing
```

### Tier 2 (Gemma): Logit Soft-Capping

Add to `src/inference/engine.rs` after the final `lm_head` forward:

```rust
// Gemma logit soft-capping
if self.config.logit_soft_cap > 0.0 {
    let cap = self.config.logit_soft_cap;
    for logit in logits.iter_mut() {
        *logit = cap * (*logit / cap).tanh();
    }
}
```

And read the cap from GGUF metadata:

```rust
// In extract_config:
cfg.logit_soft_cap = Self::get_meta_f32(file, &[
    "gemma.logit_cap",
    "gemma3.logit_cap",
    "gemma2.logit_cap",
]).unwrap_or(0.0);
```

### Tier 2 (Phi-4): Architecture Detection + Chat Template

Phi-4 uses standard Llama architecture. Just add detection:

```rust
"phi" | "phi3" | "phi4" => ModelArchitecture::Phi,
```

Your existing ChatML detection in `chat_template.rs` will handle the template.

### Tier 3 (DeepSeek, MiniMax): FFI for Now

These are validated through your **existing auto-fallback path**. No code changes needed — just test and confirm they work. Native implementation is a long-term project.

Test DeepSeek via FFI:
```bash
cargo run --release --features llama-ffi --bin test_generation -- \
    --model DeepSeek-V3-Q4_K_M.gguf \
    --prompt "Explain quantum mechanics"
```

---

## Quantization Format Roadmap

| Format | Priority | Why | Effort |
|--------|----------|-----|--------|
| **Q6_K** | High | "Almost lossless" — high user demand | ~3 hours |
| **Q5_K_M** | High | Better quality than Q4_K_M | ~2 hours |
| **IQ2_XXS** | Medium | Extreme compression for DeepSeek (~1.9GB active) | ~6 hours |
| **IQ3_XXS** | Medium | Fit 70B on 8GB RAM | ~4 hours |
| **IQ2_S** | Low | Better quality than IQ2_XXS | ~5 hours |

Q6_K and Q5_K_M use the same super-block structure as Q4_K — different bit allocations within the block. The shard format would also need extension to support these.

---

## The Dream Collection: Models You Want + Feasibility

| Your Target | Architecture | Native Feasibility | Timeline |
|-------------|-------------|-------------------|----------|
| **MiniMax** | Lightning Attention + Top-2 MoE | Very Hard | FFI now, native after v1.0 |
| **Z AI / GLM** | GLM (blank infilling) | Hard (non-causal) | Wait for GGUF ecosystem |
| **DeepSeek** | MLA + MoE | Hard | FFI now, native after v1.0 |
| **Falcon H1** | Hybrid Transformer + Mamba | Medium | Can reuse SSM code, 2-3 weeks |
| **Yi** | Standard Llama | Easy | This week |
| **Gemma** | Standard + logit cap | Easy | This week |
| **Phi-4** | Standard Llama + ChatML | Easy | This week |
| **MPT** | ALiBi (no RoPE) | Medium | 2-3 weeks |

---

# PART 3: Quick Reference

## Key Numbers (Memorize for Interviews)

| Number | What |
|--------|------|
| **1,155 MB** | Peak RSS for 70B Llama-3.1 on ThinkPad |
| **4.4x** | Speedup over naive Rust |
| **0.003** | Max layer error vs. reference |
| **113** | Tests passing, 0 failures |
| **12** | Architectures detected |
| **9** | Quantization formats supported |
| **5** | Chat template families |
| **42 days** | IQ4_NL bug hunt |
| **1,000,000** | Ministral RoPE base (vs default 10,000) |

## One-Line Explanations

**Why Rust?** "Zero-cost abstractions, no GC pauses during inference, `madvise(MADV_DONTNEED)` actually evicts pages. The difference between fits in 1 GB and needs 8 GB."

**Why layer streaming?** "At most one transformer layer resident in RAM at any moment. Load, compute, evict, repeat. 70B parameters in 1.1 gigabytes."

**Why another inference engine?** "llama.cpp optimizes for speed. I optimize for accessibility. If you have 8GB RAM, use llama.cpp. If you have 2-4GB and want any 70B model to run at all, use LeafcutterLLM."

**What's hard about it?** "Config correction — GGUF metadata lies about tensor shapes. Ministral claims 32 layers, has 26. Claims hidden_size 4096, actual is 3072. RoPE base 10K, actual is 1M. Trust the tensors, not the headers."

## Social Media Posting Schedule

| Day | Platform | Content |
|-----|----------|---------|
| Monday AM | LinkedIn | Journey post |
| Monday PM | Instagram | Post 1 (The Hook) |
| Tuesday AM | Twitter/X | 8-tweet thread |
| Wednesday AM | LinkedIn | Technical deep-dive |
| Thursday AM | Instagram | Post 2 (The Mission) |
| Friday AM | Instagram | Post 3 (Human Story) |

---

*Local AI is a public good. Not a luxury product.*
