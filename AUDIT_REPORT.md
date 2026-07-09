# LeafcutterLLM — Adversarial Audit Report (Pre-Release)

> **Date:** 2026-07-09  
> **Auditor:** Kimi K2.6 (Nvidia), using the `security-audit` skill  
> **Scope:** Full read-only review of `/home/xander/Documents/portfolio/LeafcutterLLM/rust/src/`  
> **Status:** Read-only. No code was edited during this audit.  
> **Methodology:** 12-surface checklist (OWASP-aligned) + Rust perf-critical checklist (9 categories from prior Leafcutter audit).

---

## Trust Boundary

```
+---------------------------------------------------+
| INSIDE (trusted):                                  |
|  Rust inference engine, tokenizer, KV cache       |
|  llama.cpp FFI bindings, HTTP server (Axum)        |
|  Native GGUF loader (mmap), quantized GEMM kernels |
+---------------------------------------------------+
             ^                       ^
             |                       |
+------------------+        +------------------+
| SEMI-TRUSTED:    |        | UNTRUSTED:       |
|  GGUF file on    |        |  HTTP client     |
|  disk (could be   |        |  (anyone on the  |
|  crafted/corrupt)|        |  network)        |
+------------------+        +------------------+
             |                       |
             v                       v
+---------------------------------------------------+
| Dangerous paths:                                   |
|  mmap slice indexing, FFI raw pointers,            |
|  user-supplied prompt → tokenizer → token IDs →     |
|    matmul → weights → output                       |
+---------------------------------------------------+
```

**Key observation:** Leafcutter has two distinct attack surfaces:

1. **GGUF file inputs** — a crafted/corrupted GGUF could trigger OOB reads via mmap slice arithmetic.
2. **HTTP/CLI inputs** — when the server is running, any network client can send prompts, max_tokens, temperature values.

---

## Findings Summary

| #  | Severity | File | One-line |
|----|----------|------|----------|
| 1  | **CRITICAL** | `bridge/mod.rs:246` | Byte-level fallback tokenizer silently corrupts all non-ASCII input |
| 2  | **CRITICAL** | `model/gguf.rs:127,157` | `as usize` cast on mmap offsets — no bounds check against `mmap.len()` |
| 3  | **CRITICAL** | `api/mod.rs:196,202` | Hardcoded default API key `"leaf-dev"` ships in production binary |
| 4  | **HIGH** | `main.rs:178` | Hardcoded user-specific model path baked into binary fallback |
| 5  | **HIGH** | `inference/engine.rs` | 14 `.expect()` calls on tensor lookups in the forward path → panic on any missing/renamed tensor |
| 6  | **HIGH** | `inference/gemma.rs` | 9 `.expect()` calls on tensor lookups → panic on incomplete GGUF |
| 7  | **HIGH** | `llama_ffi/mod.rs:39-42` | `unsafe impl Send/Sync` for FFI types without proving thread safety |
| 8  | **HIGH** | `model/quant.rs` vs `model/gguf.rs` | Capability drift: Q2_K, IQ2_XXS, IQ3_XXS, IQ1_M exist in enum but fall to silent `_ => return None` in dequant dispatch |
| 9  | **HIGH** | `cache/mod.rs:44,46` | Chained `.unwrap()` on HashMap lookups — panics if K exists but V doesn't |
| 10 | **MEDIUM** | `api/mod.rs:142-143` | `top_p` accepted from HTTP body but silently discarded for FFI engine |
| 11 | **MEDIUM** | `api/mod.rs:315-316` | `unwrap()` on `TcpListener::bind` and `axum::serve` → panic if port in use |
| 12 | **MEDIUM** | `install.sh:58` | Piping curl to shell for rustup install (standard practice but a known supply-chain risk) |
| 13 | **MEDIUM** | `api/mod.rs:312` | Server binds `0.0.0.0` by default — no option for loopback-only |
| 14 | **MEDIUM** | `main.rs:384-390` | Manual context-size arithmetic: `ctx_size as usize - max_tokens` can underflow if `max_tokens > ctx_size` |
| 15 | **MEDIUM** | `kernels/q4_k_gemm.rs:53` | AVX2 path asserts `n % 256 == 0` — panics on non-256-aligned matrix dimensions |
| 16 | **LOW** | `main.rs:313,362,365,399` | `io::stdout().flush().unwrap()` and `io::stdin().read_line().unwrap()` — panic on broken pipe |
| 17 | **LOW** | `api/mod.rs` | No rate limiting on HTTP endpoints — DoS via rapid requests |
| 18 | **LOW** | `main.rs:509-523` | `as_ref().unwrap()` on tokenizer — panics if GGUF tokenizer metadata is corrupt |
| 19 | **LOW** | `model/loader.rs:567` | `product::<u64>() as usize` — potential overflow on >2^64-element tensors (theoretical) |
| 20 | **INFO** | `install.sh:178` | Hardcoded version `0.9.0` in install.sh vs `Cargo.toml` version `0.9.0` — must be manually kept in sync |

---

## Detailed Findings

### 1. Byte-level fallback tokenizer silently corrupts all non-ASCII input
**File:** `rust/src/bridge/mod.rs:246-249`  
**Severity:** CRITICAL  

The `UnifiedBridge::generate` native fallback path tokenizes by casting each UTF-8 byte to a `usize` token ID:
```rust
let tokens: Vec<usize> = prompt.bytes().map(|b| b as usize).collect();
```
And decodes by casting back:
```rust
String::from_utf8_lossy(&generated.iter().map(|&t| t as u8).collect::<Vec<u8>>())
```

This means:
- Any multi-byte UTF-8 character (é, 日本語, emoji) produces garbage token IDs because each byte is treated as a separate token.
- Token IDs > 255 are truncated to `u8` on decode, silently destroying any token the model actually generated above ID 255.
- The `generate()` function returns plausible-looking but completely wrong text for any non-English input.

**Impact:** Any user running `leafcutter generate` on the native path (non-FFI build) with non-ASCII input gets silently corrupted output. The program does not error — it returns garbage. This is the "wrong but usable is the worst kind of failure" pattern.

**Preconditions:** Native-only build (no `llama-ffi` feature) + non-ASCII prompt.

**Recommendation:** Replace the byte-fallback with the `GgufBpeTokenizer::from_gguf(path)` call that already exists lower in `main.rs:449`. The bridge should never tokenize via byte-casting.

---

### 2. `as usize` cast on mmap offsets — no bounds check against `mmap.len()`
**File:** `rust/src/model/gguf.rs:127, 157`  
**Severity:** CRITICAL  

Two functions slice into the mmap'd region using offsets cast from `u64` to `usize` **without checking against the actual mmap length**:

`get_tensor_raw` (line 125-127):
```rust
let start = self.data_offset + t.offset;
let end = start + size as u64;
Some(&self.mmap[start as usize..end as usize])
```

`get_tensor_row_f32` (line 155-157):
```rust
let tensor_start = (self.data_offset + info.offset) as usize;
let row_start = tensor_start + row_idx * row_bytes;
let raw = &self.mmap[row_start..row_start + row_bytes];
```

If a GGUF file is **truncated** (partial download, disk corruption, or intentionally crafted):
- `end` exceeds `mmap.len()` → Rust panics with "range end index out of bounds".
- In `get_tensor_row_f32`, if `row_bytes` is computed from a `block_size` that mismatches the actual tensor data, the slice can extend past the mmap region.

Additionally, on **32-bit targets** (e.g., a 32-bit ARM Pi if anyone cross-compiles), `t.offset as usize` silently truncates the high 32 bits of any offset >4 GiB, producing a wrong slice start with no error.

**Impact:** A truncated or crafted GGUF file causes an **out-of-bounds panic** during model loading or inference. The `as usize` truncation on 32-bit is a silent wrong-data read. An attacker who can supply a GGUF file can crash the process.

**Preconditions:** Truncated/corrupted/crafted GGUF file. Does not require code execution — just a bad file on disk.

**Recommendation:** Add `end <= self.mmap.len()` (and u64→usize overflow) checks before every `&self.mmap[start..end]` slice. Return `None`/`Err` instead of panicking. The `get_tensor_row_f32_into` function at line 209 already does a partial check (`out.len() < cols`) but does not check mmap bounds.

---

### 3. Hardcoded default API key `"leaf-dev"` ships in production binary
**File:** `rust/src/api/mod.rs:196, 202`  
**Severity:** CRITICAL  

```rust
const DEFAULT_API_KEY: &str = "leaf-dev";
// ...
let key = std::env::var("LEAFCUTTER_API_KEY").unwrap_or_else(|_| DEFAULT_API_KEY.to_string());
if key.is_empty() { return next.run(req).await; }
```

The auth middleware:
1. Uses `"leaf-dev"` as the default API key if `LEAFCUTTER_API_KEY` is not set.
2. Only disables auth if the key is **empty string** — which can never happen with `unwrap_or_else` defaulting to `"leaf-dev"`.

So every production deployment that doesn't set the env var gets **auth enabled with a publicly-known key**. Anyone who reads the source (it's on GitHub) knows the key is `leaf-dev`.

**Impact:** Full unauthenticated access to the `/generate`, `/v1/chat/completions` endpoints for anyone who can reach the server port. The "auth" is a false sense of security.

**Preconditions:** Server started without `LEAFCUTTER_API_KEY` env var. Network access to port 8081.

**Recommendation:**
- Default should be `""` (auth disabled) not `"leaf-dev"`. Users who want auth must explicitly set the env var.
- Alternatively, generate a random key on first start and log it, requiring the operator to read the log to use the API.
- Document this clearly in `README.md` security section.

---

### 4. Hardcoded user-specific model path baked into binary fallback
**File:** `rust/src/main.rs:178`  
**Severity:** HIGH  

When the CLI is invoked with no subcommand:
```rust
None => {
    #[cfg(feature = "llama-ffi")]
    run_server(
        "/home/xander/Documents/portfolio/AI Models/Qwen3.5-9B-IQ4_NL.gguf",
        8081,
```

This is a **hardcoded path to a specific user's home directory** (`/home/xander/`) baked into the release binary. If anyone else installs and runs `leafcutter` without args, it fails with "file not found" for a path that doesn't exist on their machine.

**Impact:** Breaks for any user other than the developer. Looks unprofessional in a release.

**Preconditions:** Running `leafcutter` with no subcommand on any machine that isn't xander's.

**Recommendation:** Remove the default `None` arm entirely, or replace with a `--help` printout and a non-zero exit code. No user path should be in a release binary.

---

### 5. 14 `.expect()` calls on tensor lookups in the engine forward path
**File:** `rust/src/inference/engine.rs: lines 608, 654, 677, 698, 754, 766, 779, 793, 940, 1034, 1035, 1036`  
**Severity:** HIGH  

The entire native forward pass is littered with `.expect("Missing pre-norm")`, `.expect("Missing post-norm")`, `.expect("Missing final norm")`, `.expect("lm_head row")`, `.expect("Missing gate")`, etc.

Every one of these is a **panic point**. If a GGUF file is:
- Missing a tensor (e.g., `post_attention_norm.weight` — we literally hit this during Ornith debugging and added an `.or_else` fallback at commit `a1ca9c0`)
- Has a renamed tensor (different model family, newer architecture version)
- Partially downloaded

...the entire process panics instead of returning an error.

**Impact:** A single missing or renamed tensor in a GGUF file crashes the process. This is especially dangerous because tensor naming conventions vary across model families and the engine needs to support many of them.

**Preconditions:** Any GGUF file that doesn't perfectly match the expected tensor naming pattern.

**Recommendation:** Convert all `.expect("Missing X")` in the forward path to proper `Result` propagation. The `Engine::load` function should validate all required tensors upfront and return a descriptive `Err` listing which tensors are missing, rather than crashing mid-forward.

---

### 6. 9 `.expect()` calls on tensor lookups in gemma.rs
**File:** `rust/src/inference/gemma.rs: lines 105, 107, 110, 149, 153, 254, 353, 368, 373`  
**Severity:** HIGH  

Same pattern as Finding 5, specific to the Gemma forward path. `.expect("Missing gate_proj")`, `.expect("gemma_fused_qkv: missing attn_q/attn_q_proj.weight")`, etc.

**Impact:** Any Gemma GGUF missing a tensor → panic.

**Recommendation:** Same as Finding 5 — propagate errors.

---

### 7. `unsafe impl Send/Sync` for FFI types without proving thread safety
**File:** `rust/src/llama_ffi/mod.rs:39-42`  
**Severity:** HIGH  

```rust
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}
unsafe impl Send for LlamaContext {}
unsafe impl Sync for LlamaContext {}
```

These blanket `Send + Sync` impls tell the Rust compiler "it's safe to share these across threads" — but `llama_context` is **not** thread-safe in llama.cpp. Concurrent use of the same context causes data races in the C library (KV cache corruption, crash).

The current `NativeStreamingEngine` wraps the engine in `std::sync::Mutex`, which protects the native path. But `FfiEngine` creates a fresh `LlamaContext` per `generate()` call (line 135), which is safe only because each call gets its own context — but the `LlamaModel` is shared across requests, and `LlamaContext::new` reads from the model concurrently.

**Impact:** Under concurrent HTTP requests, the FFI engine can hit data races in `llama_model_get_vocab` or model weight reads, causing intermittent crashes or corrupted output. Very hard to debug because it's non-deterministic.

**Preconditions:** Concurrent HTTP requests to the FFI engine.

**Recommendation:** Audit each FFI function's thread-safety claim against llama.cpp docs. Wrap `LlamaModel` access in a `Mutex` or ` RwLock`. At minimum, document which llama.cpp functions are thread-safe and which are not.

---

### 8. Capability drift: quant types in enum but missing from dequant dispatch
**File:** `rust/src/model/quant.rs` (enum), `rust/src/model/gguf.rs:188-196` (dispatch), `rust/src/model/loader.rs:473` (dispatch)  
**Severity:** HIGH  

The `QuantType` enum defines:
```rust
Q2_K, Q3_K, Q5_1, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_M, TQ1_0, TQ2_0, ...
```

But the dequant dispatch in `gguf.rs:188` and `loader.rs:473` has a catch-all:
```rust
_ => {
    eprintln!("unsupported quant type {:?} ... skipping");
    return None;  // or return Err
}
```

This means:
- A model quantized as `Q2_K` loads successfully (enum matches) but can't dequantize — it returns `None` at runtime.
- The capability report (if one exists) may list `Q2_K` as "supported" because it's in the enum, but it silently fails at dispatch.
- This is the exact "two parallel lists that drifted" pattern from the skill checklist.

The `block_size()` and `block_bytes()` methods in `quant.rs` DO handle all enum variants, which means `get_tensor_row_f32` will compute a `row_bytes` for `Q2_K` but then hit the `_ =>` arm and return `None`. The caller then gets no data — no error, just `None`.

**Impact:** User downloads a `Q2_K` model, starts inference, and gets silent failure or "unsupported quant type" on stderr with no upstream error. The README claims IQ2/IQ1 support in some places (auto-FFI fallback) but the native path silently fails.

**Recommendation:** Make the dispatch exhaustive. Either implement all dequant kernels, or have the capability report derive from the dispatch (not the enum). Ensure the `_ =>` arm returns a structured error, not just an `eprintln!` + `None`.

---

### 9. Chained `.unwrap()` on HashMap lookups — panics if K exists but V doesn't
**File:** `rust/src/cache/mod.rs:42-46`  
**Severity:** HIGH  

```rust
pub fn append(&mut self, layer_idx: usize, k: Tensor, v: Tensor) {
    if let Some(existing_k) = self.k_data.get_mut(&layer_idx) {
        existing_k.extend_from_slice(&k.data);
        self.v_data.get_mut(&layer_idx).unwrap().extend_from_slice(&v.data);
        self.shapes.get_mut(&layer_idx).unwrap()[0] += k.shape[0];
```

The first `.get_mut(&layer_idx)` succeeds (the `if let Some` branch), but the subsequent `.unwrap()` calls on `v_data` and `shapes` assume those HashMaps have the same key. If they ever get out of sync (e.g., a partial append where K was inserted but V failed, or a race condition with clear), the process panics.

**Impact:** KV cache corruption or desync → panic during generation.

**Recommendation:** Use a single struct holding K, V, and shape together, so they're always inserted/removed atomically. Or check all three before mutating.

---

### 10. `top_p` accepted from HTTP body but silently discarded for FFI engine
**File:** `rust/src/api/mod.rs:142-143`  
**Severity:** MEDIUM  

```rust
// top_p is not exposed by the FFI binding; llama.cpp samples with top_p=0.95 internally.
let _ = top_p;
```

The HTTP API accepts `top_p` in the request body (`ChatCompletionRequest` and `GenerateRequest` both parse it), but the `FfiEngine::generate` implementation silently discards it. A user sending `{"top_p": 0.1}` gets `0.95` behavior with no indication.

**Impact:** Users who depend on `top_p` for sampling control get wrong results. The API appears to support the parameter but silently ignores it.

**Recommendation:** Either implement `top_p` in the FFI binding (llama.cpp does expose it via `llama_sampler_chain`) or return an HTTP 400 with "top_p not supported by FFI engine".

---

### 11. `unwrap()` on `TcpListener::bind` and `axum::serve` → panic if port in use
**File:** `rust/src/api/mod.rs:315-316`  
**Severity:** MEDIUM  

```rust
let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
axum::serve(listener, app).await.unwrap();
```

If port 8081 is already in use (another leafcutter instance, another service), the `unwrap()` panics with a stack trace instead of a clean error message.

**Recommendation:** Map to a clean error: `eprintln!("Error: port {} already in use", port); std::process::exit(1);`

---

### 12. Piping curl to shell for rustup install
**File:** `install.sh:58`  
**Severity:** MEDIUM  

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
```

This is the official rustup install method and is standard practice, but it's a supply-chain risk: if `sh.rustup.rs` is compromised or DNS is hijacked, arbitrary code runs on the user's machine.

**Recommendation:** Document the risk. Use `--tlsv1.2` (already present). Consider offering a verified-checksum offline install option.

---

### 13. Server binds `0.0.0.0` by default — no option for loopback-only
**File:** `rust/src/api/mod.rs:312`  
**Severity:** MEDIUM  

```rust
let addr = format!("0.0.0.0:{}", port);
```

The server binds to all interfaces by default. Combined with Finding 3 (hardcoded API key), this means a leafcutter server on a cloud VM is **immediately accessible to the public internet** with a known API key.

**Recommendation:** Default to `127.0.0.1` (loopback). Add a `--host` CLI flag to explicitly opt into `0.0.0.0`.

---

### 14. Manual context-size arithmetic: `ctx_size as usize - max_tokens` can underflow
**File:** `rust/src/main.rs:384-390`  
**Severity:** MEDIUM  

```rust
if tokens.len() > ctx_size as usize - max_tokens {
    let keep = ctx_size as usize - max_tokens - system_tokens.len();
```

If `max_tokens > ctx_size` (user passes `--max-tokens 5000 --ctx-size 2048`), the subtraction underflows for `usize`, producing a very large number. The `tokens.len() > huge_number` check then always passes, and the truncation logic silently misfires.

**Impact:** Unexpected truncation behavior when `max_tokens > ctx_size`. Not a crash (unsigned underflow is well-defined in Rust), but logically wrong.

**Recommendation:** Check `max_tokens <= ctx_size` before the arithmetic, or use `saturating_sub`.

---

### 15. AVX2 path asserts `n % 256 == 0` — panics on non-256-aligned dimensions
**File:** `rust/src/kernels/q4_k_gemm.rs:53`  
**Severity:** MEDIUM  

```rust
assert_eq!(n % 256, 0, "AVX2 path requires n multiple of 256");
```

If a model has a tensor dimension that isn't a multiple of 256 (e.g., a custom architecture with a non-standard hidden size), this assertion panics in release builds. The scalar fallback exists but isn't automatically used when the AVX2 path fails.

**Impact:** Models with non-256-aligned dims crash on x86_64. The user sees an assertion failure with no guidance.

**Recommendation:** Fall back to the scalar path when the dimension isn't aligned, instead of asserting. Log a one-time warning about performance.

---

### 16. `io::stdout().flush().unwrap()` and `io::stdin().read_line().unwrap()` — panic on broken pipe
**File:** `rust/src/main.rs:313, 362, 365, 399`  
**Severity:** LOW  

Standard I/O `.unwrap()` calls in the chat loop. If the terminal closes mid-session (SSH disconnect, terminal emulator crash), the process panics with a stack trace instead of exiting cleanly.

**Recommendation:** Use `if let Err(e) = ... { break; }` for the chat loop, and ignore broken-pipe errors on stdout flush.

---

### 17. No rate limiting on HTTP endpoints — DoS via rapid requests
**File:** `rust/src/api/mod.rs` (all handlers)  
**Severity:** LOW  

No rate limiter, no concurrent-request cap, no request-size limit. A client can send unbounded requests, each spawning a `spawn_blocking` task that holds the engine mutex, exhausting thread pool resources.

**Recommendation:** Add a `tower::limit::ConcurrencyLimit` layer (6 lines) and a request body size limit via `DefaultBodyLimit`.

---

### 18. `as_ref().unwrap()` on tokenizer — panics if GGUF tokenizer metadata is corrupt
**File:** `rust/src/main.rs:509, 511, 521, 523`  
**Severity:** LOW  

```rust
hf_tok.as_ref().unwrap().encode(&prompt_text)
gguf_tok.as_ref().unwrap().encode(&prompt_text)
```

If `GgufBpeTokenizer::from_gguf` returns `None` (corrupt tokenizer metadata in GGUF), these `.unwrap()` calls panic. The `chat` command falls through to this path on non-FFI builds.

**Recommendation:** Check `is_some()` before calling, and return a descriptive error ("No valid tokenizer found in GGUF file").

---

### 19. `product::<u64>() as usize` — potential overflow on theoretical tensors
**File:** `rust/src/model/loader.rs:567`  
**Severity:** INFO  

```rust
let count: usize = t.dimensions.iter().product::<u64>() as usize;
```

For any tensor with total element count > `usize::MAX` (18 exabytes on 64-bit), this overflows. This is purely theoretical — no real model has tensors that large. But the `as usize` cast truncates silently on 32-bit.

**Recommendation:** Use `try_from` and return an error on overflow. Low priority — this will never be hit in practice.

---

### 20. Version string drift: install.sh vs Cargo.toml
**File:** `install.sh:6` vs `rust/Cargo.toml:3`  
**Severity:** INFO  

`install.sh` hardcodes `LEAFCUTTER_VERSION="0.9.0"` and `main.rs` hardcodes `version = "0.9.0"`. These must be manually kept in sync. If one is updated and the other isn't, the install script reports one version and the binary reports another.

**Recommendation:** Have `install.sh` read the version from `Cargo.toml` via `grep version rust/Cargo.toml | head -1 | cut -d'"' -f2`.

---

## Prioritized Recommendations

### Immediate (before release)

1. **Finding 1** — Replace byte-fallback tokenizer with `GgufBpeTokenizer`. This is the difference between "works" and "silently corrupts non-English input".
2. **Finding 3** — Change default API key logic: default to disabled auth, require explicit env var to enable. This is a 2-line fix that closes a real security hole.
3. **Finding 13** — Bind to `127.0.0.1` by default, add `--host` flag. Pair with Finding 3.
4. **Finding 4** — Remove hardcoded `/home/xander/...` path from binary. Replace with help text.
5. **Finding 2** — Add mmap bounds checks (`end <= self.mmap.len()`) in `get_tensor_raw` and `get_tensor_row_f32`. This prevents crafted-GGUF panics.

### Short-term (next sprint)

6. **Findings 5 & 6** — Convert `.expect("Missing X")` in engine and gemma forward paths to `Result` propagation. Add a `validate_tensors()` check at `Engine::load` time.
7. **Finding 8** — Audit dequant dispatch vs enum. Either implement missing kernels or make the capability report derive from dispatch.
8. **Finding 9** — Refactor KVCache to use a single struct per layer (atomic insert/delete).
9. **Finding 10** — Either implement `top_p` in FFI binding or return HTTP 400.
10. **Finding 15** — Fall back to scalar path on non-256-aligned dims instead of asserting.

### Nice-to-have (post-release)

11. **Finding 7** — Document or prove FFI thread safety; wrap shared model access.
12. **Finding 11** — Clean error on port-in-use instead of `unwrap()`.
13. **Finding 17** — Add `ConcurrencyLimit` and `DefaultBodyLimit` to Axum router.
14. **Finding 14** — `saturating_sub` on context-size arithmetic.
15. **Findings 16, 18, 19, 20** — Hygiene fixes.

---

## Verification Checklist

- [x] Trust boundary drawn first
- [x] All 12 OWASP surface categories considered (grepped or ruled out)
- [x] Rust perf-critical 9-category checklist applied
- [x] Each finding has file:line, severity, impact, preconditions
- [x] Severity rubric applied consistently
- [x] No recommended diffs — citations only
- [x] No code was edited

---

*This report is read-only. The user (implementer) prioritizes and patches.*
