# airllm-go Audit Record

**Date:** 2026-04-21  
**Auditor:** Senior Go Engineer & Code Auditor  

---

## Audit Summary

The Go rewrite (`airllm-go`) correctly captures the high-level architecture of the Python `airllm`:  layer-by-layer inference with prefetching, safetensors loading, quantization, and KV caching scaffolding. However, **four critical stub functions** made the program unable to perform any real inference. Several additional concurrency, numeric, and logic bugs were found.

Translation quality: **Architecture ✅ | Numerics ❌ (before fixes) | Concurrency ⚠️ (goroutine leak) | Completeness ❌ (stubs)**

---

## Weaknesses Found

| ID  | Severity | Description |
|-----|----------|-------------|
| C-1 | 🔴 Critical | `embedLookup()` was a `return nil, error` stub |
| C-2 | 🔴 Critical | `scaledDotProductAttention()` returned `v` unchanged — no attention computed |
| C-3 | 🔴 Critical | `layerNorm()` returned input unchanged — no normalization |
| C-4 | 🔴 Critical | `Tokenizer` type referenced but not defined — compile error if used |
| H-1 | 🟠 High | Prefetch goroutine leak: goroutine for layer 1 launched at i=0 but channel not read until i=1; last-layer goroutine blocked forever |
| H-2 | 🟠 High | `matmul()` used wrong weight dimension (`Shape[-2]` instead of `Shape[-1]`) for HuggingFace `[out, in]` convention |
| H-3 | 🟠 High | `mul()` no shape validation — silent wrong results on mismatched tensors |
| M-1 | 🟡 Medium | `Float16ToFloat32` subnormal: `float32(-1)*float32(sign)*...` produced wrong sign |
| M-2 | 🟡 Medium | `Transpose` N-D fallback `copy(result.Data, t.Data)` — data not permuted |
| M-3 | 🟡 Medium | `isLMHead`: `contains(name,"lm_head") \|\| contains(name,"lm_head")` — duplicate predicate |
| M-4 | 🟡 Medium | `NewTensor` allowed zero/negative dimensions with no error |
| L-1 | 🟢 Low | Hand-rolled `contains()` was buggy and slow |
| L-2 | 🟢 Low | Periodic `runtime.GC()` every 10 layers — not targeted enough |
| L-3 | 🟢 Low | Interactive `fmt.Scanln` — truncates multi-word prompts at first space |
| L-4 | 🟢 Low | `LayerNorm.Load` ignored `.bias` tensors |

---

## Improvements Applied

1. **`embedLookup`** — Full implementation reading `int64` token IDs, validating range, doing row lookup in the embedding matrix.

2. **`scaledDotProductAttention`** — Full `softmax(QK^T/sqrt(d)) × V` with causal masking.

3. **`layerNorm`** — Full mean/variance normalization with weight scale and bias.

4. **`Tokenizer` stub** — Added minimal struct so the package compiles.

5. **Prefetch lifecycle** — Restructured: load layer 0 directly, kick off prefetch for layer 1 after that, consume prefetch results at `i > 0`, never launch prefetch for the last layer.

6. **`matmulTransposed`** — Replaced `matmul()` with `matmulTransposed()` implementing `input @ weight^T` matching HuggingFace `[out, in]` weight layout.

7. **`mulElemwise`** — Added size check, returns error on mismatch.

8. **`Float16ToFloat32`** — Full IEEE 754 bit construction for all cases (zero, subnormal, normal, infinity, NaN).

9. **`Transpose`** — General N-D element permutation via stride decomposition.

10. **`isLMHead`** — Second predicate is now `"output_layer"` (ChatGLM).

11. **`NewTensor`** — Panics with a clear message if any dimension ≤ 0.

12. **`contains`** → `strings.Contains` — stdlib, correct, faster.

13. **GC strategy** — `debug.FreeOSMemory()` after each `Unload()` instead of `runtime.GC()` every 10 layers.

14. **Interactive input** — `bufio.Scanner.Scan()` reads full lines.

15. **`LayerNorm.Load`** — Now captures both `.weight` and `.bias`.

---

## Performance Notes

- **Prefetch fix** eliminates goroutine accumulation under load; previously `N-1` goroutines could be blocked indefinitely across a forward pass.
- **`matmulTransposed`** still uses the naive O(B·M·N·K) loop. For production, replace with a BLAS call (e.g., `gonum.org/v1/gonum/blas/blas64`) or use `unsafe` + SIMD intrinsics.
- **`debug.FreeOSMemory()`** is heavier than `runtime.GC()` but more effective after weight unloads since it returns pages to the OS immediately — critical for the low-RAM use case.
- **N-D Transpose** is O(total elements) with per-element coordinate decomposition. For the common 2D and 4D cases a specialized path would be ~5× faster; left as a future optimization.
- **`scaledDotProductAttention`** is O(B·H·S²·D) — correct but O(S²) in memory. A chunked or Flash-Attention style approach would be needed for long contexts.
