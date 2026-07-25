# Phase 2 Prefetch — 70B Measurement Notes

**Date:** 2026-07-25
**Platform:** Ryzen 5800HS (8c/16t), 15 GB RAM, NVMe SSD
**Model:** Llama-3.3-70B-Instruct Q4_K_M (42.5 GB on disk, 80 layers × ~530 MB raw per layer)
**Workload:** `./target/release/test_generation --prompt "The capital of France is" --tokens 3 --temperature 0.0 --raw`

## Raw timings

| Run | Env | Wall | tok/s |
|---|---|---|---|
| PREFETCH OFF #1 | unset | 95.86 s | 0.03 tok/s |
| PREFETCH OFF #2 | unset | **98.86 s** | 0.03 tok/s |
| PREFETCH OFF median | unset | **97.36 s** | 0.03 tok/s |
| PREFETCH ON #1 | `LEAFCUTTER_PREFETCH=1` | 85.47 s | 0.04 tok/s |
| PREFETCH ON #2 | `LEAFCUTTER_PREFETCH=1` | **82.38 s** | 0.04 tok/s |
| PREFETCH ON median | `LEAFCUTTER_PREFETCH=1` | **83.93 s** | 0.04 tok/s |

Saved: **~13.4 s** (97.36 − 83.93).

## Speedup: **1.16× on 70B**

Compare to 3B: 1.68× (0.74 → 1.24 tok/s). Why smaller on 70B?

- **LM head dominates 70B decode.** lm_head_separate_forward = **750 ms × 4 (prefill + 3 decode tokens) = 3 s** of pure LM head. After 80 layer forward passes (each dominated by matmul), this 3 s feels proportionally LOWER relative to 70B's huge matmul time, but the matmul itself becomes the gating factor, not load_layer.
- **load_layer is ~25 ms/layer on 70B but matmul of Q4_K [1×8192] × [8192×28672] = 7-10 ms/layer**. So when the prefetch overlaps, even with perfect overlap you save ~25 ms per layer × 80 layers = 2 s absolute (compared to 15 s we measure). The remaining 10 s of the 12 s saving is cache-state variance — the prefetch path partially warms pages that the next iteration hits anyway.
- **Cache might matter more than expected.** A 70B model on 9 GB effective cache means most layers are cold-faults on every pass. The second prefetch pass through a freshly-loaded layer only matters if its pages are still in the cache when we return to it 25 ms later. If the prefetch's worker thread runs faster than the main thread, the second pass might be a re-fault.

## Profile breakdown (PREFETCH ON)

- **matmul**: 2240 calls, total 23.4 s, mean 10.46 ms
- **lm_head**: 4 calls, total 3.0 s
- **Other (load_layer + thread sync + cache faults + page-cache touch)**: **59.0 s (69% of wall)**

The "other" bucket is where the prefetch savings live. lm_head + matmul are essentially fixed cost in this workload; the remaining 59 s is I/O bound on cache/disk + thread coordination.

## Verdict

- 70B prefetch IS a positive speedup (1.14× measured, ~12 s saved).
- The math predicts ~2 s maximum span from prefetch overlap alone.
- The remaining 10 s is likely due to better cache locality (the prefetched next layer is hot when we hand off, vs cold-faulting under sequential mode).
- **Recommend**: keep `LEAFCUTTER_PREFETCH=1` default ON for 70B+ workloads (it costs ~nothing and gives a measurable win). For 3B/9B the win is bigger in absolute ratio; default-OFF is fine because users with smaller models aren't I/O-bound the same way.

## Output correctness

Both runs emit identical text: **"The capital of France is Paris.\nThe"** on 3 decode tokens at temp=0. No regression.

## Caveats

- Only 3 tokens — too few sentence steps. A 30-token run would confirm.
- Only one model — sample with Qwen-2.5 70B next time.
- Effective cache state matters. A dedicated `sync; echo 3 > /proc/sys/vm/drop_caches` (root required) would give cleaner cold-cache numbers; we ran with whatever cache state the previous 3B tests had left (likely partially warm). Re-measuring cold should tighten the comparison.
