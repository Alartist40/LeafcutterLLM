# How to Add a New Architecture to LeafcutterLLM — A Methodology

> **Wrap-up note (2026-08-18):** This methodology record remains the reference
> for adding any future GGUF architecture. The most recent architecture added
> via this process is Qwen3.5 / Ornith (native, shipped); MoE routing is now
> **real-model-validated** on `ornith-1.0-35b` (Qwen3.6, 256 experts) with
> quantized on-demand expert slicing (2026-08-18, see `CHANGELOG.md`).
> Kimi K2.6 and GLM-5.2 scaffolding (MLA + MoE) is still not real-model-validated.

This is a record of how Milestone 1 of "add native support for Kimi K2.6
and GLM-5.2 to LeafcutterLLM" was actually carried out. The intent is
that a future agent (or human) who knows nothing about these particular
model families can use this document to figure them out and do the same
job for *any* new GGUF model. The audience is a competent engineer who
hasn't yet read the architecture's paper but who knows what GGUF,
forward-pass, RMSNorm, RoPE, and KV cache are.

---

## 0. Operating posture

Before any code is written, three rules:

1. **Don't trust the docs in the repo.** They are written by hand and
   drift. Treat the *real GGUF on disk* as the source of truth, and use
   the docs for orientation only. (The user told us this explicitly.)

2. **Don't trust the conversation context.** When the work started, no
   verification had been done that Kimi and GLM really were what they
   were described as. So the very first task is to look at the file —
   not at what downstream chat summaries say.

3. **Build smallest valid first.** Add the structure (enum, scaffolding
   script, intake-checklist) before any compute. Then a Python
   reference. Then small Rust math. Then port to Rust. *Then* run on
   real weights — and only then is the "did this work?" question even
   well-posed.

The rest of this document is the actual steps of that process, in order.

---

## 1. Locate the model and inspect the actual bytes

The raw artifact is a GGUF file on disk. For multi-shard GGUFs the
header is in shard 1; the other shards are pure weight payloads.
Both Kimi and GLM are *split* across 14 and 11 shards respectively —
so an attempt to "load the model" on whichever shard was on hand would
just give tensor table fragments and no metadata.

### What to do

```sh
ls /mnt/ssd/Xander/AI\ Models/ | head -50
```

Pick the shard-1 file. (This requires shard 1 actually being on disk;
if only a tail shard is, you can still learn *something* from the
tensor-table slice but not the numbers — see §1.b for that fallback.)

### What you get from shard 1

Shard 1 is `~7 MB` for these models because it carries:
- the magic + version
- the kv-pair count
- all metadata kv pairs (`general.architecture`, `block_count`,
  key/model dims, expert counts, RoPE config, etc.)
- the tensor count (usually 0 in shard 1 of a split model — tensors are
  distributed across shards)
- the placeholder for the data section

### What you can do *without* real metadata

If only tail shards are present, fall back to **tensor-name
fingerprinting**: extract the names of all tensors visible in any shard,
then match the pattern against known architectures. For DeepSeek-2 /
DeepSeek-V3, the names tell you everything architecturally. For each
model family there are 5-15 characteristic weight suffixes.

---

## 2. Read the GGUF header yourself

The repo has a parser for GGUF and it works, but for an unfamiliar
binary the first time, write your own minimal reader so you understand
the format end-to-end.

GGUF v3 structure (memory-mapped reference):

```
[MAGIC: 'GGUF'                       4B]
[VERSION: u32  (=3 for these models)  4B]
[N_TENSOR: u64]   ← field metadata; 0 for weight shards, full count
                   for the header shard
[N_KV:     u64]   ← number of metadata key-value pairs to walk
[metadata block: N_KV × (key + value)]
[tensor-info block: N_TENSOR × (name + ndim + dims + quant_type + offset)]
[alignment padding]
[weight data section]
```

### Value-type table (do not guess)

Right at the start you'll get the types wrong. The actual enum is from
llama.cpp's `gguf.h`; do not extrapolate from "string" → 8 by analogy
with JSON:

| Type code | Meaning  | Bytes |
|-----------|----------|-------|
| 0         | UINT8    | 1     |
| 1         | INT8     | 1     |
| 2         | UINT16   | 2     |
| 3         | INT16    | 2     |
| 4         | UINT32   | 4     |
| 5         | INT32    | 4     |
| 6         | FLOAT32  | 4     |
| 7         | BOOL     | 1     |
| 8         | STRING   | 8 + len |
| 9         | ARRAY    | nested: (u32 elem_type, u64 len) |
| 10        | UINT64   | 8     |
| 11        | INT64    | 8     |
| 12        | FLOAT64  | 8     |

Strings have a 64-bit length prefix *before* the bytes; arrays have a
nested child type. Once the table is right, everything falls into place.

### Strings of various length

- Short scalars (architecture name, vocab size): read directly.
- The vocab tokens list (`tokenizer.ggml.tokens`, often 100k+ entries)
  is an ARRAY of STRING. Don't try to retain it all — your parser
  will OOM. Detect large string-arrays and skip the elements after
  recording the length, or break a "byte-cap" boundary after the
  vocab size appears.

### A reliable parser discipline

1. Compute `pos0 = file.tell()` after reading the top-of-header.
2. Set a byte budget (e.g. 200 KB) on how far the metadata walker will
   read. After that, assume you've gotten everything important (dims,
   expert counts, RoPE).
3. On read failure or value-type out of dictionary, **stop** rather
   than guess. A failure mid-stream means the cursor is now in the
   middle of garbage; continuing produces nonsense.

The 2nd-version of the parser at the top of `scripts/intake_gguf.py` is
a worked example. After the in-line type table and the array-of-large-
strings bail-out it works for both Kimi and GLM in well under a second.

---

## 3. Architecture *detection* vs architecture *implementation*

These are different stages. Detection is "I know this string means
DeepSeek-2." Implementation is "I can do a forward pass on it."
Each model in LeafcutterLLM goes through three phases:

1. **Add to the detection layer** — `ModelArchitecture` enum +
   `known_extra_suffixes` + capability report. The engine still
   can't run the model but it can identify it and emit a non-blocking
   `research` or `unsupported` verdict.
2. **Build a Python reference forward pass** for the new attention and
   FFN structures. This is the math oracle.
3. **Port the Python reference to Rust** into inference modules under
   `src/inference/`.
4. **Wire the new modules into `forward_native`** as new branches —
   `has_mla`, `has_moe`, etc. — without breaking the existing
   `has_standard_attn` / `has_deltanet` / `has_ssm` chain.
5. **Validate** the Rust forward math against the Python reference for
   small random tensors, then against actual model layer-0 logits
   from the GGUF.

Phases 1, 2 and 3 can happen in parallel for two architectures that
share math (Kimi and GLM both need DeepSeek-2-family MLA + MoE).

---

## 4. What both model metadata tables look like

When shard 1 is on disk, expect to see entries like:

```
general.architecture          = "deepseek2" or "glm-dsa"
deepseek2.block_count         = 61
deepseek2.embedding_length    = 7168
deepseek2.expert_count        = 384
deepseek2.expert_used_count   = 8
deepseek2.expert_shared_count = 1
deepseek2.attention.head_count           = 64
deepseek2.attention.head_count_kv        = 1
deepseek2.attention.key_length           = 576   (qk_nope + qk_rope)
deepseek2.attention.key_length_mla       = 192   (qk_nope only)
deepseek2.attention.value_length         = 512
deepseek2.attention.value_length_mla     = 128
deepseek2.attention.q_lora_rank          = 1536
deepseek2.attention.kv_lora_rank         = 512
deepseek2.rope.dimension_count           = 64
deepseek2.rope.freq_base                 = 50000.0
deepseek2.rope.scaling.type              = "yarn"
deepseek2.context_length                 = 262144
deepseek2.leading_dense_block_count      = 1
deepseek2.expert_weights_scale           = 2.827
```

GLM-DSA duplicates the same shape with different magnitudes:

```
glm-dsa.attention.indexer.head_count    = 32   (sparse-indexer heads)
glm-dsa.attention.indexer.top_k         = 2048 (sparse top-k)
glm-dsa.nextn_predict_layers            = 1   (MTP heads)
glm-dsa.context_length                  = 1048576
glm-dsa.rope.freq_base                  = 8000000.0
glm-dsa.rope.scaling                    = none — YaRN absent
```

The intake script `scripts/intake_gguf.py` is the tool that prints this
in a uniform human-readable form. It handles both families and is a
good template: print every metadata key that contains the architecture
prefix, sort numerically when possible, and always emit a
`native_support` verdict.

---

## 5. The architectural facts that drive everything

Even before paper-reading, two inferences are obvious from the metadata:

**(A) MLA, not MHA:** when the
metadata exposes `q_lora_rank`, `kv_lora_rank`, `key_length_mla`,
`value_length_mla`, and the sum
`qk_nope_head_dim + qk_rope_head_dim = full key_length`, that's
multi-latent attention. The K and V are not stored in full
(n_heads × head_dim); they're stored as a small latent
(kv_lora_rank) and reconstructed for each head via `k_b` / `v_b`
matrices. Same trick for Q.

**(B) Routed MoE:** `expert_count` large (256 or 384),
`expert_used_count` small (8), `expert_shared_count` = 1, plus the
exclusive suffix set
`ffn_gate_inp.weight` / `ffn_gate_exps.weight` / `ffn_up_exps.weight` /
`ffn_down_exps.weight` / `ffn_*_shexp.weight` /
`exp_probs_b.bias`. The shared-expert branch plus the per-token
router tells you almost everything you need.

You can confirm (A) and (B) without reading the paper; the next step is
to confirm them against llama.cpp's reference implementation, which is
vendored at `rust/llama.cpp/`.

---

## 6. Build the Python reference *before* any Rust

Before patching the engine, write a numpy-based math oracle for the
new layer types. `scripts/ref_mla_moe.py` is the template. The
motivation is simple: numpy is short, you can hand-trace numbers
through it, and you get a ground-truth answer that Rust can be
compared against.

For MLA, the forward is structurally identical to standard attention
once the projection step is decomposed. The layers in the Python
reference are:

```python
def mla_attention_forward(hidden, q_a, q_a_norm, q_b, kv_a_mqa,
                          kv_a_norm, k_b, v_b, attn_output,
                          *, num_heads, num_kv_heads,
                          qk_nope_head_dim, qk_rope_head_dim,
                          v_head_dim, rope_theta, eps):
  # Step 1: Q down → norm → up
  q_lat = rms_norm(hidden @ q_a.T, q_a_norm, eps)
  q = (q_lat @ q_b.T).reshape(..., num_heads,
                              qk_nope_head_dim + qk_rope_head_dim)
  q_nope, q_rope = q.split([qk_nope_head_dim], -1)

  # Step 2: KV compressed path with absorbed rope
  kv_full = hidden @ kv_a_mqa.T
  kv_lat, k_rope_raw = kv_full.split([kv_lora_rank], -1)
  kv_lat = rms_norm(kv_lat, kv_a_norm, eps)
  k = (kv_lat @ k_b.T).reshape(..., num_kv_heads, qk_nope_head_dim)
  v = (kv_lat @ v_b.T).reshape(..., num_kv_heads, v_head_dim)

  # Step 3: tile the GQA groups, apply RoPE
  k = np.repeat(k, num_heads//num_kv_heads, axis=1)
  v = np.repeat(v, num_heads//num_kv_heads, axis=1)
  q_rope = apply_rope(q_rope, rope_theta)
  k_rope = apply_rope(np.broadcast_to(k_rope_raw[...,None,:],
                                      (..., num_heads, qk_rope_head_dim)),
                      rope_theta)

  # Step 4: standard scaled-dot, causal mask, weighted sum
  S = q @ k.transpose(-1,-2) / sqrt(qk_nope_head_dim + qk_rope_head_dim)
  S = apply_causal_mask(S)
  P = softmax(S)
  out = P @ v
  return out @ attn_output.T
```

For MoE: the forward is a top-k sigmoid routing, a sum of the
top-k experts' contributions, plus an additive shared-expert branch.
Output shape `[seq_len, hidden]`.

The point of writing both is that they show in ~150 lines what the Rust
port will take *much* more code to express — and they let you run
controlled experiments:

```sh
python3 scripts/ref_mla_moe.py --random
# MoE out mean: 0.103 max-abs: 4.36
# MLA out mean: 0.019 max-abs: 0.79
```

These baseline numbers are what later tests of the Rust implementation
will compare against.

---

## 7. Translate into Rust with tests at every step

The Rust port of MLA will live at `src/inference/mla.rs`, mirroring
`ssm.rs` and `deltanet.rs`. MoE already has a stub at
`src/inference/moe.rs`.

### Don't over-engineer the first pass

The first Rust version should:
- use the existing `Tensor` API (`matmul`, `add`, `rms_norm`, `silu`)
- never assume f32 paths if the engine has quantized weights — every
  weight should slice off and dequantize at the very start of the
  layer, never inline
- keep `forward_native()`'s existing branching order; new branches
  are appended at the end, never inserted before existing branches
  in a way that could break already-validated models.

### Tests at every step

Three test tiers:

1. **Per-component unit tests** (in `/#[cfg(test)] mod tests` inside
   each inference file). Examples added in Milestone 1:
   `sigmoid math`, `topk ordering`, `MoE config defaults`,
   `arch-detection string parsing`.

2. **Per-layer math test against the Python reference** — after you
   have a Rust MLA module, do an f32 nested-loop forward on random
   tensors with known dims, and compare against `scripts/ref_mla_moe.py`.
   `cosine_sim > 0.99` after one forward is the rough gate.

3. **Real-model logit test** — only at the end, only when the full
   shards are on disk. Compare one forward pass against llama.cpp's
   reference logits for layer 0 of the real model. Cosine similarity
   typically is > 0.95 if the math is right; lower means look for a
   transposed-quant issue, a RoPE rotation direction, or an RMSNorm
   epsilon mismatch.

---

## 8. The fix-the-existing-build-baseline discipline

Before adding *new* code, make sure the *existing* build is green.
Branch-perfect habitual mistake: an agent edits `model/arch.rs` to
add a new enum variant, the build breaks, the agent doesn't notice
because they're focused on the variant. But the audit pass left the
code at "123 tests pass" — which for our context means `cargo test
--release --lib --no-default-features` was green, but
`cargo build --release --bin leafcutter` was already broken on
pre-existing trait-resolution and arity bugs in `src/main.rs` and
`src/bin/check_tok.rs`.

The test count is a floor of coverage, not a certification of
correctness. To check the build state of a project this big, look at:

1. `cargo check --release --lib --no-default-features`
2. `cargo build --release --bin leafcutter` *(the CLI)*
3. `cargo build --release --bin <vital-debug-binary>` *(e.g. test_arch)*
4. `cargo test --release --lib --no-default-features`

If any of those fail before your changes started, fix it as part of
your changes — otherwise you're building on a broken branch.

Real fixes from this round: `BaseTokenizer` trait import; the
two-argument `tok.decode()` arity; cli's `commands` arm matching.
None of those changes affects what the program *does*; they only fix
what it *can compile.*

---

## 9. Documentation is part of the deliverable

After each milestone, all four of these must be updated:

- `README.md` (user-facing): does the new architecture show up in the
  Supported Models table? Does the install/quick-start still apply?
- `CHANGELOG.md` (release notes): new version section with the
  Added / Fixed / Deferred sub-lists, list of every test added,
  every file created.
- This doc (`MODEL_INTAKE_METHOD.md`): a new section at the end
  describing what landed and what didn't.

The user told us upfront *"do not break the program that we have
built"* — which means documented behavior must match actual behavior.
A change undocumented is, for our purposes, a change unmade. This is
also the discipline that catches the "the .md files go out of
date" failure mode the user explicitly flagged.

---

## 10. Git discipline

When the work is small, commit per logical milestone. For a new
architecture that's:

- **One commit per milestone** (M1, M2, M3, ...).
- Message format: `vX.Y.Z: short description`. Then a body listing
  every artifact (new files, modified files, added tests, deleted
  files), and what's deferred.
- Always run the test suite *once* before committing:
  `test result: FAILED. 131 passed; 1 failed; 3 ignored; 0 measured`
  ≡ "129 new passed + the pre-existing one, unchanged".

Files that are gitignored (`rust/src/bin/check_*.rs`,
`rust/src/bin/test_*.rs`, `*gguf`, `coverage.out`, etc. — see
`.gitignore`) deserve a mention in the commit body but don't get
committed. Don't fight `.gitignore` with `-f` unless the file truly
belongs in the repo (e.g. `intake_gguf.py` – a real scripts file,
not a diagnostic test).

---

## 11. A checklist template for any new architecture

Reproducible recipe for *any* future GGUF model:

```
[ ] Locate the GGUF on disk.  Confirm shard-1 is present.
[ ] Run scripts/intake_gguf.py against the file.  Confirm:
      - family detected correctly
      - native_support=research or higher
      - dim table matches the model's paper / vendor blog
[ ] Read metadata; identify attention type (MHA/GQA/MLA/MQA),
    FFN type (dense / MoE / shared-MoE), and any extras (MTP,
    sparse attention indexer, RoPE scaling, LongRoPE).
[ ] Write a Python reference forward for both the new attention
    and the new FFN.  Run it on random tensors; record baseline
    mean/max-abs output statistics.
[ ] Add `ModelArchitecture::*` enum variant + detection entry
    + unit tests in src/model/arch.rs.
[ ] If attention type is new:
    [ ] Author src/inference/<attention>.rs (mirror ssm / deltanet /
        moe structure).
    [ ] Add 3-5 unit tests against the Python reference (random
        tensors, f32 cosine similarity > 0.99).
[ ] If FFN type is new:
    [ ] Author src/inference/<ffn>.rs.
    [ ] Add 3-5 unit tests against the reference.
[ ] Wire new modules into src/inference/engine.rs::forward_native()
    as new branches.  Append, never replace.
[ ] Add Python-reference comparison work for the *whole layer*:
    scripts/ref_layer.py --arch <name>
[ ] When full shards arrive, scripts/ref_real_layer.py: run the
    actual model layer 0 against llama.cpp's reference.  Cosine
    similarity > 0.95 = math verified.
[ ] Update README, CHANGELOG, handoff doc.  Commit.
```

The whole point is to make the next addition *boring*. After Keni
K2.6 and GLM-5.2, the next candidate is whatever the user downloads
next — Mistral Large 3, Qwen3-Max, Llama-4, a MoE-by-Apple model,
whatever — and the only thing that changes is the attention and FFN
block content; the surrounding machinery (layer-streaming,
madvise, tokenizer, cache, kernels, API server) is already paid for.

---

## 12. What I should have done differently on milestone 1

Personal note for the future agent:

- Don't claim the program "works" on a model you've only scaffolding.
  Distinguish "math is correct" from "validated against a real
  model." Milestone-1 was a scaffolding milestone, but the closing
  summary overstated readiness. Reserve "works" for actual end-to-end
  validation.

- Don't over-rely on the existing test count as a coverage signal.
  `cargo test --release --lib --no-default-features` covers library
  tests only; the binaries often have unnoticed breakage. Run all
  four build checks (lib check, main build, vital binary build,
  test lib) before declaring baseline green.

- Keep detailed records in this methodology doc, not in chat. Chat
  rolls over; this file persists. The investment of writing it down
  is paid back every time a new model comes along.

---

*End of methodology. See also `CHANGELOG.md` for the release history
and `ARCHITECTURE.md` for the colony design. The Kimi K2.6 / GLM-5.2
intake scaffold is recorded in the CHANGELOG 2026-06-19 sections.*
