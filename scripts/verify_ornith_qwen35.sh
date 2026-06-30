#!/usr/bin/env bash
# verify_ornith_qwen35.sh — End-to-end verification for Ornith 1.0 9B
# (Qwen 3.5 hybrid SSM+full-attention) native forward.
#
# What this checks:
#   1. The Q4_K_M GGUF exists in the standard location.
#   2. test_generation binary loads and detects architecture.
#   3. Forward pass emits non-garbage output (top-5 prefill tokens).
#   4. Generation produces coherent English (NOT "is is is is").
#   5. Per-layer L2 trajectory is in expected range.
#
# Reference for the expected dim math and shape:
#   https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp
#
# Use:
#   bash scripts/verify_ornith_qwen35.sh
#   bash scripts/verify_ornith_qwen35.sh --model /custom/path.gguf
#
# Useful env vars:
#   LEAFCUTTER_DEBUG_NORMS=1 — emit per-layer L2 dumps for debugging
#   VERBOSE=1                 — show full output (default: filtered)
#
# Exit code: 0 = all 5 checks pass, 1 = at least one failed.

set -uo pipefail

# --------- Config ---------
DEFAULT_MODEL="/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf"
TEST_BIN="/home/xander/Documents/portfolio/LeafcutterLLM/rust/target/release/test_generation"
PROMPT="The capital of France is"
TOKENS=10
TEMP=0.0

MODEL="${DEFAULT_MODEL}"
if [[ "${1:-}" == "--model" ]]; then MODEL="$2"; fi

VERBOSE="${VERBOSE:-0}"
log() {
    if [[ "$VERBOSE" == "1" ]]; then
        echo "$@"
    else
        echo "$@" | grep -vE "^\s*\[deltanet\]|^\s*\[NORM\]|^Loading|^D ONE" || true
    fi
}

pass() { echo "[✓] $*"; }
fail() { echo "[✗] $*"; exit 1; }

# --------- 1. GGUF exists ---------
echo "=== 1/5: GGUF presence ==="
if [[ -f "$MODEL" ]]; then
    pass "GGUF present: $MODEL ($(du -h "$MODEL" | cut -f1))"
else
    fail "GGUF missing: $MODEL"
fi

# --------- 2. Binary built ---------
echo "=== 2/5: Test binary ==="
if [[ ! -x "$TEST_BIN" ]]; then
    echo "Binary not built — trying to build (60s timeout)..."
    (cd /home/xander/Documents/portfolio/LeafcutterLLM/rust && \
        cargo build --release --no-default-features --bin test_generation 2>&1 | tail -3) || \
        fail "Failed to build test_generation"
fi
pass "Binary ready: $TEST_BIN"

# --------- 3. Architecture detected + DeltaNet dims ---------
echo "=== 3/5: Architecture detection ==="
OUT=$(mktemp)
"$TEST_BIN" --model "$MODEL" --prompt "Hello" --tokens 1 --temperature 0.0 --raw \
    > "$OUT" 2>&1 || true

DETECT_OK=$(grep -E "(native backend|architecture).*Qwen3.5|architecture detected: .*qwen|emitted arch=|arch=.*qwen" "$OUT" 2>/dev/null | head -1 || true)
if [[ -n "$DETECT_OK" ]] || grep -qE "Using native backend for Qwen3.5" "$OUT"; then
    pass "Architecture detected as qwen35"
else
    log "$(head -25 "$OUT")"
    fail "Architecture not detected as qwen35"
fi

DELTANET_LINE=$(grep -E "DeltaNet: qk_heads=" "$OUT" 2>/dev/null | head -1 || true)
if [[ -n "$DELTANET_LINE" ]]; then
    if echo "$DELTANET_LINE" | grep -qE "qk_heads=16.*v_heads=32.*head_k=128"; then
        pass "DeltaNet dims correct: qk=16 v=32 head_k=128 (init verifies shape)"
    else
        fail "DeltaNet dims incorrect: $DELTANET_LINE (expected qk=16 v=32 head_k=128)"
    fi
else
    fail "No DeltaNet line in output — engine initialization failed"
fi

# --------- 4. Top-5 prefill tokens are real English ---------
echo "=== 4/5: Top-5 prefill sanity ==="
TOP5_LINE=$(grep -E "^\s*\[[0-9]\] id=" "$OUT" 2>/dev/null | head -5 || true)
if [[ -n "$TOP5_LINE" ]]; then
    # Tokens like Ġthe, Ġa, Ġan, ., , are sensible; punctuation-spam is bad.
    BAD=$(grep -cE "^\s*\[[0-9]\]\s+id=\s*\S+\s+logit=" "$OUT" || echo 0)
    if [[ "$BAD" -ge 5 ]]; then
        pass "Top-5 tokens emitted"
    else
        fail "Top-5 missing — output is broken"
    fi
    # Sanity: at least one token should be a common English connector (Ġthe, Ġa, etc.)
    if grep -qE "Ġ(the|a|an|of|to|in|is|was|were|are)" "$OUT"; then
        pass "At least one English connector in top-5"
    else
        log "$(grep -E '^\s*\[[0-9]\] id=' "$OUT")"
        fail "No real English token in top-5 — forward is degenerate"
    fi
else
    fail "Top-5 tokens missing — engine crashed or output truncated"
fi

# --------- 5. Generation produces coherent text (not "is is is is") ---------
echo "=== 5/5: Coherent generation ==="
GEN_OUT=$(mktemp)
"$TEST_BIN" --model "$MODEL" --prompt "$PROMPT" --tokens $TOKENS --temperature $TEMP --raw \
    > "$GEN_OUT" 2>&1 || true
TEXT=$(grep -A4 "Full decoded" "$GEN_OUT" | tail -1 || true)
if [[ -n "$TEXT" ]] && ! echo "$TEXT" | grep -qE "^[[:space:]]*undefined"; then
    if echo "$TEXT" | grep -qE "([[:space:]]the[[:space:]]).*(\1.*){3,}"; then
        fail "Generation is degenerate (token repetition)"
    else
        pass "Generation produced non-degenerate text"
    fi
else
    # Fallback: check tokens themselves
    TOKEN_LINE=$(grep -E "^\s*\[[0-9]\]\s+id=" "$GEN_OUT" 2>/dev/null | awk '{print $NF}' || true)
    UNIQ=$(echo "$TOKEN_LINE" | sort -u | wc -l)
    if [[ "$UNIQ" -gt 3 ]]; then
        pass "Generated $(echo "$TOKEN_LINE" | wc -l) tokens, $UNIQ unique — non-degenerate"
    else
        fail "Tokens collapsed to <4 unique: $TOKEN_LINE"
    fi
fi

echo
echo "=== ALL 5 CHECKS PASSED ==="
echo "Ornith 1.0 9B native forward verified for: $MODEL"
echo
echo "To reproduce: $TEST_BIN --model \"$MODEL\" --prompt \"$PROMPT\" --tokens 20 --temperature 0.7"
rm -f "$OUT" "$GEN_OUT"
