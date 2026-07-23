#!/bin/bash
# Profile a 70B run: capture top %CPU + VmRSS + tok/s in a single pass.
# This generates ground-truth data for "where is the CPU going?"
#
# Usage: ./profile_70b_cpu.sh <threads>  (e.g., ./profile_70b_cpu.sh 7)
#   threads defaults to 7 (auto-cap by configure_thread_pool)

set -u
THREADS="${1:-7}"
LABEL="70b_t${THREADS}"
OUT="/tmp/leafcutter_${LABEL}_output.txt"
CPU="/tmp/leafcutter_${LABEL}_cpu.txt"
TOP="/tmp/leafcutter_${LABEL}_top.txt"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUST_DIR="${REPO_ROOT}/rust"
BIN="${RUST_DIR}/target/release/test_generation"

cd "$RUST_DIR"

echo "Begin ${LABEL}  threads=${THREADS}"

export LEAFCUTTER_THREADS="${THREADS}"

# Start under top to capture per-sample CPU%.
# top -b runs in batch mode; -d sets sample interval in seconds.
# We sample every 0.2s, but only show leafcutter process lines.
"${BIN}" \
    --model "/home/xander/Downloads/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf" \
    --prompt "The capital of France is" \
    --tokens 2 \
    --temperature 0.7 \
    > "$OUT" 2>&1 &

BGPID=$!
sleep 0.5

echo "=== top %CPU + VmRSS for ${LABEL} (pid found from process tree) ===" > "$TOP"
echo "poll-start: $(date +%H:%M:%S.%N)" >> "$TOP"

PEAK_TOP=0
SUM_TOP=0
NSAMP=0
while [ -d "/proc/$BGPID" ]; do
    # The actual test_generation child
    REALPID=$(pgrep -P "$BGPID" -f test_generation | head -1)
    if [ -z "$REALPID" ]; then
        REALPID=$BGPID
    fi
    # Per-process: top -b -n 1 -p <pid>
    SNAP="$(top -b -n 1 -p "$REALPID" 2>/dev/null | tail -n +8 | head -1)"
    PCPU=$(echo "$SNAP" | awk '{print $9}')
    RSS=$(awk '/VmRSS:/{print $2}' "/proc/$REALPID/status" 2>/dev/null)
    echo "$(date +%H:%M:%S.%N)  $SNAP  rss=$RSS" >> "$TOP"
    if [ -n "$PCPU" ]; then
        SUM_TOP=$(echo "$SUM_TOP + $PCPU" | bc -l 2>/dev/null || echo "$SUM_TOP")
        NSAMP=$((NSAMP+1))
        # quick peak by integer parse
        PCPU_INT=$(echo "$PCPU" | cut -d. -f1)
        if [ "$PCPU_INT" -gt "$PEAK_TOP" ]; then PEAK_TOP=$PCPU_INT; fi
    fi
    sleep 0.2
done

echo "poll-end: $(date +%H:%M:%S.%N)" >> "$TOP"
echo "summary-samples: $NSAMP  peak-pcpu-int: $PEAK_TOP" >> "$TOP"

echo "Wrote: $TOP"
echo
echo "=== Generation output (last 30 lines) ==="
tail -n 30 "$OUT" 2>&1
