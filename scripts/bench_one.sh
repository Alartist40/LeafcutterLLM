#!/bin/bash
# Run ./target/release/test_generation with default thread settings and capture
# both stdout and a CPU%/RSS profile.
#
# Usage: ./bench_one.sh <label>
#
# Output:
#   /tmp/leafcutter_<label>_output.txt   binary's stdout/stderr
#   /tmp/leafcutter_<label>_cpu.txt      per-sample CPU/RSS profile
#
set -u
LABEL="${1:-run}"
OUT="/tmp/leafcutter_${LABEL}_output.txt"
CPU="/tmp/leafcutter_${LABEL}_cpu.txt"

# Anchor: this script lives at <repo>/scripts/bench_one.sh. Defend against
# being run from anywhere by deriving the repo root from our own location.
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
RUST_DIR="${REPO_ROOT}/rust"
cd "$RUST_DIR"
echo "Begin bench [${LABEL}]  rustdir=$RUST_DIR"
LIB_PATH="$(dirname $(find . -name 'libllama.so' 2>/dev/null | head -1))"
[ -n "$LIB_PATH" ] && export LD_LIBRARY_PATH="${LIB_PATH}:${LD_LIBRARY_PATH:-}"

./target/release/test_generation \
    --model "/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf" \
    --prompt "The capital of France is" \
    --tokens 5 \
    --temperature 0.7 \
    > "$OUT" 2>&1 &
BGPID=$!
[ -d "/proc/$BGPID" ] || sleep 0.2
[ -d "/proc/$BGPID" ] || { echo "FATAL: bg pid ${BGPID} did not start" >&2; exit 1; }

echo "=== CPU profile for $LABEL (pid=$BGPID) ===" > "$CPU"
PEAK=0
SUM_F=0
N=0
PREV_CPU=$(awk '{u=$14+$15+$16+$17; print u}' /proc/$BGPID/stat 2>/dev/null)
PREV_TS=$(($(date +%s%N)))
CLK=$(getconf CLK_TCK)

while [ -d "/proc/$BGPID" ]; do
    sleep 0.1
    [ -d "/proc/$BGPID" ] || break
    CUR_CPU=$(awk '{u=$14+$15+$16+$17; print u}' /proc/$BGPID/stat 2>/dev/null)
    CUR_TS=$(($(date +%s%N)))
    [ -z "$CUR_CPU" ] || [ -z "$PREV_CPU" ] && continue

    DCPU=$(( CUR_CPU - PREV_CPU ))
    DT_MS=$(( (CUR_TS - PREV_TS) / 1000000 ))
    [ "$DT_MS" -le 0 ] && continue
    PCT_10=$(( DCPU * 10000000 / (DT_MS * CLK * 10) ))  # pct * 10
    if [ "$PCT_10" -gt 0 ]; then
        RSS=$(awk '/VmRSS/{print $2}' /proc/$BGPID/status 2>/dev/null)
        TH=$(ls /proc/$BGPID/task 2>/dev/null | wc -l)
        INT=$((PCT_10 / 10))
        DEC=$((PCT_10 % 10))
        echo "$(date +%H:%M:%S)  ${INT}.${DEC}%  VmRSS=${RSS}KB  threads=$TH" >> "$CPU"
        SUM_F=$(( SUM_F + PCT_10 ))
        N=$((N+1))
        [ "$PCT_10" -gt "$PEAK" ] && PEAK=$PCT_10
    fi
    PREV_CPU=$CUR_CPU
    PREV_TS=$CUR_TS
done

wait $BGPID 2>/dev/null || true

if [ "$N" -gt 0 ]; then
    AVG_INT=$(( SUM_F / (N * 10) ))
    AVG_DEC=$(( (SUM_F / N) % 10 ))
else
    AVG_INT=0; AVG_DEC=0
fi
PEAK_INT=$(( PEAK / 10 ))
PEAK_DEC=$(( PEAK % 10 ))
ELAPSED=$(grep -oE 'Generated[^a-zA-Z]*[0-9.]+s' "$OUT" | head -1 | grep -oE '[0-9.]+s' | head -1)

{
    echo "─── SUMMARY ($LABEL) ───"
    echo "    elapsed=${ELAPSED:-?}  samples=$N  avg=${AVG_INT}.${AVG_DEC}% peak=${PEAK_INT}.${PEAK_DEC}%"
    echo "─── OUTPUT (last 8 lines) ───"
    tail -8 "$OUT"
} >> "$CPU"

echo "[$LABEL] elapsed=${ELAPSED:-?} avg=${AVG_INT}.${AVG_DEC}% peak=${PEAK_INT}.${PEAK_DEC}%"
