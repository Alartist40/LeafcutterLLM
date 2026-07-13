#!/bin/bash
# Profile kernel call distribution + cost during a forward pass.
# Uses the LEAFCUTTER_PROFILE env var hook in Tensor::matmul.
#
# Usage: ./profile_kernels.sh <label> [extra args to test_generation]
set -u
LABEL="${1:-profile}"
shift || true

OUT="/tmp/leafcutter_${LABEL}_profile.txt"
SLICE="/tmp/leafcutter_${LABEL}_kernels.txt"

cd "$(cd "$(dirname "$0")" && pwd)/../rust"

LIB_PATH="$(dirname $(find . -name 'libllama.so' 2>/dev/null | head -1))"
[ -n "$LIB_PATH" ] && export LD_LIBRARY_PATH="${LIB_PATH}:${LD_LIBRARY_PATH:-}"

export LEAFCUTTER_PROFILE=1
echo "Begin profiling [${LABEL}]"
./target/release/test_generation \
    --model "/home/xander/Downloads/models/ornith-1.0-9b-Q4_K_M.gguf" \
    --prompt "The capital of France is" \
    --tokens 5 \
    --temperature 0.7 \
    > "$OUT" 2>&1

# Extract just the kernel log lines
grep -E "\[PROFILE\] matmul" "$OUT" > "$SLICE"

# Aggregate by quant type
echo "=== Kernel aggregate ($LABEL) ==="
awk '{
    type=$3
    # fields after m, k, n:
    sub(/.*matmul */,"")
    # parse: TYPE  m=N k=N n=N  TTTT.Tms
    # the format has fixed-width columns. Use split-by-fields:
    cnt[type]++
    sum[type]+=$5
}' "$SLICE"
# the awk above is shell-quirky; redo with python-free approach using grep+sort
declare -A SUM COUNT
while IFS= read -r line; do
    T=$(echo "$line" | awk '{print $4}')
    V=$(echo "$line" | awk '{print $NF}' | tr -d 'ms')
    [[ -z "$T" || -z "$V" ]] && continue
    # V is like "  123.45ms" — strip
    V=${V%ms}
    V=$(echo "$V" | tr -d ' ')
    [[ -z "$V" ]] && continue
    COUNT[$T]=$(( ${COUNT[$T]:-0} + 1 ))
    SUM[$T]=$(awk -v a="${SUM[$T]:-0}" -v b="$V" 'BEGIN{printf "%.4f", a + b}')
done < <(grep -E "\[PROFILE\] matmul " "$OUT")

# Print summary
echo
echo "Kernel       |   calls     | total_ms %share"
echo "-------------+-------------+----------------"
TOTAL=0
for T in "${!SUM[@]}"; do
    TOTAL=$(awk -v a="$TOTAL" -v b="${SUM[$T]}" 'BEGIN{printf "%.4f", a + b}')
done
for T in "${!SUM[@]}"; do
    PCT=$(awk -v a="${SUM[$T]}" -v b="$TOTAL" 'BEGIN{ if(b>0) printf "%.1f", 100*a/b; else print "0.0" }')
    printf "%-12s | %6d      | %8.2f %s%%\n" "$T" "${COUNT[$T]}" "${SUM[$T]}" "$PCT"
done | sort -k4 -rn
echo
echo "GRAND TOTAL kernel time: ${TOTAL} ms"

# Show n>... weight counts
echo
echo "Top 20 longest individual calls:"
sort -k7 -rn "$SLICE" | head -20
