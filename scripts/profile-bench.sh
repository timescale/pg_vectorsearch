#!/usr/bin/env bash
#
# profile-bench.sh - Quick profiling for distance benchmarks
#
# Usage:
#   ./scripts/profile-bench.sh [dim] [count] [impl]
#
# Examples:
#   ./scripts/profile-bench.sh 768 10000 avx512
#   ./scripts/profile-bench.sh 384 50000
#   ./scripts/profile-bench.sh 1536 5000 scalar

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
DIM="${1:-768}"
COUNT="${2:-10000}"
IMPL="${3:-}"

# Build command
CMD="./bin/vectorsearch bench distance --dim $DIM --count $COUNT --metric l2"
if [[ -n "$IMPL" ]]; then
    CMD="$CMD --impls $IMPL"
fi

# Generate output name
OUTPUT="profile-distance-d${DIM}-n${COUNT}"
if [[ -n "$IMPL" ]]; then
    OUTPUT="${OUTPUT}-${IMPL}"
fi

echo "Profiling: $CMD"
echo "Output: ${OUTPUT}.svg"
echo

# Run profiler
# shellcheck disable=SC2086  # CMD intentionally unquoted for word splitting
"$SCRIPT_DIR/profile.sh" --output "$OUTPUT" $CMD
