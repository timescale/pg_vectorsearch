#!/usr/bin/env bash
#
# profile.sh - Profile meerkat CLI with perf and generate flame graphs
#
# Usage:
#   ./scripts/profile.sh [command] [args...]
#
# Examples:
#   ./scripts/profile.sh ./bin/mkt bench distance --dim 768 --count 10000
#   ./scripts/profile.sh ./bin/mkt bench distance --dim 384 --impls avx512
#
# Requirements:
#   - perf (linux-tools-common, linux-tools-generic)
#   - FlameGraph (cloned to ../FlameGraph or /tmp/FlameGraph)
#
# Output:
#   - perf.data (raw perf data)
#   - perf.folded (collapsed stacks)
#   - flamegraph.svg (interactive flame graph)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[profile]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[profile]${NC} $*"
}

error() {
    echo -e "${RED}[profile]${NC} $*" >&2
}

die() {
    error "$@"
    exit 1
}

# Check for perf
if ! command -v perf &> /dev/null; then
    die "perf not found. Install with: sudo apt-get install linux-tools-common linux-tools-generic"
fi

# Check for FlameGraph
FLAMEGRAPH_DIR=""
TEMP_BASE="${TMPDIR:-/tmp/claude}"

if [[ -d "$PROJECT_ROOT/../FlameGraph" ]]; then
    FLAMEGRAPH_DIR="$PROJECT_ROOT/../FlameGraph"
elif [[ -d "$TEMP_BASE/FlameGraph" ]]; then
    FLAMEGRAPH_DIR="$TEMP_BASE/FlameGraph"
else
    warn "FlameGraph not found. Cloning to $TEMP_BASE/FlameGraph..."
    mkdir -p "$TEMP_BASE"
    git clone https://github.com/brendangregg/FlameGraph.git "$TEMP_BASE/FlameGraph"
    FLAMEGRAPH_DIR="$TEMP_BASE/FlameGraph"
fi

info "Using FlameGraph from: $FLAMEGRAPH_DIR"

# Parse arguments
if [[ $# -eq 0 ]]; then
    cat <<EOF
Usage: $0 <command> [args...]

Profile a command with perf and generate a flame graph.

Examples:
  $0 ./bin/mkt bench distance --dim 768 --count 10000
  $0 ./bin/mkt bench distance --dim 384 --impls avx512

Options:
  --events <events>  Perf events to record (default: cycles)
  --freq <hz>        Sampling frequency (default: 999)
  --output <name>    Output file basename (default: flamegraph)
  --keep-perf-data   Don't delete perf.data after processing

Output:
  <name>.svg         Flame graph (open in browser)
  <name>.folded      Collapsed stacks (for manual analysis)
  perf.data          Raw perf data (deleted unless --keep-perf-data)
EOF
    exit 1
fi

# Default options
EVENTS="cycles"
FREQ="999"
OUTPUT="flamegraph"
KEEP_PERF_DATA=false

# Parse profile.sh options
COMMAND_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --events)
            EVENTS="$2"
            shift 2
            ;;
        --freq)
            FREQ="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --keep-perf-data)
            KEEP_PERF_DATA=true
            shift
            ;;
        *)
            COMMAND_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#COMMAND_ARGS[@]} -eq 0 ]]; then
    die "No command specified"
fi

COMMAND="${COMMAND_ARGS[*]}"

info "Command: $COMMAND"
info "Events: $EVENTS"
info "Frequency: $FREQ Hz"
info "Output: $OUTPUT.svg"

# Use existing build (dwarf call-graph works without frame pointers)
BUILDDIR="$PROJECT_ROOT/builddir"
if [[ ! -f "$BUILDDIR/mkt" ]]; then
    die "Build not found. Run: meson compile -C builddir"
fi

# Create profiles output directory
PROFILE_DIR="$PROJECT_ROOT/profiles"
mkdir -p "$PROFILE_DIR"
info "Output directory: $PROFILE_DIR"

# Check perf_event_paranoid
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
if [[ $PARANOID -gt 1 ]]; then
    error "perf_event_paranoid is set to $PARANOID (too restrictive for profiling)"
    echo
    echo "To enable profiling, run ONE of the following:"
    echo
    echo "  1. Lower paranoid setting (recommended):"
    echo "     sudo sysctl -w kernel.perf_event_paranoid=1"
    echo
    echo "  2. Make it permanent (survives reboot):"
    echo "     echo 'kernel.perf_event_paranoid = 1' | sudo tee -a /etc/sysctl.conf"
    echo
    echo "  3. Run this script with sudo (not recommended):"
    echo "     sudo $0 $*"
    echo
    echo "Current value: $PARANOID (need ≤ 1 for call-graph profiling)"
    echo "Explanation: https://www.kernel.org/doc/html/latest/admin-guide/perf-security.html"
    exit 1
fi

# Record with perf
info "Recording with perf..."
PERF_DATA="$PROFILE_DIR/perf.data"

# Remove old perf.data if it exists
[[ -f "$PERF_DATA" ]] && rm "$PERF_DATA"

cd "$PROJECT_ROOT"

# Run perf record
# -F: frequency (samples per second)
# -g: record call graph (stack traces)
# --call-graph dwarf: use DWARF for accurate stacks (works without frame pointers)
# -o: output file
if perf record -F "$FREQ" -g --call-graph dwarf -o "$PERF_DATA" -- "${COMMAND_ARGS[@]}"; then
    info "Recording complete"
else
    die "perf record failed"
fi

# Check perf.data size
PERF_SIZE=$(du -h "$PERF_DATA" | cut -f1)
info "perf.data size: $PERF_SIZE"

# Generate flame graph
info "Generating flame graph..."

# Convert perf.data to folded stacks
FOLDED="$PROFILE_DIR/${OUTPUT}.folded"
perf script -i "$PERF_DATA" | \
    "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" > "$FOLDED"

# Generate SVG
SVG="$PROFILE_DIR/${OUTPUT}.svg"
"$FLAMEGRAPH_DIR/flamegraph.pl" "$FOLDED" > "$SVG"

info "Flame graph generated: $SVG"

# Generate reverse flame graph (icicle graph)
ICICLE="$PROFILE_DIR/${OUTPUT}-icicle.svg"
"$FLAMEGRAPH_DIR/flamegraph.pl" --reverse --inverted "$FOLDED" > "$ICICLE"
info "Icicle graph generated: $ICICLE"

# Clean up perf.data unless requested to keep
if [[ "$KEEP_PERF_DATA" == "false" ]]; then
    rm "$PERF_DATA"
    info "Cleaned up perf.data"
else
    info "Kept perf.data for manual analysis"
fi

# Print summary
echo
info "Profiling complete!"
echo
echo "  Flame graph:  $SVG"
echo "  Icicle graph: $ICICLE"
echo "  Folded:       $FOLDED"
echo
info "Open in browser: firefox $SVG"
echo
info "Top functions:"
perf script -i "$PERF_DATA" 2>/dev/null | \
    "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" | \
    awk -F';' '{gsub(/^[^;]+;/, ""); print}' | \
    awk '{a[$1]+=$2} END {for(i in a) print a[i],i}' | \
    sort -rn | head -10 || true
