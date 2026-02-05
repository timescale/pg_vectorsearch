#!/bin/bash
# Build with coverage and generate report
# Usage: ./scripts/ci/coverage.sh [builddir]

set -euo pipefail

BUILDDIR="${1:-builddir-cov}"

echo "==> Setting up coverage build: $BUILDDIR"
meson setup "$BUILDDIR" -Db_coverage=true --wipe 2>/dev/null || \
    meson setup "$BUILDDIR" -Db_coverage=true

echo "==> Building"
meson compile -C "$BUILDDIR"

echo "==> Running tests"
TERM=xterm-256color meson test -C "$BUILDDIR" --verbose

echo "==> Generating coverage report"
LOGDIR="$BUILDDIR/meson-logs"
mkdir -p "$LOGDIR/coveragereport"

# Minimum line coverage threshold (percentage)
#
# Set to 85% for cross-platform SIMD code. We use mkt_simd_set_override()
# to test all same-architecture SIMD implementations (scalar, AVX2, AVX-512
# on x86-64; scalar, NEON on ARM), achieving 88%+ coverage of testable code.
# The remaining code is mostly unsupported SIMD variants for the CPU.
MIN_COVERAGE=85

# Use gcovr for coverage (works in sandboxed environments unlike lcov which
# has /tmp hardcoded; use lcov --tempdir if lcov is preferred)
#
# Exclusions:
# - builddir.* : build artifacts
# - test/.* : test code
# - src/cli/.* : CLI tools (tested via integration, not unit tests)
# - .*_pg\.h : PostgreSQL-specific headers (not compiled in standalone mode)
# - Platform-specific SIMD files (can't be tested on other architectures)

# Build base exclusion patterns
#
# Note: We use mkt_simd_set_override() to test all same-architecture SIMD
# variants (scalar, AVX2, AVX-512 on x86; scalar, NEON on ARM), achieving
# full coverage of testable code paths. We only exclude code for foreign
# architectures that cannot execute at all.
GCOVR_EXCLUDES=(
    --exclude 'builddir.*'
    --exclude 'test/.*'
    --exclude 'src/cli/.*'
    --exclude '.*_pg\.h'
)

# Exclude foreign architecture SIMD implementations and unsupported variants
ARCH=$(uname -m)
OS=$(uname -s)

case "$ARCH" in
    x86_64|amd64|i686)
        # On x86-64: exclude ARM NEON (cannot execute)
        GCOVR_EXCLUDES+=(--exclude '.*distance_neon\.c')

        # Check if CPU supports AVX2
        HAS_AVX2=0
        if [[ "$OS" == "Linux" ]]; then
            # Linux: check /proc/cpuinfo
            if grep -q ' avx2 ' /proc/cpuinfo 2>/dev/null; then
                HAS_AVX2=1
            fi
        elif [[ "$OS" == "Darwin" ]]; then
            # macOS: check sysctl
            if sysctl -n machdep.cpu.features machdep.cpu.leaf7_features 2>/dev/null | grep -q AVX2; then
                HAS_AVX2=1
            fi
        fi

        if [[ $HAS_AVX2 -eq 0 ]]; then
            # No AVX2 support: exclude AVX2 code
            GCOVR_EXCLUDES+=(--exclude '.*distance_avx2\.c')
        fi

        # Check if CPU supports AVX-512 (needs avx512f flag)
        HAS_AVX512=0
        if [[ "$OS" == "Linux" ]]; then
            # Linux: check /proc/cpuinfo
            if grep -q ' avx512f ' /proc/cpuinfo 2>/dev/null; then
                HAS_AVX512=1
            fi
        elif [[ "$OS" == "Darwin" ]]; then
            # macOS: check sysctl (note: most Macs don't have AVX-512)
            if sysctl -n machdep.cpu.features 2>/dev/null | grep -q AVX512F; then
                HAS_AVX512=1
            fi
        fi

        if [[ $HAS_AVX512 -eq 0 ]]; then
            # No AVX-512 support: exclude AVX-512 code
            GCOVR_EXCLUDES+=(--exclude '.*distance_avx512\.c')
        fi
        ;;
    aarch64|arm64|arm*)
        # On ARM: exclude x86 AVX implementations (cannot execute)
        GCOVR_EXCLUDES+=(--exclude '.*distance_avx.*\.c')
        ;;
esac

# Generate coverage reports
gcovr --root . --object-directory "$BUILDDIR" \
    "${GCOVR_EXCLUDES[@]}" \
    --txt "$LOGDIR/coverage.txt" \
    --xml "$LOGDIR/coverage.xml" \
    --cobertura "$LOGDIR/coverage-cobertura.xml" \
    --json "$LOGDIR/coverage.json" \
    --html-details "$LOGDIR/coveragereport/index.html"

# Display coverage summary in terminal
echo ""
echo "==> Coverage Summary"
echo ""
cat "$LOGDIR/coverage.txt"
echo ""

# Generate markdown summary for GitHub Actions
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "## Coverage Report"
        echo ""
        echo "\`\`\`"
        cat "$LOGDIR/coverage.txt"
        echo "\`\`\`"
        echo ""
        echo "📊 Full HTML report available in artifacts"
    } >> "$GITHUB_STEP_SUMMARY"
fi

# Extract total coverage percentage for threshold check
TOTAL_COVERAGE=$(grep -E '^TOTAL' "$LOGDIR/coverage.txt" | awk '{print $4}' | sed 's/%//')

# Check coverage threshold using awk for floating point comparison
if awk -v cov="$TOTAL_COVERAGE" -v min="$MIN_COVERAGE" 'BEGIN {exit !(cov < min)}'; then
    echo "==> FAILED: Line coverage is ${TOTAL_COVERAGE}% (minimum ${MIN_COVERAGE}%)"
    echo "    See coverage report: $LOGDIR/coveragereport/index.html"
    exit 1
fi

echo "==> Coverage check passed: ${TOTAL_COVERAGE}% (minimum ${MIN_COVERAGE}%)"
echo "    Text report: $LOGDIR/coverage.txt"
echo "    XML report:  $LOGDIR/coverage.xml"
echo "    JSON report: $LOGDIR/coverage.json"
echo "    HTML report: $LOGDIR/coveragereport/index.html"
