#!/bin/bash
# Build with coverage and generate report
# Usage: ./scripts/ci/coverage.sh [builddir]
#
# Environment variables:
#   PG_CONFIG  - path to pg_config (default: auto-detect)
#   CC         - C compiler (default: auto-detect from pg_config)

set -euo pipefail

# Allow a crashing test process to leave a core dump (CI analyzes them via the
# coredumps action). Harmless locally; cores go to the cwd.
ulimit -c unlimited 2>/dev/null || true

BUILDDIR="${1:-builddir-cov}"

# Detect pg_config and compiler
PG_CONFIG="${PG_CONFIG:-$(command -v pg_config 2>/dev/null || true)}"
MESON_PG_ARGS=()

if [[ -n "$PG_CONFIG" ]]; then
    PG_CC=$("$PG_CONFIG" --cc 2>/dev/null || true)
    if [[ -n "$PG_CC" ]]; then
        CC="${CC:-$PG_CC}"
        export CC
        echo "==> PostgreSQL compiler: $CC (from $PG_CONFIG)"
        MESON_PG_ARGS+=("-Dpostgresql=enabled" "-Dpg_config=$PG_CONFIG")
    else
        echo "==> Warning: pg_config found but --cc failed, building without PG"
        MESON_PG_ARGS+=("-Dpostgresql=disabled")
    fi
else
    echo "==> No pg_config found, building without PostgreSQL extension"
    MESON_PG_ARGS+=("-Dpostgresql=disabled")
fi

echo "==> Setting up coverage build: $BUILDDIR"
meson setup "$BUILDDIR" -Db_coverage=true "${MESON_PG_ARGS[@]}" --wipe 2>/dev/null || \
    meson setup "$BUILDDIR" -Db_coverage=true "${MESON_PG_ARGS[@]}"

echo "==> Building"
meson compile -C "$BUILDDIR"

# Install extension so regression tests can find it via --temp-instance
if [[ ${#MESON_PG_ARGS[@]} -gt 0 && "${MESON_PG_ARGS[0]}" != "-Dpostgresql=disabled" ]]; then
    echo "==> Installing extension (required for regression tests)"
    if ! meson install -C "$BUILDDIR" 2>/dev/null; then
        echo "    Retrying with sudo (system PostgreSQL)"
        sudo meson install -C "$BUILDDIR"
    fi
fi

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
#
# The --filter whitelist restricts the report to the project's own
# sources. Without it, anything under --root gets counted — in CI the
# debug PostgreSQL install lives inside the repo, so PG's own header
# inlines (near-100% covered) entered the report and inflated the
# aggregate the MIN_COVERAGE gate checks, masking under-tested project
# code.
GCOVR_EXCLUDES=(
    --filter 'src/.*'
    --exclude 'src/cli/.*'
    --exclude '.*_pg\.h'
)

# Exclude foreign architecture SIMD implementations and unsupported variants
ARCH=$(uname -m)
OS=$(uname -s)

case "$ARCH" in
    x86_64|amd64|i686)
        # On x86: exclude ARM NEON (cannot execute)
        GCOVR_EXCLUDES+=(--exclude '.*_neon\.c')

        # Check if CPU supports AVX2
        HAS_AVX2=0
        if [[ "$OS" == "Linux" ]]; then
            if grep -q ' avx2 ' /proc/cpuinfo 2>/dev/null; then
                HAS_AVX2=1
            fi
        elif [[ "$OS" == "Darwin" ]]; then
            if sysctl -n machdep.cpu.features machdep.cpu.leaf7_features 2>/dev/null | grep -q AVX2; then
                HAS_AVX2=1
            fi
        fi

        if [[ $HAS_AVX2 -eq 0 ]]; then
            GCOVR_EXCLUDES+=(--exclude '.*_avx2\.c')
        fi

        # Check if CPU supports AVX-512
        HAS_AVX512=0
        if [[ "$OS" == "Linux" ]]; then
            if grep -q ' avx512f ' /proc/cpuinfo 2>/dev/null; then
                HAS_AVX512=1
            fi
        elif [[ "$OS" == "Darwin" ]]; then
            if sysctl -n machdep.cpu.features 2>/dev/null | grep -q AVX512F; then
                HAS_AVX512=1
            fi
        fi

        if [[ $HAS_AVX512 -eq 0 ]]; then
            GCOVR_EXCLUDES+=(--exclude '.*_avx512\.c')
        fi
        ;;
    aarch64|arm64|arm*)
        # On ARM: exclude x86 AVX implementations (cannot execute)
        GCOVR_EXCLUDES+=(--exclude '.*_avx2\.c')
        GCOVR_EXCLUDES+=(--exclude '.*_avx512\.c')
        ;;
esac

# Select the right gcov tool to match the compiler
GCOV_TOOL=()
CC_BASE=$(basename "${CC:-cc}")
if [[ "$CC_BASE" == clang* ]]; then
    GCOV_TOOL=(--gcov-executable "llvm-cov gcov")
elif [[ "$CC_BASE" == gcc-* ]] && command -v "gcov-${CC_BASE#gcc-}" >/dev/null; then
    # gcov must match the gcc that produced the .gcda files: a
    # major-version mismatch makes gcov emit no records and the report
    # silently comes out empty.
    GCOV_TOOL=(--gcov-executable "gcov-${CC_BASE#gcc-}")
fi

# Generate coverage reports
gcovr --root . --object-directory "$BUILDDIR" \
    "${GCOV_TOOL[@]}" \
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
