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
MIN_COVERAGE=90

# Use gcovr for coverage (works in sandboxed environments unlike lcov which
# has /tmp hardcoded; use lcov --tempdir if lcov is preferred)
gcovr --root . --object-directory "$BUILDDIR" \
    --exclude 'builddir.*' --exclude 'test/.*' \
    --txt "$LOGDIR/coverage.txt" \
    --xml "$LOGDIR/coverage.xml" \
    --html-details "$LOGDIR/coveragereport/index.html" \
    --fail-under-line "$MIN_COVERAGE" \
    || { echo ""; \
         echo "==> FAILED: Line coverage is below ${MIN_COVERAGE}%"; \
         echo "    See coverage report: $LOGDIR/coveragereport/index.html"; \
         exit 1; }

echo "==> Coverage check passed (minimum ${MIN_COVERAGE}% line coverage)"
echo "    Text report: $LOGDIR/coverage.txt"
echo "    XML report:  $LOGDIR/coverage.xml"
echo "    HTML report: $LOGDIR/coveragereport/index.html"
