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
ninja -C "$BUILDDIR" coverage

echo "==> Coverage report generated"
echo "    Text report: $BUILDDIR/meson-logs/coverage.txt"

# Generate HTML if available
if command -v gcovr &>/dev/null || command -v lcov &>/dev/null; then
    ninja -C "$BUILDDIR" coverage-html 2>/dev/null || true
    echo "    HTML report: $BUILDDIR/meson-logs/coveragereport/"
fi
