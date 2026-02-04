#!/bin/bash
# Build and run tests
# Usage: ./scripts/ci/build.sh [builddir]

set -euo pipefail

BUILDDIR="${1:-builddir}"

echo "==> Setting up build directory: $BUILDDIR"
meson setup "$BUILDDIR" --wipe 2>/dev/null || meson setup "$BUILDDIR"

echo "==> Building"
meson compile -C "$BUILDDIR"

echo "==> Running tests"
meson test -C "$BUILDDIR"

echo "==> Build and tests passed"
