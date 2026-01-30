#!/bin/bash
# Build and test with sanitizers
# Usage: ./scripts/ci/sanitizers.sh [sanitizer]
# Sanitizers: address, undefined, address,undefined (default), thread, memory

set -euo pipefail

SANITIZER="${1:-address,undefined}"
BUILDDIR="builddir-san-${SANITIZER//,/-}"

echo "==> Setting up sanitizer build: $SANITIZER"
echo "    Build directory: $BUILDDIR"

meson setup "$BUILDDIR" -Db_sanitize="$SANITIZER" --wipe 2>/dev/null || \
    meson setup "$BUILDDIR" -Db_sanitize="$SANITIZER"

echo "==> Building"
meson compile -C "$BUILDDIR"

echo "==> Running tests with $SANITIZER sanitizer"
TERM=xterm-256color meson test -C "$BUILDDIR" --verbose

echo "==> Sanitizer tests passed: $SANITIZER"
