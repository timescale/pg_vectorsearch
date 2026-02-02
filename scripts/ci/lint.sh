#!/bin/bash
# Run all lint checks via pre-commit
# Usage: ./scripts/ci/lint.sh [builddir]

set -euo pipefail

BUILDDIR="${1:-builddir}"

# Ensure build directory exists (needed for clang-tidy's compile_commands.json)
if [ ! -d "$BUILDDIR" ]; then
    echo "==> Setting up build directory: $BUILDDIR"
    meson setup "$BUILDDIR"
fi

echo "==> Running pre-commit hooks"
pre-commit run --all-files
