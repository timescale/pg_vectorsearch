#!/bin/bash
# Run linting: format check and clang-tidy
# Usage: ./scripts/ci/lint.sh [builddir]

set -euo pipefail

BUILDDIR="${1:-builddir}"

# Ensure build directory exists (needed for compile_commands.json)
if [ ! -d "$BUILDDIR" ]; then
    echo "==> Setting up build directory: $BUILDDIR"
    meson setup "$BUILDDIR"
fi

echo "==> Checking code formatting"
CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"

# Find all source files (use arrays to handle filenames properly)
mapfile -t SRC_FILES < <(find src -name '*.c' -o -name '*.h' 2>/dev/null)
mapfile -t TEST_FILES < <(find test -name '*.c' -o -name '*.h' 2>/dev/null)
SOURCES=("${SRC_FILES[@]}" "${TEST_FILES[@]}")

if [ ${#SOURCES[@]} -gt 0 ]; then
    FORMAT_DIFF=$("$CLANG_FORMAT" --dry-run -Werror "${SOURCES[@]}" 2>&1) || {
        echo "Format check failed. Run: meson compile -C $BUILDDIR format"
        echo "$FORMAT_DIFF"
        exit 1
    }
    echo "    Format check passed"
else
    echo "    No source files found"
fi

echo "==> Running clang-tidy"
if command -v clang-tidy &>/dev/null; then
    mapfile -t TIDY_SOURCES < <(find src -name '*.c' 2>/dev/null)
    if [ ${#TIDY_SOURCES[@]} -gt 0 ]; then
        clang-tidy -p "$BUILDDIR" "${TIDY_SOURCES[@]}"
        echo "    clang-tidy passed"
    else
        echo "    No source files to analyze"
    fi
else
    echo "    clang-tidy not found, skipping"
fi

echo "==> Lint checks passed"
