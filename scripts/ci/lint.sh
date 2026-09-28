#!/bin/bash
# Run all lint checks via pre-commit
# Usage: ./scripts/ci/lint.sh [builddir]

set -euo pipefail

BUILDDIR="${1:-builddir}"

# Ensure build directory exists (needed for clang-tidy's compile_commands.json)
if [ ! -d "$BUILDDIR" ]; then
    echo "==> Setting up build directory: $BUILDDIR"
    # postgresql=enabled, not auto: if the extension cannot be configured
    # then src/pg never enters the compile database and clang-tidy silently
    # covers less than it appears to. Failing here says why.
    meson setup "$BUILDDIR" -Dpostgresql=enabled
fi

# src/pg/support_pg.c includes git_commit.h, which vcs_tag generates at build
# time; without it clang-tidy cannot parse that translation unit. Building the
# one target costs nothing -- the rest of the tree is never compiled here.
meson compile -C "$BUILDDIR" git_commit.h >/dev/null

echo "==> Running pre-commit hooks"
pre-commit run --all-files
