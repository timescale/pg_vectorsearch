#!/bin/bash
# Run the lint checks via pre-commit
# Usage: ./scripts/ci/lint.sh [builddir]
#
# Environment:
#   BASE_SHA, HEAD_SHA  When both are set, lint only the files that changed
#                       between them (a pull request's range). Left unset,
#                       lint every tracked file.
#
# A change to the lint configuration itself, or to this script, lints every
# file regardless of the range, since the change can affect files it does
# not touch.

set -euo pipefail

BUILDDIR="${1:-builddir}"
BASE_SHA="${BASE_SHA:-}"
HEAD_SHA="${HEAD_SHA:-}"

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

lint_config='^(\.pre-commit-config\.yaml|\.clang-format[^/]*|\.clang-tidy|\.markdownlint[^/]*|\.squawk[^/]*|scripts/ci/(lint|check-[a-z-]+)\.sh)$'

if [[ -n "$BASE_SHA" && -n "$HEAD_SHA" ]]; then
    if git diff --name-only "$BASE_SHA...$HEAD_SHA" | grep -Eq "$lint_config"; then
        echo "==> Lint configuration changed; running pre-commit on all files"
    else
        echo "==> Running pre-commit on files changed in $BASE_SHA...$HEAD_SHA"
        exec pre-commit run --from-ref "$BASE_SHA" --to-ref "$HEAD_SHA" \
            --show-diff-on-failure
    fi
else
    echo "==> Running pre-commit on all files"
fi
pre-commit run --all-files --show-diff-on-failure
