#!/bin/bash
# Verify every GitHub Actions status badge in README.md points at a workflow
# file that actually exists.
#
# Usage: ./scripts/ci/check-readme-badges.sh
#
# A badge silently renders "not found" once its workflow is renamed or
# deleted -- nothing else catches that, since it's a live external image URL,
# not something meson or pg_regress touches. Guards against that going unfixed
# by pinning the badge to the workflow file at review time.

set -euo pipefail

readme="README.md"
missing=()

while IFS= read -r workflow; do
    if [[ ! -f ".github/workflows/$workflow" ]]; then
        missing+=("$workflow (referenced by README.md badge)")
    fi
done < <(grep -oE 'actions/workflows/[A-Za-z0-9_.-]+\.yml/badge\.svg' "$readme" |
    sed -E 's#actions/workflows/(.+)/badge\.svg#\1#' | sort -u)

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "==> README.md badge(s) reference a workflow file that doesn't exist:" >&2
    printf '    %s\n' "${missing[@]}" >&2
    exit 1
fi

echo "==> All README.md workflow badges point at an existing workflow file"
