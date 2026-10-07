#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# TAP tests for scripts/ci/paths-changed.sh, which decides whether a
# workflow's expensive steps need to run. Getting a false out of it
# skips a check, so the undecidable cases are tested too.
#
# Usage: ./test/scripts/test_paths_changed.sh

set -uo pipefail

# shellcheck source=test/scripts/vs_test.sh
source "$(dirname "${BASH_SOURCE[0]}")/vs_test.sh"

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$SCRIPT/scripts/ci/paths-changed.sh"

# A repository with a base commit and one commit on top of it touching
# the paths named in $1 (space separated).
new_repo() {
    local touched="$1" dir
    dir="$VS_WORKSPACE/pc$RANDOM-$VS_TESTS"
    git init -q -b main "$dir"
    (
        cd "$dir" || exit 1
        git config user.email t@example.invalid
        git config user.name "Paths Test"
        mkdir -p sql scripts/ci
        printf 'base\n' >README.md
        git add -A && git commit -qm base
        git branch -q base-marker
        for f in $touched; do
            mkdir -p "$(dirname "$f")"
            printf 'changed\n' >"$f"
        done
        git add -A && git commit -qm change
    ) >/dev/null 2>&1
    printf '%s\n' "$dir"
}

run_in() {
    local dir="$1" base="$2"
    shift 2
    env -C "$dir" BASE_SHA="$base" HEAD_SHA=HEAD "$SCRIPT" "$@" 2>/dev/null
}

# A touched path inside the filter.
dir="$(new_repo 'sql/pg_vectorsearch.sql')"
expect_eq "a matching path reports changed" \
    true "$(run_in "$dir" base-marker 'sql/**')"

# A touched path outside it.
expect_eq "an unrelated path reports unchanged" \
    false "$(run_in "$dir" base-marker 'scripts/ci/pgspot.sh')"

# Several filters, one of which matches.
dir2="$(new_repo 'scripts/ci/pgspot.sh')"
expect_eq "any one filter matching is enough" \
    true "$(run_in "$dir2" base-marker 'sql/**' 'scripts/ci/pgspot.sh')"

# Undecidable: a base that does not exist must not silently skip a
# security check.
expect_eq "an unresolvable base reports changed" \
    true "$(run_in "$dir" no-such-ref 'sql/**')"

# Undecidable the other way: both ends resolve, but a three-dot diff
# across unrelated histories has no merge base and fails outright.
dir3="$(new_repo 'sql/pg_vectorsearch.sql')"
(
    cd "$dir3" || exit 1
    git checkout -q --orphan unrelated
    git rm -rq --cached .
    printf 'elsewhere\n' >other.txt
    git add other.txt && git commit -qm unrelated
) >/dev/null 2>&1
expect_eq "a diff with no merge base reports changed" \
    true "$(run_in "$dir3" main 'sql/**')"

# An empty range reports unchanged, which is why a workflow passes an
# explicit base rather than letting the default compare a branch tip
# against itself.
expect_eq "a range with identical endpoints reports unchanged" \
    false "$(run_in "$dir" HEAD 'sql/**')"

# The verdict reaches a workflow through GITHUB_OUTPUT.
out="$dir.output"
: >"$out"
env -C "$dir" BASE_SHA=base-marker HEAD_SHA=HEAD GITHUB_OUTPUT="$out" \
    "$SCRIPT" 'sql/**' >/dev/null 2>&1
expect_eq "the verdict is written to GITHUB_OUTPUT" \
    "changed=true" "$(cat "$out")"

# Called with no filters at all, which is a workflow authoring mistake
# rather than a verdict.
expect_status "no pathspec is a usage error" 2 \
    env -C "$dir" "$SCRIPT"

tap_finish
