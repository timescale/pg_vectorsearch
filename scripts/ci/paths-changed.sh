#!/bin/bash
# Report whether a pull request touches any of the given paths.
#
# Usage: ./scripts/ci/paths-changed.sh <pathspec>...
#
# Environment:
#   BASE_SHA   start of the range to compare (default: origin/main)
#   HEAD_SHA   end of the range to compare (default: HEAD)
#
# Writes changed=true|false to $GITHUB_OUTPUT when set, and prints the
# verdict. Always exits 0: the answer is the output, not the status.
#
# This exists so a workflow can keep its path filtering while still
# reporting a check run. A paths filter on the `on:` trigger skips the
# whole workflow, which creates no check run at all, so a ruleset that
# requires the context waits on it forever. Deciding inside the job
# instead leaves the job running, and so reporting, while the work
# itself is skipped with `if:`.
#
# Undecidable cases answer true. A security check that silently skips
# because a range could not be resolved is worse than one that runs
# when it did not need to.

set -uo pipefail

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <pathspec>..." >&2
    exit 2
fi

emit() {
    echo "$1"
    [[ -z "${GITHUB_OUTPUT:-}" ]] || echo "changed=$1" >>"$GITHUB_OUTPUT"
}

base="${BASE_SHA:-origin/main}"
head="${HEAD_SHA:-HEAD}"

if ! git rev-parse -q --verify "$base" >/dev/null ||
    ! git rev-parse -q --verify "$head" >/dev/null; then
    echo "cannot resolve $base..$head; assuming changed" >&2
    emit true
    exit 0
fi

# Three dots: compare against the merge base, so commits that landed on
# the base branch after this one was cut do not count as changes here.
if ! files="$(git diff --name-only "$base...$head" -- "$@" 2>/dev/null)"; then
    echo "git diff failed for $base...$head; assuming changed" >&2
    emit true
    exit 0
fi

if [[ -n "$files" ]]; then
    echo "matched:" >&2
    while IFS= read -r f; do echo "  $f" >&2; done <<<"$files"
    emit true
else
    emit false
fi
