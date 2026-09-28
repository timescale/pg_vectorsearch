#!/bin/bash
# Enforce a single-commit pull request.
#
# Usage: ./scripts/ci/check-commit-count.sh
#
# Environment:
#   BASE_SHA            Base of the pull request (default: origin/main).
#   HEAD_SHA            Head of the pull request (default: HEAD).
#   BODY                The pull request description.
#   AUTO_MERGE_METHOD   The auto-merge method if one is enabled.
#
# A pull request merges as one commit unless its description carries the
# trailer
#
#     Disable-check: commit-count
#
# which says every commit stands on its own (see the create-pr skill). The
# match is case-insensitive and ignores the rest of the line. Squash
# auto-merge makes the count moot, so it passes too. Six or more commits
# fail regardless: that history belongs on a branch, not in a pull request.

set -euo pipefail

TRAILER='disable-check:.*\bcommit-count\b'
HARD_LIMIT=5

range="${BASE_SHA:-origin/main}..${HEAD_SHA:-HEAD}"
BODY=${BODY:-}
AUTO_MERGE_METHOD=${AUTO_MERGE_METHOD:-}

count=$(git rev-list --no-merges --count "$range")
list() { git log --format='    %h %s' "$range"; }

if (( count > HARD_LIMIT )); then
    echo "==> $count commits in the pull request; the limit is $HARD_LIMIT:" >&2
    list >&2
    echo "This limit has no override. Squash the history before merging." >&2
    exit 1
fi

if [[ ${AUTO_MERGE_METHOD,,} == "squash" ]]; then
    echo "==> $count commit(s); squash auto-merge is enabled, so the count is moot"
    exit 0
fi

if grep -Eqi "^$TRAILER" <<<"$BODY"; then
    echo "==> $count commit(s); the description disables the commit-count check"
    exit 0
fi

if (( count != 1 )); then
    echo "==> $count commits in the pull request; there should be one:" >&2
    list >&2
    echo "Squash them, or add this trailer to the pull request description" >&2
    echo "if each commit stands on its own:" >&2
    echo >&2
    echo "    Disable-check: commit-count" >&2
    exit 1
fi

echo "==> One commit in the pull request"
