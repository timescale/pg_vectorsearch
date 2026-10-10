#!/bin/bash
# Refuse ordinary work while a release is in flight, so nothing lands on
# a version number that has already shipped.
#
# Usage: ./scripts/ci/release-window.sh <base-ref>
#
# Between the release pull request merging and the development-cycle
# pull request merging, the base branch carries a released version. A
# pull request that lands in that window is built, tested and recorded
# as a version someone can already download, and its commits are
# reachable from a tag that does not contain them.
#
# The window needs no state of its own: the base's own VERSION says
# which side of it a pull request is on. It opens when the release
# lands and closes when the cycle reopens, both of which are commits on
# the base branch.
#
# The base is named rather than defaulted. A pull_request checkout is a
# synthetic merge of the branch into its base, so VERSION in the
# checkout is the branch's value -- reading HEAD would ask the wrong
# side of the very question this answers.
#
# Reports success on every pull request outside the window, which is
# what lets it be required on all of them.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

require_repo_root

[[ $# -eq 1 ]] || die "usage: $0 <base-ref>"

BASE="$1"
HEAD_REF="${HEAD_REF:-$(git branch --show-current)}"

base_version="$(git show "$BASE:VERSION" | tr -d '[:space:]')" ||
    die "cannot read VERSION at $BASE"

if is_dev_version "$base_version"; then
    log "$BASE is at $base_version: no release is in flight"
    exit 0
fi

# The one pull request allowed through is the one that closes the
# window. Letting it through costs nothing: release-dev-bump-check.sh
# has already asserted it changes VERSION and nothing else, so a branch
# named this way cannot smuggle anything in alongside the bump.
if [[ "$HEAD_REF" == chore/dev-* ]]; then
    log "$BASE is at $base_version and $HEAD_REF reopens the" \
        "development cycle: letting it through"
    exit 0
fi

die "$BASE is at $base_version, a version that has been released, so" \
    "a release is in flight -- wait for the chore/dev-* pull request" \
    "to reopen the development cycle, then re-run this check"
