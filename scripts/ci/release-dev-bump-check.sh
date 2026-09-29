#!/bin/bash
# Assert that a development-cycle pull request does nothing but reopen
# the cycle, so it can be required on every pull request.
#
# Usage: ./scripts/ci/release-dev-bump-check.sh <base-sha> <head-sha>
#
# Reports success on anything that is not a dev-cycle branch, because a
# required check has to return a verdict on every pull request. On a
# chore/dev-* branch it asserts that only VERSION changed, that the new
# value carries a -dev suffix, and that it is the bump the base's
# version calls for -- the guarantees a ruleset cannot express, since
# rulesets never look at file contents.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

require_repo_root

[[ $# -eq 2 ]] || die "usage: $0 <base-sha> <head-sha>"

BASE_SHA="$1"
HEAD_SHA="$2"
HEAD_REF="${HEAD_REF:-$(git branch --show-current)}"

if [[ "$HEAD_REF" != chore/dev-* ]]; then
    log "$HEAD_REF is not a development-cycle branch; nothing to check"
    exit 0
fi

log "checking the $HEAD_REF bump"

changed="$(git diff --name-only "$BASE_SHA" "$HEAD_SHA")"
[[ -n "$changed" ]] ||
    die "this pull request changes nothing, so there is no bump to" \
        "check -- the branch is at its base"
[[ "$changed" == "VERSION" ]] ||
    die "a development-cycle pull request may only change VERSION," \
        "but this one changes:" $'\n'"$changed"

base_version="$(git show "$BASE_SHA:VERSION" | tr -d '[:space:]')"
head_version="$(git show "$HEAD_SHA:VERSION" | tr -d '[:space:]')"

validate_version "$head_version"

is_dev_version "$head_version" ||
    die "VERSION becomes $head_version, which carries no -dev suffix" \
        "-- the release gate would read the next push as a release"

if is_dev_version "$base_version"; then
    die "the base already carries the development version" \
        "$base_version, so there is no cycle to reopen"
fi

# The branch name and the file have to agree, or the bump that merges
# is not the one the branch was reviewed as.
[[ "$HEAD_REF" == "chore/dev-$head_version" ]] ||
    die "branch $HEAD_REF does not match VERSION $head_version"

expected="$(next_dev_version "$base_version" \
    "${GITHUB_BASE_REF:-$(git branch --show-current)}")"
[[ "$head_version" == "$expected" ]] ||
    warn "bumping $base_version to $head_version, where the default" \
        "bump is $expected -- intended for a major bump, wrong" \
        "otherwise"

log "$base_version -> $head_version, VERSION only"
