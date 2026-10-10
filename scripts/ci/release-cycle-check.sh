#!/bin/bash
# Keep pull requests consistent with where the release cycle is.
#
# Usage: ./scripts/ci/release-cycle-check.sh <base> <pr-head>
#
# VERSION on main and on the release (patch) branches is either a
# development version carrying a `-dev` suffix, or a version that has
# been released. Which one it is decides what may merge:
#
#   0.2.0-dev  ordinary work merges, and a release PR may start 0.2.0
#   0.2.0      nothing merges except the bump PR back to 0.3.0-dev
#
# The second state is the release window. It opens when the release PR
# merges and closes when the bump PR does, and the bump PR needs an
# approval, so it lasts as long as that takes. Anything merged in
# between is built and tested as a version people can already
# download, and the tag of that version will not contain it.
#
# What a pull request is gets decided by what it does to VERSION, never
# by what its branch is called. Nothing here reads a branch name: a
# bump is a bump whether the pipeline opened it as chore/dev-0.3.0-dev
# or somebody cut it by hand as foo/bar.
#
# A pull request that leaves VERSION alone is reported on and passed,
# so this can be required on every pull request.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

require_repo_root

[[ $# -eq 2 ]] || die "usage: $0 <base> <pr-head>"

# Where the pull request would merge, read live rather than from the
# event, so a re-run sees the branch as it is now.
BASE="$1"
# The tip of the pull request's branch -- the commit that was reviewed,
# not the merge GitHub synthesizes to test it. That merge carries a
# merged VERSION belonging to neither side, and a base snapshot that
# goes stale.
PR_HEAD="$2"

base_version="$(git show "$BASE:VERSION" | tr -d '[:space:]')" ||
    die "cannot read VERSION at $BASE"

in_window() {
    ! is_dev_version "$base_version"
}

# Three dots: diffed from the fork point, so this is what the pull
# request carries. Against the base tip instead, a branch that predates
# a release would show VERSION as changed because the base moved it,
# and an empty stale branch would read as the bump.
changed="$(git diff --name-only "$BASE...$PR_HEAD")"

if ! grep -qx VERSION <<<"$changed"; then
    if in_window; then
        die "$BASE is at $base_version, which has been released, so" \
            "a release is in flight -- this can merge once the bump" \
            "PR reopens the development cycle"
    fi
    log "VERSION untouched, $BASE is at $base_version: nothing to check"
    exit 0
fi

head_version="$(git show "$PR_HEAD:VERSION" | tr -d '[:space:]')" ||
    die "cannot read VERSION at $PR_HEAD"

validate_version "$head_version"

# A release PR, which release-check.sh examines. All this decides is
# whether one may start here.
if ! is_dev_version "$head_version"; then
    if in_window; then
        die "this releases $head_version, but $BASE is already at" \
            "$base_version and that release is still in flight --" \
            "the bump PR has to reopen the development cycle first"
    fi
    log "$base_version -> $head_version is a release;" \
        "release-check.sh has this one"
    exit 0
fi

# The bump PR. The pipeline opens it and arms auto-merge, so nothing
# human reads it before it lands, and a ruleset cannot help: rulesets
# see refs, metadata and paths, never file contents.
log "checking the bump to $head_version"

if ! in_window; then
    die "this reopens the development cycle at $head_version, but" \
        "$BASE is already at $base_version, so there is no cycle to" \
        "reopen"
fi

[[ "$changed" == "VERSION" ]] ||
    die "the bump PR may only change VERSION, but this one changes:" \
        $'\n'"$changed"

expected="$(next_dev_version "$base_version" \
    "${GITHUB_BASE_REF:-$(git branch --show-current)}")"
[[ "$head_version" == "$expected" ]] ||
    warn "bumping $base_version to $head_version, where the default" \
        "bump is $expected -- intended for a major bump, wrong" \
        "otherwise"

log "$base_version -> $head_version, VERSION only"
