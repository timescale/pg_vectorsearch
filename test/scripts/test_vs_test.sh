#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# TAP tests for vs_test.sh, the framework that the other test files in
# this directory source -- so the suite checking itself, rather than
# checking anything in scripts/.
#
# What it asserts: a repository built by a test must not read or write
# the repository that test was started in, whatever GIT_DIR and
# GIT_INDEX_FILE happen to be set to. Getting that wrong rewrites the
# caller's index rather than failing a test.
#
# Usage: ./test/scripts/test_vs_test.sh

set -uo pipefail

# shellcheck source=test/scripts/vs_test.sh
source "$(dirname "${BASH_SOURCE[0]}")/vs_test.sh"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stands in for the repository a developer runs the suite from. One
# commit and one tracked file is all the assertions below need: they
# check that both are still there once a fixture has been built.
caller_repo="$VS_WORKSPACE/caller-repo"
git init -q -b main "$caller_repo"
git -C "$caller_repo" config user.email test@example.invalid
git -C "$caller_repo" config user.name "Framework Test"
printf 'keep me\n' >"$caller_repo/tracked.txt"
git -C "$caller_repo" add tracked.txt
git -C "$caller_repo" commit -qm "a commit to lose"

# Build a fixture with GIT_DIR and GIT_INDEX_FILE pointing at
# caller_repo, which is what `git rebase --exec` and git's hooks
# export. The work tree is left unset, as those leave it, so git
# falls back to the current directory -- the fixture -- and stages the
# fixture's files into caller_repo's index.
#
# A child shell, because what this reproduces is a test script that
# starts with GIT_DIR and GIT_INDEX_FILE already in its environment
# and sources vs_test.sh itself -- the order a hook or a rebase exec
# produces. This script cannot stand in for it: it sourced vs_test.sh
# at the top, which cleared both variables here before any of this
# ran.
#
# shellcheck disable=SC2016  # $1 is the child's argument, not ours
env GIT_DIR="$caller_repo/.git" \
    GIT_INDEX_FILE="$caller_repo/.git/index" \
    bash -c '
        set -uo pipefail
        source "$1/vs_test.sh"
        source "$1/release_fixtures.sh"
        install_stub_gh
        dir="$(new_fixture 0.1.0)"
        cd "$dir" || exit 1
        git add -A
    ' _ "$HERE" >/dev/null 2>&1

expect_eq "a fixture leaves the caller's repository unchanged" \
    "" "$(git -C "$caller_repo" status --porcelain)"

# tracked.txt is never removed from disk, only from the index, which
# is why the failure looks like every file being deleted at once.
expect_eq "the caller's index still lists its files" \
    "tracked.txt" "$(git -C "$caller_repo" ls-files)"

tap_finish
