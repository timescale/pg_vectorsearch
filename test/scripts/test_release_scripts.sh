#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# TAP tests for scripts/ci/release-open-dev-cycle.sh and
# scripts/ci/release-dev-bump-check.sh.
#
# Usage: ./test/scripts/test_release_scripts.sh
#
# These scripts push branches and open pull requests. Rather than reach
# a real remote, a fixture is a local bare repository standing in for
# one with a clone pointing at it, and `gh` is a stub earlier on PATH
# that records its invocations in $GH_LOG and answers from the
# environment:
#
#   STUB_OPEN_PR   what `gh pr list` reports, empty for none
#   STUB_PR_URL    what `gh pr create` echoes
#   STUB_MERGE_RC  what `gh pr merge` exits with
#
# So the git work stays real -- branches, file writes, commits, pushes
# -- while the GitHub calls become observable. What that cannot cover
# is GitHub's own behaviour, which needs a repository with a ruleset.
#
# The fixtures live here rather than beside vs_test.sh because this is
# the only file using them; test/unit does the same, extracting
# posting_fixtures.h only once a second test file needed it.

set -uo pipefail

# shellcheck source=test/scripts/vs_test.sh
source "$(dirname "${BASH_SOURCE[0]}")/vs_test.sh"

# The repository under test, which fixtures copy the scripts from.
VS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# A fixture repository: a bare remote, a clone of it carrying the real
# release scripts, and one commit whose VERSION is $1. Echoes the
# clone's path; the bare remote is that path with .git appended, and
# the stub's log is .gh.log.
#
# Pass RELEASE_REPO=<clone path> when running a script against it:
# release_remote() matches remotes by URL with any .git suffix
# stripped, so the slug is the path without it.
new_fixture() {
    local version="$1" dir
    dir="$VS_WORKSPACE/fx$VS_TESTS-$RANDOM"
    git init -q --bare "$dir.git"
    git init -q -b main "$dir"
    (
        cd "$dir" || exit 1
        git remote add origin "$dir.git"
        git config user.email test@example.invalid
        git config user.name "Release Test"
        printf "project('pg_vectorsearch', 'c')\n" >meson.build
        printf '%s\n' "$version" >VERSION
        mkdir -p scripts/ci
        cp "$VS_ROOT"/scripts/*.sh scripts/
        cp "$VS_ROOT"/scripts/ci/*.sh scripts/ci/
        git add -A
        git commit -qm "release $version"
        git push -q origin main
    ) >/dev/null 2>&1
    : >"$dir.gh.log"
    printf '%s\n' "$dir"
}

# Run a script inside a fixture with the stub's environment set.
# $1 fixture path, $2.. the command and its arguments.
in_fixture() {
    local dir="$1"
    shift
    env -C "$dir" RELEASE_REPO="$dir" GH_LOG="$dir.gh.log" "$@"
}

# How many times the stub saw a command, e.g. gh_calls "$dir" 'pr create'.
gh_calls() {
    grep -c "^gh $2" "$1.gh.log" 2>/dev/null || true
}

# A stub gh earlier on PATH than the real one.
install_stub_gh() {
    mkdir -p "$VS_WORKSPACE/bin"
    cat >"$VS_WORKSPACE/bin/gh" <<'STUB'
#!/bin/bash
printf 'gh %s\n' "$*" >>"$GH_LOG"
case "$1 $2" in
    "pr list")
        [[ -z "${STUB_OPEN_PR:-}" ]] || printf '%s\n' "$STUB_OPEN_PR" ;;
    "pr create")
        printf '%s\n' "${STUB_PR_URL:-https://example.invalid/pr/1}" ;;
    "pr merge") exit "${STUB_MERGE_RC:-0}" ;;
    *) : ;;
esac
STUB
    chmod +x "$VS_WORKSPACE/bin/gh"
    PATH="$VS_WORKSPACE/bin:$PATH"
    export PATH
}

install_stub_gh

# ----------------------------------------------------------------
# release-dev-bump-check.sh
# ----------------------------------------------------------------

check_case() {
    local desc="$1" base_version="$2" head_ref="$3" want="$4"
    shift 4
    local dir base head
    dir="$(new_fixture "$base_version")"
    base="$(git -C "$dir" rev-parse HEAD)"
    (
        cd "$dir" || exit 1
        "$@" >/dev/null 2>&1
        git add -A >/dev/null 2>&1
        git commit -qm "bump" >/dev/null 2>&1 || true
    )
    head="$(git -C "$dir" rev-parse HEAD)"
    expect_status "dev-bump: $desc" "$want" env -C "$dir" \
        HEAD_REF="$head_ref" ./scripts/ci/release-dev-bump-check.sh \
        "$base" "$head"
}

check_case "a branch that is not a dev cycle is not checked" \
    0.1.0 feat/whatever 0 true

check_case "VERSION only, correct -dev bump" \
    0.1.0 chore/dev-0.2.0-dev 0 \
    bash -c 'printf "0.2.0-dev\n" >VERSION'

check_case "a second changed file is refused" \
    0.1.0 chore/dev-0.2.0-dev 1 \
    bash -c 'printf "0.2.0-dev\n" >VERSION; printf x >>meson.build'

check_case "a value without a -dev suffix is refused" \
    0.1.0 chore/dev-0.2.0 1 \
    bash -c 'printf "0.2.0\n" >VERSION'

check_case "a branch name disagreeing with VERSION is refused" \
    0.1.0 chore/dev-0.2.0-dev 1 \
    bash -c 'printf "0.3.0-dev\n" >VERSION'

check_case "a base already on a -dev version is refused" \
    0.1.0-dev chore/dev-0.2.0-dev 1 \
    bash -c 'printf "0.2.0-dev\n" >VERSION'

check_case "a pull request changing nothing is refused" \
    0.1.0 chore/dev-0.2.0-dev 1 true

# ----------------------------------------------------------------
# release-open-dev-cycle.sh
# ----------------------------------------------------------------

open_cycle() {
    local dir="$1"
    shift
    in_fixture "$dir" ./scripts/ci/release-open-dev-cycle.sh "$@"
}

dir="$(new_fixture 0.1.0)"
expect_status "open-cycle: too few arguments" 1 open_cycle "$dir" 0.1.0
expect_status "open-cycle: a next version without -dev" 1 \
    open_cycle "$dir" 0.1.0 0.2.0
expect_status "open-cycle: a next version not greater than released" 1 \
    open_cycle "$dir" 0.2.0 0.1.0-dev
expect_status "open-cycle: VERSION disagreeing with the release" 1 \
    open_cycle "$dir" 0.9.9 1.0.0-dev

# The release job checks out a sha, so the base cannot be inferred.
git -C "$dir" switch -q --detach HEAD
expect_status "open-cycle: a detached checkout without a base" 1 \
    open_cycle "$dir" 0.1.0 0.2.0-dev
git -C "$dir" switch -q main

# The whole path, as the release job runs it.
happy="$(new_fixture 0.1.0)"
expect_status "open-cycle: opens the cycle" 0 \
    open_cycle "$happy" 0.1.0 0.2.0-dev main
expect_eq "open-cycle: VERSION carries the next cycle" \
    0.2.0-dev "$(tr -d '[:space:]' <"$happy/VERSION")"
expect_eq "open-cycle: the commit changes VERSION alone" \
    VERSION "$(git -C "$happy" diff --name-only HEAD~1 HEAD 2>/dev/null)"
expect_eq "open-cycle: the branch reaches the remote" \
    "refs/heads/chore/dev-0.2.0-dev" \
    "$(git -C "$happy.git" show-ref --heads | awk '$2 ~ /chore/ {print $2}')"
expect_eq "open-cycle: a pull request is opened" \
    1 "$(gh_calls "$happy" 'pr create')"
expect_eq "open-cycle: auto-merge is armed" \
    1 "$(gh_calls "$happy" 'pr merge .*--auto')"

# Re-running the release workflow must not open a second pull request.
: >"$happy.gh.log"   # count only the re-run
git -C "$happy" switch -q main
expect_status "open-cycle: re-run with the pull request open is a no-op" 0 \
    in_fixture "$happy" env STUB_OPEN_PR=https://example.invalid/pr/7 \
    ./scripts/ci/release-open-dev-cycle.sh 0.1.0 0.2.0-dev main
expect_eq "open-cycle: the re-run opens nothing" \
    0 "$(gh_calls "$happy" 'pr create')"

# A branch left behind with no pull request is ambiguous, not resumable.
expect_status "open-cycle: a stranded branch is refused" 1 \
    open_cycle "$happy" 0.1.0 0.2.0-dev main

# A release must not be reported failed because auto-merge is off.
armless="$(new_fixture 0.1.0)"
expect_status "open-cycle: auto-merge refusing does not fail the job" 0 \
    in_fixture "$armless" env STUB_MERGE_RC=1 \
    ./scripts/ci/release-open-dev-cycle.sh 0.1.0 0.2.0-dev main

tap_finish
