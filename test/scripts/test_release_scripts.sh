#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# TAP tests for scripts/ci/release-open-dev-cycle.sh and
# scripts/ci/release-dev-bump-check.sh.
#
# Usage: ./test/scripts/test_release_scripts.sh
#
# Fixtures come from release_fixtures.sh; read its header for how a
# local bare repository stands in for the remote, and for what that
# approach cannot cover.

set -uo pipefail

# shellcheck source=test/scripts/vs_test.sh
source "$(dirname "${BASH_SOURCE[0]}")/vs_test.sh"
# shellcheck source=test/scripts/release_fixtures.sh
source "$(dirname "${BASH_SOURCE[0]}")/release_fixtures.sh"

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
