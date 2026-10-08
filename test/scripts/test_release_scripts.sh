#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# TAP tests for scripts/ci/release-open-dev-cycle.sh,
# scripts/ci/release-dev-bump-check.sh, scripts/ci/release-check.sh's
# classification and scripts/ci/release-tarball-comment.sh.
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

# A candidate is a step toward its own version, so it reopens at that
# version rather than past it. release-lib.sh's trailer check already
# treats an equal base as correct for a candidate; this is the same rule
# on the other side of the publish.
rc="$(new_fixture 0.1.0-rc1)"
expect_status "open-cycle: a candidate reopens at its own version" 0 \
    open_cycle "$rc" 0.1.0-rc1 0.1.0-dev main
expect_eq "open-cycle: the candidate's cycle reopens at its version" \
    0.1.0-dev "$(tr -d '[:space:]' <"$rc/VERSION")"

# An equal base is correct only for a candidate: a final release
# reopening at its own version would leave the cycle where it was.
expect_status "open-cycle: a final release may not reopen at its own" 1 \
    open_cycle "$dir" 0.1.0 0.1.0-dev

# ----------------------------------------------------------------
# release-check.sh: the verdict it hands the packaging job
# ----------------------------------------------------------------
#
# The job packages a tarball only when this says the PR is a release, so
# a wrong verdict either ships nothing or builds on every pull request.

release_verdict() {
    local desc="$1" version="$2" want_release="$3" want_version="$4"
    local dir out
    dir="$(new_fixture "$version")"
    if [[ "$want_release" == true ]]; then
        printf '# Changelog\n\n## [%s] - 2026-01-01\n\n- it works\n' \
            "$version" >"$dir/CHANGELOG.md"
        git -C "$dir" add CHANGELOG.md >/dev/null 2>&1
        # The trailer naming the next cycle, which a release commit
        # carries and the check refuses to release without.
        git -C "$dir" commit -q -m "release notes" \
            -m "Next-Version: 0.2.0-dev" >/dev/null 2>&1
    fi
    out="$dir.output"
    : >"$out"
    expect_status "release-check: $desc exits zero" 0 \
        in_fixture "$dir" env GITHUB_OUTPUT="$out" \
        ./scripts/ci/release-check.sh
    expect_eq "release-check: $desc reports release=$want_release" \
        "release=$want_release" "$(grep '^release=' "$out")"
    expect_eq "release-check: $desc reports the version" \
        "version=$want_version" "$(grep '^version=' "$out")"
}

release_verdict "a release checkout" 0.1.0 true 0.1.0
release_verdict "a development checkout" 0.2.0-dev false ""

# A malformed VERSION must fail rather than be classified. The suffix
# test alone would read "not-a-version" as neither a release nor a
# development version, and "1.2-dev" as a development version -- so the
# packaging job would be handed a verdict derived from a typo. Nothing
# may be emitted either: a job reading release= from a half-written
# output would act on it.
release_verdict_refused() {
    local desc="$1" version="$2" dir out
    dir="$(new_fixture 0.1.0)"
    printf '%s\n' "$version" >"$dir/VERSION"
    out="$dir.output"
    : >"$out"
    expect_status "release-check: $desc is refused" 1 \
        in_fixture "$dir" env GITHUB_OUTPUT="$out" \
        ./scripts/ci/release-check.sh
    expect_eq "release-check: $desc emits no verdict" "" "$(cat "$out")"
}

release_verdict_refused "a version that is not one" not-a-version
release_verdict_refused "a two-component version" 1.2
release_verdict_refused "a two-component development version" 1.2-dev
release_verdict_refused "a v-prefixed version" v0.1.0
release_verdict_refused "a four-component version" 0.1.0.1
release_verdict_refused "a doubled suffix separator" 0.1.0--dev
release_verdict_refused "an empty VERSION" ""

# ----------------------------------------------------------------
# release-tarball-comment.sh
# ----------------------------------------------------------------

comment() {
    local dir="$1"
    shift
    in_fixture "$dir" env GH_TOKEN=x REPO=owner/repo PR=7 \
        VERSION=0.1.0 ARTIFACT_URL=https://example.invalid/a/1 \
        HEAD_SHA=abc123def4567890 "$@" \
        ./scripts/ci/release-tarball-comment.sh
}

fresh="$(new_fixture 0.1.0)"
expect_status "tarball-comment: comments on a pull request" 0 \
    comment "$fresh" env STUB_COMMENT_ID=
expect_eq "tarball-comment: the comment is created" \
    1 "$(gh_calls "$fresh" 'api --method POST')"
expect_eq "tarball-comment: nothing is patched" \
    0 "$(gh_calls "$fresh" 'api --method PATCH')"
# The packaged commit, so a reviewer can tell whether the archive is the
# one their latest push produced.
expect_status "tarball-comment: the body names the packaged commit" 0 \
    grep -qE 'Packaged from .abc123def456.' "$fresh.gh.log"
expect_status "tarball-comment: the body links the artifact" 0 \
    grep -q 'https://example.invalid/a/1' "$fresh.gh.log"

# A release PR is re-pushed whenever the notes are amended, and a second
# comment each time would bury the rest of the review.
again="$(new_fixture 0.1.0)"
expect_status "tarball-comment: a re-push updates in place" 0 \
    comment "$again" env STUB_COMMENT_ID=4242
patched='api --method PATCH repos/owner/repo/issues/comments/4242'
expect_eq "tarball-comment: the existing comment is patched" \
    1 "$(gh_calls "$again" "$patched")"
expect_eq "tarball-comment: no second comment is created" \
    0 "$(gh_calls "$again" 'api --method POST')"

# Every input is load-bearing: the link is the whole point of the
# comment, and the rest decide which pull request it reaches. Missing or
# empty, each must fail rather than post a review request naming the
# wrong PR or pointing nowhere. Empty matters as much as unset -- these
# arrive from workflow expressions, which render an absent field as "".
comment_refused() {
    local desc="$1"
    shift
    local dir
    dir="$(new_fixture 0.1.0)"
    expect_status "tarball-comment: $desc is refused" 1 \
        in_fixture "$dir" env GH_TOKEN=x "$@" \
        ./scripts/ci/release-tarball-comment.sh
    expect_eq "tarball-comment: $desc posts nothing" \
        0 "$(gh_calls "$dir" 'api --method')"
}

full=(REPO=owner/repo PR=7 VERSION=0.1.0
      ARTIFACT_URL=https://example.invalid/a/1 HEAD_SHA=abc123def456)

for missing in REPO PR VERSION ARTIFACT_URL HEAD_SHA; do
    without=()
    for kv in "${full[@]}"; do
        [[ "$kv" == "$missing="* ]] || without+=("$kv")
    done
    comment_refused "an unset $missing" "${without[@]}"
    comment_refused "an empty $missing" "${without[@]}" "$missing="
done

tap_finish
