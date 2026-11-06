#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# TAP tests for scripts/ci/release-cycle-check.sh,
# scripts/ci/release-open-dev-cycle.sh, scripts/prepare-release.sh,
# scripts/open-release-pr.sh, scripts/ci/release-check.sh's
# classification,
# scripts/ci/release-tarball-comment.sh, scripts/ci/release-create.sh
# and scripts/ci/release-publish.sh.
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
install_stub_git_cliff

# ----------------------------------------------------------------
# release-cycle-check.sh
# ----------------------------------------------------------------

# A base commit and a pull request on top of it. $@ makes the change
# the pull request carries.
cycle_case() {
    local desc="$1" base_version="$2" want="$3"
    shift 3
    local dir base head
    dir="$(new_fixture "$base_version")"
    base="$(git -C "$dir" rev-parse HEAD)"
    (
        cd "$dir" || exit 1
        "$@" >/dev/null 2>&1
        git add -A >/dev/null 2>&1
        git commit -qm "pull request" >/dev/null 2>&1 || true
    )
    head="$(git -C "$dir" rev-parse HEAD)"
    expect_status "cycle: $desc" "$want" env -C "$dir" \
        ./scripts/ci/release-cycle-check.sh "$base" "$head"
}

touch_file() { bash -c 'printf x >>meson.build'; }
set_version() { bash -c "printf '%s\\n' '$1' >VERSION"; }

# The development cycle: ordinary work merges, and a release may start.
cycle_case "ordinary work during the cycle" 0.1.0-dev 0 touch_file
cycle_case "a release may start from the cycle" 0.1.0-dev 0 \
    set_version 0.1.0
cycle_case "a bump with no cycle to reopen is refused" 0.1.0-dev 1 \
    set_version 0.2.0-dev

# The release window: only the bump gets through.
cycle_case "ordinary work in the window is refused" 0.1.0 1 touch_file
cycle_case "a second release in the window is refused" 0.1.0 1 \
    set_version 0.2.0
cycle_case "the bump reopens the cycle" 0.1.0 0 set_version 0.2.0-dev
cycle_case "a candidate opens a window too" 0.1.0-rc1 1 touch_file
cycle_case "a candidate's cycle reopens" 0.1.0-rc1 0 \
    set_version 0.1.0-dev

# Nothing here reads a branch name, so a bump cut by hand under any
# name is admitted, and a bump carrying anything else is not.
cycle_case "a bump carrying a second file is refused" 0.1.0 1 \
    bash -c 'printf "0.2.0-dev\\n" >VERSION; printf x >>meson.build'

cycle_case "an unparseable version is refused" 0.1.0 1 \
    set_version not-a-version
cycle_case "a pull request changing nothing is passed over" 0.1.0-dev \
    0 true

dir="$(new_fixture 0.1.0)"
expect_status "cycle: both refs are required" 1 env -C "$dir" \
    ./scripts/ci/release-cycle-check.sh HEAD
expect_status "cycle: an unreadable base is refused" 1 env -C "$dir" \
    ./scripts/ci/release-cycle-check.sh nope HEAD

# ----------------------------------------------------------------
# release-open-dev-cycle.sh
# ----------------------------------------------------------------

open_cycle() {
    local dir="$1"
    shift
    in_fixture "$dir" ./scripts/ci/release-open-dev-cycle.sh "$@"
}

dir="$(new_fixture 0.1.0)"
expect_status "open-cycle: no arguments at all" 1 open_cycle "$dir"
# The fixture's commit is not one open-release-pr.sh wrote, so there is
# no trailer to fall back on and nothing says which cycle to open.
expect_status "open-cycle: no next version and no trailer" 1 \
    open_cycle "$dir" 0.1.0
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


# The next version comes from the release commit when it is not given,
# which is what lets a standalone re-run be told only what was
# released.
trailered="$(new_fixture 0.1.0)"
git -C "$trailered" commit -q --allow-empty \
    -m "chore: release 0.1.0" -m "Next-Version: 0.2.0-dev"
expect_status "open-cycle: the trailer supplies the next version" 0 \
    open_cycle "$trailered" 0.1.0 "" main
expect_eq "open-cycle: it reopens at the declared cycle" \
    0.2.0-dev "$(tr -d '[:space:]' <"$trailered/VERSION")"

# Attribution and the authenticated remote are the script's, not the
# workflow's, and must not touch a tree someone is running this in by
# hand.
plain="$(new_fixture 0.1.0)"
expect_status "open-cycle: no APP_SLUG opens the cycle anyway" 0 \
    open_cycle "$plain" 0.1.0 0.2.0-dev main
expect_eq "open-cycle: it leaves the committer alone" \
    "Release Test" "$(git -C "$plain" config user.name)"
expect_eq "open-cycle: it leaves the remote alone" \
    "$plain.git" "$(git -C "$plain" remote get-url origin)"

botted="$(new_fixture 0.1.0)"
expect_status "open-cycle: APP_SLUG commits as the bot" 0 \
    in_fixture "$botted" env GH_TOKEN=t APP_SLUG=vectorsearch-bot \
    STUB_COMMENT_ID=4242 \
    ./scripts/ci/release-open-dev-cycle.sh 0.1.0 0.2.0-dev main
expect_eq "open-cycle: the commit carries the bot identity" \
    "vectorsearch-bot[bot]" \
    "$(git -C "$botted" log -1 --format=%an chore/dev-0.2.0-dev)"
# The bot resolves here, so the refusal is the token's alone.
tokenless="$(new_fixture 0.1.0)"
expect_status "open-cycle: APP_SLUG without a token is refused" 1 \
    in_fixture "$tokenless" env -u GH_TOKEN APP_SLUG=vectorsearch-bot \
    STUB_COMMENT_ID=4242 \
    ./scripts/ci/release-open-dev-cycle.sh 0.1.0 0.2.0-dev main
expect_eq "open-cycle: the tokenless run leaves the committer alone" \
    "Release Test" "$(git -C "$tokenless" config user.name)"
expect_eq "open-cycle: the tokenless run pushes nothing" \
    "" "$(git -C "$tokenless.git" show-ref --heads chore/dev-0.2.0-dev ||
        true)"

# ----------------------------------------------------------------
# open-release-pr.sh
# ----------------------------------------------------------------
#
# A clean checkout of the base and two finished files, the changelog
# and the pull request body, at $dir.changelog.md and $dir.body.md.
# The script builds the release commit from them, so the refusals
# below are about those files and about the checkout.

open_pr_fixture() {
    local dir
    dir="$(new_fixture 0.1.0-dev)"
    write_changelog "$dir.changelog.md" 0.1.0
    write_body "$dir.body.md" 0.1.0 "Next-Version: 0.2.0-dev"
    printf '%s\n' "$dir"
}

# $1 path, $2 the version with an entry.
write_changelog() {
    printf '# Changelog\n\n## [%s] - 2026-01-01\n\n- it works\n' "$2" >"$1"
}

# $1 path, $2 version, $3 the trailer paragraph, $4 an optional first
# line.
write_body() {
    {
        [[ -z "${4:-}" ]] || printf '%s\n' "$4"
        printf 'Releases %s.\n\n## Highlights\n\n- it works\n\n' "$2"
        printf '%s\n' "$3"
    } >"$1"
}

open_pr() {
    local dir="$1"
    shift
    in_fixture "$dir" ./scripts/open-release-pr.sh "$@"
}

# Every refusal must leave the remote without a release branch.
pushed_branches() {
    git -C "$1.git" show-ref --heads | awk '$2 ~ /release/ {print $2}'
}

# The trailer on the fixture's current commit.
next_trailer() {
    git -C "$1" log -1 --format='%(trailers:key=Next-Version,valueonly)' |
        tr -d '[:space:]'
}

expect_status "open-pr: five arguments are required" 1 \
    open_pr "$(open_pr_fixture)" 0.1.0 "" main

happy="$(open_pr_fixture)"
happy_out="$happy.output"
: >"$happy_out"
expect_status "open-pr: rebuilds the release commit and opens the PR" 0 \
    in_fixture "$happy" env GITHUB_OUTPUT="$happy_out" \
    ./scripts/open-release-pr.sh 0.1.0 0.2.0-dev main \
    "$happy.changelog.md" "$happy.body.md"
expect_eq "open-pr: the checkout is on the release branch" \
    chore/release-0.1.0 "$(git -C "$happy" branch --show-current)"
expect_eq "open-pr: the commit changes the changelog and VERSION alone" \
    "CHANGELOG.md VERSION" \
    "$(git -C "$happy" diff --name-only HEAD~1 HEAD | tr '\n' ' ' |
        sed 's/ $//')"
expect_eq "open-pr: VERSION carries the release" \
    0.1.0 "$(tr -d '[:space:]' <"$happy/VERSION")"
expect_eq "open-pr: the commit carries the trailer" \
    0.2.0-dev "$(next_trailer "$happy")"
expect_eq "open-pr: the branch reaches the remote" \
    refs/heads/chore/release-0.1.0 "$(pushed_branches "$happy")"
expect_eq "open-pr: a pull request is opened" \
    1 "$(gh_calls "$happy" 'pr create')"
expect_eq "open-pr: the pull request URL is reported" \
    "url=https://example.invalid/pr/1" "$(grep '^url=' "$happy_out")"
expect_eq "open-pr: without APP_SLUG the author is left alone" \
    "Release Test" "$(git -C "$happy" log -1 --format=%an)"

# The workflow's open job checks out a sha, so this is the only path
# CI takes. The cut starts from the detached commit and the branch is
# pushed from it.
detached="$(open_pr_fixture)"
git -C "$detached" switch -q --detach HEAD
expect_status "open-pr: cuts from a detached checkout of the base" 0 \
    open_pr "$detached" 0.1.0 0.2.0-dev main \
    "$detached.changelog.md" "$detached.body.md"
expect_eq "open-pr: the detached cut reaches the remote" \
    refs/heads/chore/release-0.1.0 "$(pushed_branches "$detached")"

# A blank next version takes the same default prepare-release.sh
# applied in the agent's job, so the two cuts agree.
defaulted="$(open_pr_fixture)"
expect_status "open-pr: the next version defaults from the base" 0 \
    open_pr "$defaulted" 0.1.0 "" main \
    "$defaulted.changelog.md" "$defaulted.body.md"
expect_eq "open-pr: the default is the minor bump" \
    0.2.0-dev "$(next_trailer "$defaulted")"

botted="$(open_pr_fixture)"
expect_status "open-pr: APP_SLUG commits as the bot" 0 \
    in_fixture "$botted" env GH_TOKEN=t APP_SLUG=vectorsearch-bot \
    STUB_COMMENT_ID=4242 \
    ./scripts/open-release-pr.sh 0.1.0 0.2.0-dev main \
    "$botted.changelog.md" "$botted.body.md"
expect_eq "open-pr: the commit carries the bot identity" \
    "vectorsearch-bot[bot]" "$(git -C "$botted" log -1 --format=%an)"
# The bot resolves here, so the refusal is the token's alone.
tokenless="$(open_pr_fixture)"
expect_status "open-pr: APP_SLUG without a token is refused" 1 \
    in_fixture "$tokenless" env -u GH_TOKEN APP_SLUG=vectorsearch-bot \
    STUB_COMMENT_ID=4242 \
    ./scripts/open-release-pr.sh 0.1.0 0.2.0-dev main \
    "$tokenless.changelog.md" "$tokenless.body.md"
expect_eq "open-pr: the tokenless run leaves the committer alone" \
    "Release Test" "$(git -C "$tokenless" config user.name)"
expect_eq "open-pr: the tokenless run pushes nothing" \
    "" "$(pushed_branches "$tokenless")"

# The stub answers nothing for the bot account here. A commit under a
# half-written identity must not reach the remote.
unresolved="$(open_pr_fixture)"
expect_status "open-pr: an unresolved bot account is refused" 1 \
    in_fixture "$unresolved" env GH_TOKEN=t APP_SLUG=vectorsearch-bot \
    ./scripts/open-release-pr.sh 0.1.0 0.2.0-dev main \
    "$unresolved.changelog.md" "$unresolved.body.md"
expect_eq "open-pr: the unresolved bot pushes nothing" \
    "" "$(pushed_branches "$unresolved")"

# The versions.
dev="$(open_pr_fixture)"
expect_status "open-pr: a development version is not a release" 1 \
    open_pr "$dev" 0.1.0-dev 0.2.0-dev main "$dev.changelog.md" "$dev.body.md"
expect_status "open-pr: a next version without -dev is refused" 1 \
    open_pr "$dev" 0.1.0 0.2.0 main "$dev.changelog.md" "$dev.body.md"

# The checkout.
dirty="$(open_pr_fixture)"
printf x >>"$dirty/meson.build"
expect_status "open-pr: uncommitted changes are refused" 1 \
    open_pr "$dirty" 0.1.0 0.2.0-dev main \
    "$dirty.changelog.md" "$dirty.body.md"

astray="$(open_pr_fixture)"
git -C "$astray" commit -q --allow-empty -m "not on the remote"
expect_status "open-pr: a checkout ahead of the base is refused" 1 \
    open_pr "$astray" 0.1.0 0.2.0-dev main \
    "$astray.changelog.md" "$astray.body.md"

released="$(new_fixture 0.1.0)"
write_changelog "$released.changelog.md" 0.1.0
write_body "$released.body.md" 0.1.0 "Next-Version: 0.2.0-dev"
expect_status "open-pr: a base not on a development version is refused" \
    1 open_pr "$released" 0.1.0 0.2.0-dev main \
    "$released.changelog.md" "$released.body.md"

taken="$(open_pr_fixture)"
git -C "$taken" push -q origin main:refs/heads/chore/release-0.1.0
expect_status "open-pr: a branch already on the remote is refused" 1 \
    open_pr "$taken" 0.1.0 0.2.0-dev main \
    "$taken.changelog.md" "$taken.body.md"
expect_eq "open-pr: it does not open a pull request on it" \
    0 "$(gh_calls "$taken" 'pr create')"

# Recutting: the branch is recreated, the push forces with a lease,
# and the open pull request keeps its number and gets the new body.
expect_status "open-pr: --force recreates a branch on the remote" 0 \
    in_fixture "$taken" env STUB_OPEN_PR=https://example.invalid/pr/3 \
    ./scripts/open-release-pr.sh --force 0.1.0 0.2.0-dev main \
    "$taken.changelog.md" "$taken.body.md"
expect_eq "open-pr: the recut reaches the remote" \
    0.1.0 "$(git -C "$taken.git" show chore/release-0.1.0:VERSION |
        tr -d '[:space:]')"
expect_eq "open-pr: the recut refreshes the open pull request" \
    1 "$(gh_calls "$taken" 'pr edit')"
expect_eq "open-pr: the recut opens no second pull request" \
    0 "$(gh_calls "$taken" 'pr create')"

local_only="$(open_pr_fixture)"
git -C "$local_only" branch chore/release-0.1.0
expect_status "open-pr: a local branch is refused without --force" 1 \
    open_pr "$local_only" 0.1.0 0.2.0-dev main \
    "$local_only.changelog.md" "$local_only.body.md"
expect_status "open-pr: --force recreates a local branch" 0 \
    open_pr "$local_only" --force 0.1.0 0.2.0-dev main \
    "$local_only.changelog.md" "$local_only.body.md"
expect_eq "open-pr: the recreated local branch is pushed" \
    refs/heads/chore/release-0.1.0 "$(pushed_branches "$local_only")"

# The files handed over.
nofile="$(open_pr_fixture)"
expect_status "open-pr: a missing changelog is refused" 1 \
    open_pr "$nofile" 0.1.0 0.2.0-dev main \
    "$nofile.missing.md" "$nofile.body.md"
expect_status "open-pr: a missing body is refused" 1 \
    open_pr "$nofile" 0.1.0 0.2.0-dev main \
    "$nofile.changelog.md" "$nofile.missing.md"

# The one that matters most on the changelog side: no entry means
# the agent handed over the base's changelog, or one for the wrong
# version, and the release check must see it before anything pushes.
noentry="$(open_pr_fixture)"
write_changelog "$noentry.changelog.md" 0.9.9
expect_status "open-pr: a changelog without the entry is refused" 1 \
    open_pr "$noentry" 0.1.0 0.2.0-dev main \
    "$noentry.changelog.md" "$noentry.body.md"
expect_eq "open-pr: the entry refusal pushes nothing" \
    "" "$(pushed_branches "$noentry")"

# An entry with nothing under the heading passes the FILL-IN check, and
# the gate refuses it only after the merge.
blank="$(open_pr_fixture)"
printf '# Changelog\n\n## [0.1.0] - 2026-01-01\n\n\n' >"$blank.changelog.md"
expect_status "open-pr: an empty changelog entry is refused" 1 \
    open_pr "$blank" 0.1.0 0.2.0-dev main \
    "$blank.changelog.md" "$blank.body.md"
expect_eq "open-pr: the empty entry refusal pushes nothing" \
    "" "$(pushed_branches "$blank")"

unfinished="$(open_pr_fixture)"
write_body "$unfinished.body.md" 0.1.0 "Next-Version: 0.2.0-dev" \
    "FILL-IN: summary"
expect_status "open-pr: FILL-IN in the body is refused" 1 \
    open_pr "$unfinished" 0.1.0 0.2.0-dev main \
    "$unfinished.changelog.md" "$unfinished.body.md"

disagree="$(open_pr_fixture)"
write_body "$disagree.body.md" 0.1.0 "Next-Version: 0.3.0-dev"
expect_status "open-pr: a body trailer disagreeing with the run is refused" \
    1 open_pr "$disagree" 0.1.0 0.2.0-dev main \
    "$disagree.changelog.md" "$disagree.body.md"

# A squash merge concatenates two trailers into one unparseable value,
# and matching the first alone would let that through.
twice="$(open_pr_fixture)"
write_body "$twice.body.md" 0.1.0 \
    "Next-Version: 0.2.0-dev
next-version: 0.3.0-dev"
expect_status "open-pr: two Next-Version trailers are refused" 1 \
    open_pr "$twice" 0.1.0 0.2.0-dev main \
    "$twice.changelog.md" "$twice.body.md"

# A --- line is markdown, not a patch divider: the trailer after one
# is read, as the gate reads it off the squashed commit. Text after
# the trailer paragraph is refused, because the gate would then find
# no trailer at all.
ruled="$(open_pr_fixture)"
write_body "$ruled.body.md" 0.1.0 "---

Next-Version: 0.2.0-dev"
expect_status "open-pr: a horizontal rule before the trailer is fine" 0 \
    open_pr "$ruled" 0.1.0 0.2.0-dev main \
    "$ruled.changelog.md" "$ruled.body.md"
footnoted="$(open_pr_fixture)"
write_body "$footnoted.body.md" 0.1.0 "Next-Version: 0.2.0-dev

---

a footnote"
expect_status "open-pr: text after the trailer paragraph is refused" 1 \
    open_pr "$footnoted" 0.1.0 0.2.0-dev main \
    "$footnoted.changelog.md" "$footnoted.body.md"

# Turning a check off is the opener's flag, never a line in the notes.
overridden="$(open_pr_fixture)"
write_body "$overridden.body.md" 0.1.0 "Next-Version: 0.2.0-dev
Disable-check: upgrade-path"
expect_status "open-pr: a Disable-check line in the body is refused" 1 \
    open_pr "$overridden" 0.1.0 0.2.0-dev main \
    "$overridden.changelog.md" "$overridden.body.md"
hidden="$(open_pr_fixture)"
write_body "$hidden.body.md" 0.1.0 "Next-Version: 0.2.0-dev" \
    "<!--
disable-check: upgrade-path
-->"
expect_status "open-pr: a Disable-check line hidden in a comment is refused" \
    1 open_pr "$hidden" 0.1.0 0.2.0-dev main \
    "$hidden.changelog.md" "$hidden.body.md"
# shellcheck disable=SC2016  # $1 is the inner shell's
expect_status "release-lib: check_disabled ignores a comment-hidden line" \
    1 env -C "$hidden" bash -c '
        source scripts/release-lib.sh
        BODY="$(cat "$1")" check_disabled upgrade-path' _ "$hidden.body.md"
# A comment closing on the line uncovers it for the check but not for
# a guard reading the file as written.
uncovered="$(open_pr_fixture)"
write_body "$uncovered.body.md" 0.1.0 "Next-Version: 0.2.0-dev" \
    "<!-- note -->Disable-check: upgrade-path"
expect_status "open-pr: a Disable-check line after a comment is refused" \
    1 open_pr "$uncovered" 0.1.0 0.2.0-dev main \
    "$uncovered.changelog.md" "$uncovered.body.md"

# A body longer than a pipe buffer must not slip past the guard
# through a reader that stops at the first match.
long="$(open_pr_fixture)"
{
    printf '<!-- note -->Disable-check: upgrade-path\nReleases 0.1.0.\n\n'
    yes 'filler' | head -30000
    printf '\nNext-Version: 0.2.0-dev\n'
} >"$long.body.md"
expect_status "open-pr: a long body with an uncovered line is refused" 1 \
    open_pr "$long" 0.1.0 0.2.0-dev main \
    "$long.changelog.md" "$long.body.md"

# The override has to reach the check: with a previous release and no
# upgrade script, the flag is what lets this one open.
flagged="$(open_pr_fixture)"
git -C "$flagged" tag v0.0.9
expect_status "open-pr: a missing upgrade path is refused" 1 \
    open_pr "$flagged" 0.1.0 0.2.0-dev main \
    "$flagged.changelog.md" "$flagged.body.md"
expect_status "open-pr: --disable-check opens with the override" 0 \
    open_pr "$flagged" --disable-check upgrade-path 0.1.0 0.2.0-dev main \
    "$flagged.changelog.md" "$flagged.body.md"
expect_eq "open-pr: the override is a trailer in the pushed body" \
    1 "$(sed -n 's/^gh pr create .*--body-file //p' "$flagged.gh.log" |
        xargs grep -c '^Disable-check: upgrade-path$')"

# A line of spaces after the trailer would put the override in a
# paragraph of its own, and Next-Version outside the one git reads.
padded="$(open_pr_fixture)"
git -C "$padded" tag v0.0.9
printf '   \n' >>"$padded.body.md"
expect_status "open-pr: the override joins a trailer padded with blanks" 0 \
    open_pr "$padded" --disable-check upgrade-path 0.1.0 0.2.0-dev main \
    "$padded.changelog.md" "$padded.body.md"
expect_eq "open-pr: both trailers are in the paragraph git reads" \
    2 "$(sed -n 's/^gh pr create .*--body-file //p' "$padded.gh.log" |
        xargs git interpret-trailers --parse --no-divider |
        grep -ci '^\(next-version\|disable-check\):')"

# The refresh keeps the replaced description at body-previous.md
# beside the body, so the body cannot be that file.
reserved="$(open_pr_fixture)"
mkdir "$reserved.notes"
cp "$reserved.body.md" "$reserved.notes/body-previous.md"
expect_status "open-pr: a body named body-previous.md is refused" 1 \
    open_pr "$reserved" 0.1.0 0.2.0-dev main \
    "$reserved.changelog.md" "$reserved.notes/body-previous.md"
expect_eq "open-pr: that refusal pushes nothing" \
    "" "$(pushed_branches "$reserved")"

# Nor can the changelog draft be there: the refresh would write the
# old description over it.
backedup="$(open_pr_fixture)"
mkdir "$backedup.notes"
cp "$backedup.changelog.md" "$backedup.notes/body-previous.md"
cp "$backedup.body.md" "$backedup.notes/body.md"
expect_status "open-pr: a changelog named body-previous.md is refused" 1 \
    open_pr "$backedup" 0.1.0 0.2.0-dev main \
    "$backedup.notes/body-previous.md" "$backedup.notes/body.md"
expect_eq "open-pr: the backup-name refusal pushes nothing" \
    "" "$(pushed_branches "$backedup")"

untrailered="$(open_pr_fixture)"
printf 'Releases 0.1.0.\n' >"$untrailered.body.md"
expect_status "open-pr: a body without the trailer is refused" 1 \
    open_pr "$untrailered" 0.1.0 0.2.0-dev main \
    "$untrailered.changelog.md" "$untrailered.body.md"
expect_eq "open-pr: the body refusals push nothing" \
    "" "$(pushed_branches "$untrailered")"

# The release check runs before the push, so unfinished notes in the
# changelog stop here rather than on the pull request.
drafty="$(open_pr_fixture)"
printf '# Changelog\n\n## [0.1.0] - 2026-01-01\n\nFILL-IN: highlights\n' \
    >"$drafty.changelog.md"
expect_status "open-pr: FILL-IN in the changelog is refused" 1 \
    open_pr "$drafty" 0.1.0 0.2.0-dev main \
    "$drafty.changelog.md" "$drafty.body.md"
expect_eq "open-pr: the unfinished changelog pushes nothing" \
    "" "$(pushed_branches "$drafty")"

# A refusal after the cut puts the checkout back: the release check
# is what fails here, on a stale version in the docs, which nothing
# before the commit can see.
stale="$(open_pr_fixture)"
printf 'pg_vectorsearch 0.0.9 is current.\n' >"$stale/README.md"
git -C "$stale" add README.md
git -C "$stale" commit -qm "stale docs"
git -C "$stale" push -q origin main
expect_status "open-pr: a failing release check is refused" 1 \
    open_pr "$stale" 0.1.0 0.2.0-dev main \
    "$stale.changelog.md" "$stale.body.md"
expect_eq "open-pr: the refusal pushes nothing" \
    "" "$(pushed_branches "$stale")"
expect_eq "open-pr: the refusal returns the checkout to the base" \
    main "$(git -C "$stale" branch --show-current)"
expect_eq "open-pr: the refusal drops the cut branch" \
    "" "$(git -C "$stale" branch --list chore/release-0.1.0)"
expect_eq "open-pr: the refusal leaves the tree clean" \
    "" "$(git -C "$stale" status --porcelain)"

# A failed commit leaves the cut staged, and a re-run would refuse
# the dirty tree it left. The hook here fails every commit.
uncommittable="$(open_pr_fixture)"
printf '#!/bin/sh\nexit 1\n' >"$uncommittable/.git/hooks/pre-commit"
chmod +x "$uncommittable/.git/hooks/pre-commit"
expect_status "open-pr: a failed release commit is a refusal" 1 \
    open_pr "$uncommittable" 0.1.0 0.2.0-dev main \
    "$uncommittable.changelog.md" "$uncommittable.body.md"
expect_eq "open-pr: a failed commit leaves the tree clean" \
    "" "$(git -C "$uncommittable" status --porcelain)"
expect_eq "open-pr: a failed commit leaves the checkout on the base" \
    main "$(git -C "$uncommittable" branch --show-current)"

# A recut that is refused puts the branch it recreated back where it
# was, rather than dropping someone's work.
kept="$(open_pr_fixture)"
git -C "$kept" branch chore/release-0.1.0
tip="$(git -C "$kept" rev-parse chore/release-0.1.0)"
printf 'pg_vectorsearch 0.0.9 is current.\n' >"$kept/README.md"
git -C "$kept" add README.md
git -C "$kept" commit -qm "stale docs"
git -C "$kept" push -q origin main
expect_status "open-pr: a refused --force recut is a refusal" 1 \
    open_pr "$kept" --force 0.1.0 0.2.0-dev main \
    "$kept.changelog.md" "$kept.body.md"
expect_eq "open-pr: a refused recut restores the old branch tip" \
    "$tip" "$(git -C "$kept" rev-parse chore/release-0.1.0)"
expect_eq "open-pr: a refused recut returns the checkout to the base" \
    main "$(git -C "$kept" branch --show-current)"

# Before the first release there is no CHANGELOG.md to restore, so a
# refusal removes the one the cut wrote.
fresh="$(open_pr_fixture)"
git -C "$fresh" rm -q CHANGELOG.md
printf 'pg_vectorsearch 0.0.9 is current.\n' >"$fresh/README.md"
git -C "$fresh" add README.md
git -C "$fresh" commit -qm "no changelog yet, stale docs"
git -C "$fresh" push -q origin main
expect_status "open-pr: a refusal without a tracked changelog" 1 \
    open_pr "$fresh" 0.1.0 0.2.0-dev main \
    "$fresh.changelog.md" "$fresh.body.md"
expect_eq "open-pr: the refusal removes the changelog the cut wrote" \
    "" "$(git -C "$fresh" status --porcelain)"

# An untracked CHANGELOG.md would be written over and then removed by
# that, so it is refused before the cut and left as it was.
untracked="$(open_pr_fixture)"
git -C "$untracked" rm -q CHANGELOG.md
git -C "$untracked" commit -qm "no changelog yet"
git -C "$untracked" push -q origin main
printf 'mine\n' >"$untracked/CHANGELOG.md"
expect_status "open-pr: an untracked CHANGELOG.md is refused" 1 \
    open_pr "$untracked" 0.1.0 0.2.0-dev main \
    "$untracked.changelog.md" "$untracked.body.md"
expect_eq "open-pr: the untracked CHANGELOG.md is left as it was" \
    mine "$(cat "$untracked/CHANGELOG.md")"
expect_eq "open-pr: the untracked CHANGELOG.md refusal pushes nothing" \
    "" "$(pushed_branches "$untracked")"

# A dangling symlink is untracked too, though nothing is behind it.
dangling="$(open_pr_fixture)"
git -C "$dangling" rm -q CHANGELOG.md
git -C "$dangling" commit -qm "no changelog yet"
git -C "$dangling" push -q origin main
ln -s nowhere "$dangling/CHANGELOG.md"
expect_status "open-pr: a dangling CHANGELOG.md symlink is refused" 1 \
    open_pr "$dangling" 0.1.0 0.2.0-dev main \
    "$dangling.changelog.md" "$dangling.body.md"
expect_eq "open-pr: the dangling symlink is left as it was" \
    nowhere "$(readlink "$dangling/CHANGELOG.md")"

# The draft is a copy, never the checkout's own file.
selfsame="$(open_pr_fixture)"
cp "$selfsame.changelog.md" "$selfsame/CHANGELOG.md"
git -C "$selfsame" add CHANGELOG.md
git -C "$selfsame" commit -qm "notes in the tree"
git -C "$selfsame" push -q origin main
expect_status "open-pr: the checkout's own CHANGELOG.md as the draft is refused" \
    1 open_pr "$selfsame" 0.1.0 0.2.0-dev main \
    "$selfsame/CHANGELOG.md" "$selfsame.body.md"
expect_eq "open-pr: that refusal leaves the tree clean" \
    "" "$(git -C "$selfsame" status --porcelain)"

# Nor is the body: the cut writes CHANGELOG.md over it before the
# pull request is opened from it.
bodyalias="$(open_pr_fixture)"
cp "$bodyalias.body.md" "$bodyalias/CHANGELOG.md"
git -C "$bodyalias" add CHANGELOG.md
git -C "$bodyalias" commit -qm "body in the tree"
git -C "$bodyalias" push -q origin main
expect_status "open-pr: the checkout's own CHANGELOG.md as the body is refused" \
    1 open_pr "$bodyalias" 0.1.0 0.2.0-dev main \
    "$bodyalias.changelog.md" "$bodyalias/CHANGELOG.md"
expect_eq "open-pr: the body refusal pushes nothing" \
    "" "$(pushed_branches "$bodyalias")"

# A hard link to it is the same file under another name.
linked="$(open_pr_fixture)"
ln "$linked/CHANGELOG.md" "$linked.linked.md"
cat "$linked.body.md" >"$linked.linked.md"
git -C "$linked" commit -qam "body in the tree"
git -C "$linked" push -q origin main
expect_status "open-pr: a body hard-linked to CHANGELOG.md is refused" 1 \
    open_pr "$linked" 0.1.0 0.2.0-dev main \
    "$linked.changelog.md" "$linked.linked.md"
expect_eq "open-pr: the link refusal pushes nothing" \
    "" "$(pushed_branches "$linked")"

# A post-checkout hook that fails cannot stop a switch, only report
# one that happened as failed. The cut ignores it, and so does the
# undo, which still has the old tip to put back.
hooked="$(open_pr_fixture)"
printf '#!/bin/sh\nexit 1\n' >"$hooked/.git/hooks/post-checkout"
chmod +x "$hooked/.git/hooks/post-checkout"
expect_status "open-pr: a failing post-checkout hook does not stop the cut" \
    0 open_pr "$hooked" 0.1.0 0.2.0-dev main \
    "$hooked.changelog.md" "$hooked.body.md"
rehooked="$(open_pr_fixture)"
printf '#!/bin/sh\nexit 1\n' >"$rehooked/.git/hooks/post-checkout"
chmod +x "$rehooked/.git/hooks/post-checkout"
git -C "$rehooked" branch chore/release-0.1.0
tip="$(git -C "$rehooked" rev-parse chore/release-0.1.0)"
printf 'pg_vectorsearch 0.0.9 is current.\n' >"$rehooked/README.md"
git -C "$rehooked" add README.md
git -C "$rehooked" commit -qm "stale docs"
git -C "$rehooked" push -q origin main
expect_status "open-pr: a refused recut under a failing hook" 1 \
    open_pr "$rehooked" --force 0.1.0 0.2.0-dev main \
    "$rehooked.changelog.md" "$rehooked.body.md"
expect_eq "open-pr: the undo under a failing hook restores the old tip" \
    "$tip" "$(git -C "$rehooked" rev-parse chore/release-0.1.0)"
expect_eq "open-pr: the undo under a failing hook returns to the base" \
    main "$(git -C "$rehooked" branch --show-current)"

# Standing on the release branch itself leaves nowhere to go back to.
astride="$(open_pr_fixture)"
git -C "$astride" switch -q -c chore/release-0.1.0
expect_status "open-pr: a checkout on the release branch is refused" 1 \
    open_pr "$astride" --force 0.1.0 0.2.0-dev main \
    "$astride.changelog.md" "$astride.body.md"

# ----------------------------------------------------------------
# prepare-release.sh
# ----------------------------------------------------------------
#
# The draft: two files under $TMPDIR, and a checkout left as it was.
# The git-cliff stub writes the entry, so what is checked here is the
# shape around it -- the template, the trailer, the preconditions --
# and that the draft opens a pull request when handed to
# open-release-pr.sh.

prepare() {
    local dir="$1"
    shift
    in_fixture "$dir" env TMPDIR="$dir.tmp" \
        ./scripts/prepare-release.sh "$@" </dev/null
}

# The one draft directory a run left under the fixture's TMPDIR.
draft_dir() {
    local dirs=("$1.tmp"/pg_vectorsearch-release-*)
    [[ "${#dirs[@]}" == 1 && -d "${dirs[0]}" ]] || return 1
    printf '%s\n' "${dirs[0]}"
}

drafted="$(new_fixture 0.1.0-dev)"
mkdir -p "$drafted.tmp"
expect_status "prepare: writes the draft" 0 \
    prepare "$drafted" --version 0.1.0 --next-version 0.2.0-dev \
    --ignore-untracked-files
draft="$(draft_dir "$drafted")"
expect_eq "prepare: the checkout is unchanged" \
    "" "$(git -C "$drafted" status --porcelain)"
expect_eq "prepare: the checkout stays on the base" \
    main "$(git -C "$drafted" branch --show-current)"
expect_eq "prepare: nothing is pushed" "" "$(pushed_branches "$drafted")"
expect_eq "prepare: the changelog draft carries the entry" \
    1 "$(grep -c '^## \[0.1.0\]' "$draft/CHANGELOG.md")"
expect_eq "prepare: the entry is a template to finish" \
    1 "$(grep -c '^FILL-IN: highlights' "$draft/CHANGELOG.md")"
expect_eq "prepare: the body ends with the trailer" \
    "Next-Version: 0.2.0-dev" "$(tail -1 "$draft/body.md")"
expect_eq "prepare: the body has nothing to fill in" \
    0 "$(grep -c 'FILL-IN' "$draft/body.md")"

# The printed command is for a shell, and a branch name can carry
# characters one reads.
odd="$(new_fixture 0.1.0-dev)"
mkdir -p "$odd.tmp"
git -C "$odd" switch -qc 'rel;x'
git -C "$odd" push -q origin 'rel;x'
expect_eq "prepare: the printed command quotes the base branch" \
    1 "$(prepare "$odd" --version 0.1.0 --next-version 0.2.0-dev \
        --ignore-untracked-files 2>/dev/null |
        grep -cF '0.2.0-dev rel\;x ')"

# The default versions, taken without a terminal to ask on.
defaults="$(new_fixture 0.1.0-dev)"
mkdir -p "$defaults.tmp"
expect_status "prepare: versions default from VERSION" 0 \
    prepare "$defaults" --ignore-untracked-files
expect_eq "prepare: the default next cycle is the minor bump" \
    "Next-Version: 0.2.0-dev" \
    "$(tail -1 "$(draft_dir "$defaults")/body.md")"

# The whole path: finish the draft and hand it to the opener.
finished="$draft/CHANGELOG.md"
sed -i.bak -e 's/^FILL-IN: highlights$/- it works/' \
    -e '/^FILL-IN:/d' "$finished" && rm -f "$finished.bak"
expect_status "prepare: the finished draft opens the pull request" 0 \
    open_pr "$drafted" 0.1.0 0.2.0-dev main "$finished" "$draft/body.md"
expect_eq "prepare: the draft's entry reaches the release commit" \
    1 "$(git -C "$drafted" show HEAD:CHANGELOG.md | grep -c '^- it works')"
expect_eq "prepare: the pull request is opened" \
    1 "$(gh_calls "$drafted" 'pr create')"

# Preconditions, each refused before anything is written.
dirty_draft="$(new_fixture 0.1.0-dev)"
mkdir -p "$dirty_draft.tmp"
printf x >>"$dirty_draft/meson.build"
expect_status "prepare: uncommitted changes are refused" 1 \
    prepare "$dirty_draft" --version 0.1.0 --next-version 0.2.0-dev
expect_status "prepare: nothing is drafted for a dirty tree" 1 \
    draft_dir "$dirty_draft"

released_draft="$(new_fixture 0.1.0)"
mkdir -p "$released_draft.tmp"
expect_status "prepare: a released VERSION has nothing to release" 1 \
    prepare "$released_draft" --ignore-untracked-files

tagged="$(new_fixture 0.1.0-dev)"
mkdir -p "$tagged.tmp"
git -C "$tagged" tag v0.1.0
expect_status "prepare: a version already tagged is refused" 1 \
    prepare "$tagged" --version 0.1.0 --ignore-untracked-files

behind="$(new_fixture 0.1.0-dev)"
mkdir -p "$behind.tmp"
git -C "$behind" commit -q --allow-empty -m "not on the remote"
expect_status "prepare: a base out of sync with the remote is refused" 1 \
    prepare "$behind" --version 0.1.0 --ignore-untracked-files

expect_status "prepare: a next version without -dev is refused" 1 \
    prepare "$(new_fixture 0.1.0-dev)" --version 0.1.0 \
    --next-version 0.2.0 --ignore-untracked-files

# A draft that fails half-way is removed, including when the failure
# is inside the function writing it.
halfway="$(new_fixture 0.1.0-dev)"
mkdir -p "$halfway.tmp"
STUB_CLIFF_FAIL=1 expect_status "prepare: a failing git-cliff is a refusal" \
    1 prepare "$halfway" --version 0.1.0 --ignore-untracked-files
expect_eq "prepare: the half-written draft is removed" \
    "" "$(ls -A "$halfway.tmp")"

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

# An entry with nothing under the heading is refused as unfinished
# notes, as the gate refuses it.
empty="$(new_fixture 0.1.0)"
printf '# Changelog\n\n## [0.1.0] - 2026-01-01\n\n' >"$empty/CHANGELOG.md"
git -C "$empty" add CHANGELOG.md >/dev/null 2>&1
git -C "$empty" commit -q -m "release notes" -m "Next-Version: 0.2.0-dev" \
    >/dev/null 2>&1
expect_status "release-check: an empty changelog entry is refused" 1 \
    in_fixture "$empty" env GITHUB_OUTPUT="$empty.output" \
    ./scripts/ci/release-check.sh

# On a pull_request event actions/checkout hands the job a synthetic
# merge of the release branch into its base, not the release commit:
# refs/pull/N/merge, whose message is "Merge <head> into <base>" and
# carries no trailers. Every assertion reading commit metadata has to
# name the release commit instead, and only this mode exercises that --
# the checkouts above are standalone, where HEAD is the commit itself.
# $2, when given, adds a second commit to the branch.
pr_mode_fixture() {
    local version="$1" extra="${2:-}" dir
    dir="$(new_fixture 0.1.0-dev)"
    (
        cd "$dir" || exit 1
        git switch -qc "chore/release-$version"
        printf '%s\n' "$version" >VERSION
        printf '# Changelog\n\n## [%s] - 2026-01-01\n\n- it works\n' \
            "$version" >CHANGELOG.md
        git add VERSION CHANGELOG.md
        git commit -qm "chore: release $version" \
            -m "Next-Version: 0.2.0-dev"
        if [[ -n "$extra" ]]; then
            printf -- '- and again\n' >>CHANGELOG.md
            git add CHANGELOG.md
            git commit -qm "chore: more notes"
        fi
        # Detached at the base, as the job's checkout is.
        git switch -q --detach main
        git merge -q --no-ff -m "Merge the release into main" \
            "chore/release-$version"
    ) >/dev/null 2>&1
    printf '%s\n' "$dir"
}

pr_dir="$(pr_mode_fixture 0.1.0-rc1)"
pr_out="$pr_dir.output"
: >"$pr_out"
expect_status "release-check: a release pull request passes" 0 \
    in_fixture "$pr_dir" env GITHUB_OUTPUT="$pr_out" \
    ./scripts/ci/release-check.sh origin/main
expect_eq "release-check: a release pull request reports release=true" \
    "release=true" "$(grep '^release=' "$pr_out")"
expect_eq "release-check: a release pull request reports the version" \
    "version=0.1.0-rc1" "$(grep '^version=' "$pr_out")"

# The trailer is only parsed as the message's last paragraph, so a
# second commit would strand it mid-message under a squash merge that
# concatenates the two. Rejecting it is what keeps the one-commit shape
# the trailer read depends on -- and the merge the event adds must not
# be what trips it.
two_dir="$(pr_mode_fixture 0.1.0-rc1 extra)"
expect_status "release-check: a two-commit release pull request is refused" \
    1 in_fixture "$two_dir" env GITHUB_OUTPUT=/dev/null \
    ./scripts/ci/release-check.sh origin/main

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

# ----------------------------------------------------------------
# release-publish.sh
# ----------------------------------------------------------------
#
# What these are really about is the one ordering that cannot be
# repaired. Publishing freezes a release's assets, and the tag is burned
# even if the empty release is deleted, so a publish that happens before
# every file is attached loses the version. The draft is what makes that
# recoverable, and nothing may seal it early.

# A fixture with a dist directory holding what the build job packages.
with_dist() {
    local dir
    dir="$(new_fixture "$1")"
    mkdir -p "$dir/dist"
    printf 'tarball\n' >"$dir/dist/pg_vectorsearch-$1.tar.gz"
    printf 'checksum\n' >"$dir/dist/pg_vectorsearch-$1.tar.gz.sha256sum"
    printf '%s\n' "$dir"
}

publish() {
    local dir="$1"
    shift
    in_fixture "$dir" env GH_TOKEN=x "$@" \
        ./scripts/ci/release-publish.sh 0.1.0 dist
}

# The same against a draft, with the script's own flags after the
# positionals rather than the stub's environment before them.
publish_opts() {
    local dir="$1"
    shift
    in_fixture "$dir" env GH_TOKEN=x STUB_DRAFT=true \
        ./scripts/ci/release-publish.sh 0.1.0 dist "$@"
}

draft="$(with_dist 0.1.0)"
expect_status "publish: a draft is filled and then published" 0 \
    publish "$draft" STUB_DRAFT=true
expect_eq "publish: the assets are uploaded" \
    1 "$(gh_calls "$draft" 'release upload')"
expect_eq "publish: the draft is published once they are there" \
    1 "$(gh_calls "$draft" 'release edit')"

# The upload is read back rather than trusted: a file that silently did
# not arrive must stop the publish, because afterwards it cannot be
# added.
lost="$(with_dist 0.1.0)"
expect_status "publish: an asset that did not arrive is refused" 1 \
    publish "$lost" STUB_DRAFT=true STUB_UPLOAD_LOSES=1
expect_eq "publish: nothing is published when one is missing" \
    0 "$(gh_calls "$lost" 'release edit')"

# Re-running after a failure downstream of the publish must succeed
# rather than report a release that is already correct as broken.
done_="$(with_dist 0.1.0)"
expect_status "publish: an already published release is accepted" 0 \
    publish "$done_" STUB_DRAFT=false \
    "STUB_ASSETS=pg_vectorsearch-0.1.0.tar.gz pg_vectorsearch-0.1.0.tar.gz.sha256sum"
expect_eq "publish: it is not published a second time" \
    0 "$(gh_calls "$done_" 'release edit')"
expect_eq "publish: nothing is uploaded to it" \
    0 "$(gh_calls "$done_" 'release upload')"

# The unrecoverable state, which must be reported as the failure it is
# rather than quietly passing.
partial="$(with_dist 0.1.0)"
expect_status "publish: a published release missing a file fails" 1 \
    publish "$partial" STUB_DRAFT=false \
    STUB_ASSETS=pg_vectorsearch-0.1.0.tar.gz

absent="$(with_dist 0.1.0)"
expect_status "publish: no release entry at all is refused" 1 \
    publish "$absent" STUB_DRAFT=

mismatch="$(with_dist 0.1.0)"
expect_status "publish: a version VERSION disagrees with is refused" 1 \
    in_fixture "$mismatch" env GH_TOKEN=x STUB_DRAFT=true \
    ./scripts/ci/release-publish.sh 0.2.0 dist

# --keep-draft holds back the publish and nothing else, so the files
# still have to be there and still have to be checked -- the hold is a
# decision about visibility, not a way to skip the verification.
held="$(with_dist 0.1.0)"
expect_status "publish: --keep-draft attaches the assets" 0 \
    publish_opts "$held" --keep-draft
expect_eq "publish: --keep-draft uploads them" \
    1 "$(gh_calls "$held" 'release upload')"
expect_eq "publish: --keep-draft leaves it unpublished" \
    0 "$(gh_calls "$held" 'release edit')"

held_lost="$(with_dist 0.1.0)"
expect_status "publish: --keep-draft still refuses a lost asset" 1 \
    in_fixture "$held_lost" env GH_TOKEN=x STUB_DRAFT=true \
    STUB_UPLOAD_LOSES=1 \
    ./scripts/ci/release-publish.sh 0.1.0 dist --keep-draft

bad_opt="$(with_dist 0.1.0)"
expect_status "publish: an unknown option is refused" 1 \
    in_fixture "$bad_opt" env GH_TOKEN=x STUB_DRAFT=true \
    ./scripts/ci/release-publish.sh 0.1.0 dist --publish-anyway

empty="$(new_fixture 0.1.0)"
mkdir -p "$empty/dist"
expect_status "publish: an empty dist directory is refused" 1 \
    publish "$empty" STUB_DRAFT=true
expect_eq "publish: an empty dist publishes nothing" \
    0 "$(gh_calls "$empty" 'release edit')"

# ----------------------------------------------------------------
# release-create.sh
# ----------------------------------------------------------------

# Reaching the draft decision needs the tag already placed at the
# tested commit, which is what the resumable path looks like.
# STUB_COMMENT_ID is what a `gh api` read answers -- here, that sha.
create_at_head() {
    local dir="$1" sha
    shift
    sha="$(git -C "$dir" rev-parse HEAD)"
    in_fixture "$dir" env GH_TOKEN=x STUB_COMMENT_ID="$sha" "$@" \
        ./scripts/ci/release-create.sh 0.1.0 "$sha"
}

# Published is not the same as finished: one that has its tarball and
# checksum is this step's work already done, so the run carries on.
finished="$(new_fixture 0.1.0)"
expect_status "create: a published release with its assets is done" 0 \
    create_at_head "$finished" STUB_DRAFT=false \
    STUB_ASSETS="pg_vectorsearch-0.1.0.tar.gz \
pg_vectorsearch-0.1.0.tar.gz.sha256sum"
expect_eq "create: a finished release is not created again" \
    0 "$(gh_calls "$finished" 'release create')"

# Published without them is the rc1 failure, which no re-run repairs.
stranded="$(new_fixture 0.1.0)"
expect_status "create: a published release missing its assets is fatal" \
    1 create_at_head "$stranded" STUB_DRAFT=false STUB_ASSETS=

# Where the draft decision comes from when the caller passes no flag:
# the dispatched run's own choice, then the release commit's trailer,
# then the repository variable.

# A fixture whose release commit carries Keep-Draft: $2.
with_trailer() {
    local dir="$1"
    git -C "$dir" -c user.email=t@e -c user.name=T commit -q --amend \
        --allow-empty -m "chore: release 0.1.0" -m "Keep-Draft: $2"
    printf '%s\n' "$dir"
}

by_trailer="$(with_trailer "$(with_dist 0.1.0)" true)"
expect_status "publish: a Keep-Draft trailer is honoured" 0 \
    publish "$by_trailer" STUB_DRAFT=true
expect_eq "publish: the trailer leaves it unpublished" \
    0 "$(gh_calls "$by_trailer" 'release edit')"

by_var="$(with_dist 0.1.0)"
expect_status "publish: RELEASE_KEEP_DRAFT is honoured" 0 \
    publish "$by_var" STUB_DRAFT=true KEEP_DRAFT_DEFAULT=true
expect_eq "publish: the variable leaves it unpublished" \
    0 "$(gh_calls "$by_var" 'release edit')"

# The operator who started the run outranks both of the standing
# sources, in either direction.
override="$(with_trailer "$(with_dist 0.1.0)" true)"
expect_status "publish: a dispatched no overrides the trailer" 0 \
    publish "$override" STUB_DRAFT=true KEEP_DRAFT_INPUT=no \
    KEEP_DRAFT_DEFAULT=true
expect_eq "publish: the dispatched no publishes" \
    1 "$(gh_calls "$override" 'release edit')"

# A trailer that says no beats a variable that says yes: the release
# in hand outranks the standing policy.
beats_var="$(with_trailer "$(with_dist 0.1.0)" false)"
expect_status "publish: a trailer overrides the variable" 0 \
    publish "$beats_var" STUB_DRAFT=true KEEP_DRAFT_DEFAULT=true
expect_eq "publish: the trailer publishes" \
    1 "$(gh_calls "$beats_var" 'release edit')"

tap_finish
