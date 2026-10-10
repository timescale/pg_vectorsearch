#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# TAP tests for scripts/ci/release-open-dev-cycle.sh,
# scripts/ci/release-dev-bump-check.sh, scripts/ci/release-window.sh,
# scripts/ci/release-check.sh's classification,
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
expect_status "open-cycle: no arguments at all" 1 open_cycle "$dir"
# The fixture's commit is not one prepare-release.sh wrote, so there is
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
expect_status "open-cycle: APP_SLUG without a token is refused" 1 \
    in_fixture "$botted" env APP_SLUG=vectorsearch-bot \
    ./scripts/ci/release-open-dev-cycle.sh 0.1.0 0.3.0-dev main

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

# ----------------------------------------------------------------
# release-window.sh
# ----------------------------------------------------------------

# The fixture's only commit is the base, so HEAD names it. The script
# refuses to guess this, which is the point of passing it.
window_case() {
    local desc="$1" base_version="$2" head_ref="$3" want="$4"
    local dir
    dir="$(new_fixture "$base_version")"
    expect_status "window: $desc" "$want" env -C "$dir" \
        HEAD_REF="$head_ref" ./scripts/ci/release-window.sh HEAD
}

window_case "a base in the development cycle is open" \
    0.1.0-dev feat/whatever 0
window_case "a base on a released version is closed" \
    0.1.0 feat/whatever 1
window_case "the dev-cycle branch is let through" \
    0.1.0 chore/dev-0.2.0-dev 0
window_case "a candidate counts as released" \
    0.1.0-rc1 feat/whatever 1
window_case "the dev-cycle branch is unremarkable outside the window" \
    0.1.0-dev chore/dev-0.2.0-dev 0

# A branch merely starting with the prefix is not a free pass on its
# own: release-dev-bump-check.sh is what constrains its contents, and
# both run on every pull request.
window_case "a branch named like the bump still needs that check" \
    0.1.0 chore/dev-anything 0

dir="$(new_fixture 0.1.0)"
expect_status "window: the base is required" 1 env -C "$dir" \
    HEAD_REF=feat/whatever ./scripts/ci/release-window.sh
expect_status "window: an unreadable base is refused" 1 env -C "$dir" \
    HEAD_REF=feat/whatever ./scripts/ci/release-window.sh nope

tap_finish
