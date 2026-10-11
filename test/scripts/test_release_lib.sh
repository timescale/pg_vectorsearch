#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# TAP tests for the release helpers in scripts/release-lib.sh.
#
# Usage: ./test/scripts/test_release_lib.sh
#
# The helpers are exercised directly rather than through
# release-check.sh: what needs proving is the rule each one encodes,
# and reaching them through the check would mean satisfying every
# other assertion first.

set -uo pipefail

# shellcheck source=test/scripts/vs_test.sh
source "$(dirname "${BASH_SOURCE[0]}")/vs_test.sh"
# shellcheck source=test/scripts/release_fixtures.sh
source "$(dirname "${BASH_SOURCE[0]}")/release_fixtures.sh"

# Run a release-lib function inside a fixture, in a subshell so a die()
# cannot take the test run down with it.
lib_call() {
    local dir="$1"
    shift
    ( cd "$dir" && source scripts/release-lib.sh && "$@" ) 2>&1
}

# $1 description, $2 expected status, $3 fixture, $4.. the call
expect_lib() {
    local desc="$1" want="$2" dir="$3"
    shift 3
    local out got
    out="$(lib_call "$dir" "$@")"
    got=$?
    if [[ "$got" -eq "$want" ]]; then
        ok "$desc"
    else
        nok "$desc" "expected exit $want, got $got: $(head -1 <<<"$out")"
    fi
}

add_upgrade_script() {
    local dir="$1" from="$2" to="$3"
    printf -- '-- upgrade %s -> %s\n' "$from" "$to" \
        >"$dir/sql/pg_vectorsearch--$from--$to.sql"
}

list_upgrade_script() {
    local dir="$1" from="$2" to="$3"
    printf "ext_update_scripts = ['sql/pg_vectorsearch--%s--%s.sql']\n" \
        "$from" "$to" >"$dir/src/pg/meson.build"
}

# ----------------------------------------------------------------
# check_upgrade_path
# ----------------------------------------------------------------

# The first release has nothing to upgrade from, so the rule cannot
# apply -- and 0.1.0 must not be blocked by it.
first="$(new_fixture 0.1.0)"
expect_lib "upgrade: a first release needs no upgrade script" 0 \
    "$first" check_upgrade_path 0.1.0

# With a previous release, an installation on it has to be able to
# reach the new version.
second="$(new_fixture 0.2.0)"
git -C "$second" tag v0.1.0 "$(git -C "$second" rev-parse HEAD)"
expect_lib "upgrade: a missing script is refused" 1 \
    "$second" check_upgrade_path 0.2.0

add_upgrade_script "$second" 0.1.0 0.2.0
expect_lib "upgrade: a script that is not listed is refused" 1 \
    "$second" check_upgrade_path 0.2.0

list_upgrade_script "$second" 0.1.0 0.2.0
expect_lib "upgrade: a script that exists and is listed passes" 0 \
    "$second" check_upgrade_path 0.2.0

# The script has to name this release, not merely some release.
wrong="$(new_fixture 0.3.0)"
git -C "$wrong" tag v0.1.0 "$(git -C "$wrong" rev-parse HEAD)"
add_upgrade_script "$wrong" 0.1.0 0.2.0
list_upgrade_script "$wrong" 0.1.0 0.2.0
expect_lib "upgrade: a script for another version does not count" 1 \
    "$wrong" check_upgrade_path 0.3.0

# A patch release upgrades from the minor it was branched from.
patch="$(new_fixture 0.1.1)"
git -C "$patch" tag v0.1.0 "$(git -C "$patch" rev-parse HEAD)"
add_upgrade_script "$patch" 0.1.0 0.1.1
list_upgrade_script "$patch" 0.1.0 0.1.1
expect_lib "upgrade: a patch release upgrades from its minor" 0 \
    "$patch" check_upgrade_path 0.1.1

# A candidate is not a version anything upgrades to, so none is asked
# for on the way in or on the way out: 0.1.0 -> 0.2.0-rc1 -> 0.2.0
# needs one script, from 0.1.0 to 0.2.0.
rc="$(new_fixture 0.2.0-rc1)"
git -C "$rc" tag v0.1.0 "$(git -C "$rc" rev-parse HEAD)"
expect_lib "upgrade: a candidate needs no script of its own" 0 \
    "$rc" check_upgrade_path 0.2.0-rc1

final="$(new_fixture 0.2.0)"
git -C "$final" tag v0.1.0 "$(git -C "$final" rev-parse HEAD)"
git -C "$final" tag v0.2.0-rc1 "$(git -C "$final" rev-parse HEAD)"
expect_lib "upgrade: a final release steps over its own candidates" 1 \
    "$final" check_upgrade_path 0.2.0

add_upgrade_script "$final" 0.1.0 0.2.0
list_upgrade_script "$final" 0.1.0 0.2.0
expect_lib "upgrade: the step is measured between final releases" 0 \
    "$final" check_upgrade_path 0.2.0

# The first release line is all candidates, so there is still nothing
# to upgrade from.
rc_first="$(new_fixture 0.1.0)"
git -C "$rc_first" tag v0.1.0-rc1 "$(git -C "$rc_first" rev-parse HEAD)"
expect_lib "upgrade: candidates alone are not a previous release" 0 \
    "$rc_first" check_upgrade_path 0.1.0

# A candidate may still carry an upgrade script, and an unlisted one
# would be built and never installed, so that stays checked.
rc_script="$(new_fixture 0.2.0-rc1)"
git -C "$rc_script" tag v0.1.0 "$(git -C "$rc_script" rev-parse HEAD)"
add_upgrade_script "$rc_script" 0.1.0 0.2.0-rc1
expect_lib "upgrade: a candidate's unlisted script is refused" 1 \
    "$rc_script" check_upgrade_path 0.2.0-rc1

list_upgrade_script "$rc_script" 0.1.0 0.2.0-rc1
expect_lib "upgrade: a candidate may carry a listed script" 0 \
    "$rc_script" check_upgrade_path 0.2.0-rc1

# A release that deliberately ships without a path says so in the pull
# request description.
disable_case() {
    local desc="$1" body="$2" want="$3"
    local out got
    out="$(BODY="$body" lib_call "$second" check_upgrade_path 0.3.0)"
    got=$?
    if [[ "$got" -eq "$want" ]]; then
        ok "$desc"
    else
        nok "$desc" "expected exit $want, got $got: $(head -1 <<<"$out")"
    fi
}

disable_case "upgrade: no trailer leaves the check on" "" 1
disable_case "upgrade: the trailer disables it" \
    "Some prose.

Disable-check: upgrade-path" 0
disable_case "upgrade: the trailer is case-insensitive" \
    "disable-check: Upgrade-Path" 0
disable_case "upgrade: one trailer can name several checks" \
    "Disable-check: commit-count, upgrade-path" 0
disable_case "upgrade: another check's trailer does not disable it" \
    "Disable-check: commit-count" 1
disable_case "upgrade: the trailer has to start the line" \
    "see Disable-check: upgrade-path above" 1

# ----------------------------------------------------------------
# next_dev_version
# ----------------------------------------------------------------

next_dev_case() {
    local desc="$1" released="$2" branch="$3" want="$4"
    expect_eq "next_dev_version: $desc" "$want" \
        "$(lib_call "$first" next_dev_version "$released" "$branch")"
}

next_dev_case "a minor bump on main" 0.1.0 main 0.2.0-dev
next_dev_case "a patch bump on a maintenance branch" 0.1.0 0.1.x 0.1.1-dev
next_dev_case "a later patch on a maintenance branch" 0.1.1 0.1.x 0.1.2-dev

# A candidate is a step toward its own version, not past it, so the
# cycle reopens at that version rather than skipping it.
next_dev_case "a release candidate reopens at its own version" \
    0.2.0-rc1 main 0.2.0-dev
next_dev_case "a candidate on a maintenance branch does the same" \
    0.2.0-rc1 0.2.x 0.2.0-dev
next_dev_case "an alpha behaves like a candidate" \
    0.3.0-alpha1 main 0.3.0-dev

# A malformed argument has to fail rather than produce a malformed
# answer. Before validation, 1.2 became 1.3.0-dev and 1.2.3.4 became
# 1.3.0-dev, inventing and dropping a component respectively.
bad_version_case() {
    expect_status "next_dev_version: $1 is refused" 1 \
        env -C "$first" ./scripts/ci/next-dev-probe.sh "$2" main
}

# shellcheck disable=SC2016  # $1/$2 must reach the probe unexpanded
printf '%s\n' '#!/bin/bash
set -uo pipefail
source scripts/release-lib.sh
next_dev_version "$1" "$2"' > "$first/scripts/ci/next-dev-probe.sh"
chmod +x "$first/scripts/ci/next-dev-probe.sh"

bad_version_case "a non-version"          not-a-version
bad_version_case "a two-component version" 1.2
bad_version_case "a four-component version" 1.2.3.4
bad_version_case "an empty string"        ""
bad_version_case "an uppercase suffix"    0.2.0-DEV
bad_version_case "a doubled dash"         0.2.0--dev

# ----------------------------------------------------------------
# check_next_version_order
# ----------------------------------------------------------------

order_case() {
    expect_lib "order: $1" "$2" "$first" \
        check_next_version_order "$3" "$4"
}

order_case "a minor bump reopens ahead of the release" 0 0.2.0 0.3.0-dev
order_case "reopening behind what shipped is refused" 1 0.2.0 0.1.0-dev

# The one case where the cycle reopens at the version just released.
order_case "a candidate reopens at its own version" 0 0.2.0-rc1 0.2.0-dev
order_case "an alpha does the same" 0 0.2.0-alpha1 0.2.0-dev
order_case "a final release may not reopen at its own" 1 0.2.0 0.2.0-dev

# -dev is the state the branch sits in between releases, never something
# that was released, so the candidate exception must not reach it.
# next_dev_version excludes it for the same reason, and a shared rule
# that disagrees with the value generated from it is a trap.
order_case "a -dev version is not a candidate" 1 0.2.0-dev 0.2.0-dev

# ----------------------------------------------------------------
# check_next_version_trailer
# ----------------------------------------------------------------

# Rewrites the fixture's tip message so the trailer is what $2 says, or
# removes it when $2 is empty.
set_trailer() {
    local dir="$1" next="${2:-}" msg
    msg="chore: release ${3:-0.2.0}"
    if [[ -n "$next" ]]; then
        msg="$msg"$'\n\n'"Next-Version: $next"
    fi
    git -C "$dir" -c user.email=t@e -c user.name=T \
        commit -q --amend -m "$msg" --allow-empty
}

# A tiny probe, because the helper is a shell function rather than a
# script: sourcing release-lib.sh and calling it is what CI does too.
# shellcheck disable=SC2016  # $1/$2 must reach the probes unexpanded
probe_body='#!/bin/bash
set -uo pipefail
source scripts/release-lib.sh
check_next_version_trailer "$1" "$2" HEAD'

make_probe() { printf '%s\n' "$probe_body" > "$1/scripts/ci/release-check-trailer-probe.sh"; chmod +x "$1/scripts/ci/release-check-trailer-probe.sh"; }

trailer_probe() {
    local desc="$1" released="$2" next="$3" branch="$4" want="$5"
    local dir
    dir="$(new_fixture "$released")"
    make_probe "$dir"
    set_trailer "$dir" "$next" "$released"
    expect_status "trailer: $desc" "$want" \
        env -C "$dir" ./scripts/ci/release-check-trailer-probe.sh \
        "$released" "$branch"
}

trailer_probe "the default bump passes" 0.2.0 0.3.0-dev main 0
trailer_probe "a missing trailer is refused" 0.2.0 "" main 1
trailer_probe "a value without -dev is refused" 0.2.0 0.3.0 main 1
trailer_probe "an invalid version is refused" 0.2.0 not-a-version main 1
trailer_probe "reopening at the released version is refused" \
    0.2.0 0.2.0-dev main 1
trailer_probe "reopening behind the release is refused" \
    0.2.0 0.1.0-dev main 1
# A candidate is the one case where an equal base is correct.
trailer_probe "a candidate may reopen at its own version" \
    0.2.0-rc1 0.2.0-dev main 0
# A major bump differs from the default, so it warns rather than fails.
trailer_probe "a major bump is allowed with a warning" \
    0.2.0 1.0.0-dev main 0

# A malformed trailer is the likeliest way a hand edit goes wrong, and
# it reaches validate_version from the other side.
trailer_probe "a two-component trailer is refused" 0.2.0 0.3 main 1
trailer_probe "a four-component trailer is refused" 0.2.0 0.3.0.1 main 1
trailer_probe "an uppercase suffix is refused" 0.2.0 0.3.0-DEV main 1
trailer_probe "a bare suffix is refused" 0.2.0 -dev main 1


# ----------------------------------------------------------------
# previous_release_tag
# ----------------------------------------------------------------

# A final release outranks its own candidates. Git's version sort puts
# v0.1.0-rc1 above v0.1.0 unless told which suffixes are prereleases,
# which would make the next release's upgrade script and format check
# anchor on the candidate instead of the release that replaced it.
tagged="$(new_fixture 0.2.0-dev)"
git -C "$tagged" tag v0.1.0-rc1
git -C "$tagged" tag v0.1.0-rc2
git -C "$tagged" tag v0.1.0
expect_eq "previous tag: a final release outranks its candidates" \
    v0.1.0 "$(lib_call "$tagged" previous_release_tag)"

git -C "$tagged" tag v0.2.0
expect_eq "previous tag: the newest release wins" \
    v0.2.0 "$(lib_call "$tagged" previous_release_tag)"

# Before any final ships, the newest candidate is what came before.
rc_only="$(new_fixture 0.1.0-dev)"
git -C "$rc_only" tag v0.1.0-rc1
git -C "$rc_only" tag v0.1.0-rc2
expect_eq "previous tag: the newest candidate when no final exists" \
    v0.1.0-rc2 "$(lib_call "$rc_only" previous_release_tag)"

expect_eq "previous tag: nothing before the first release" \
    "" "$(lib_call "$(new_fixture 0.1.0-dev)" previous_release_tag)"

# ----------------------------------------------------------------
# previous_final_release_tag
# ----------------------------------------------------------------

# The candidates for the release being cut sit above the release it
# upgrades from, which is the one the upgrade script has to start at.
mixed="$(new_fixture 0.2.0-dev)"
git -C "$mixed" tag v0.1.0
git -C "$mixed" tag v0.2.0-rc1
git -C "$mixed" tag v0.2.0-rc2
expect_eq "previous final: candidates above a release are skipped" \
    v0.1.0 "$(lib_call "$mixed" previous_final_release_tag)"

expect_eq "previous final: nothing when only candidates are tagged" \
    "" "$(lib_call "$rc_only" previous_final_release_tag)"

tap_finish
