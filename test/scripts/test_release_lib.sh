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

tap_finish
