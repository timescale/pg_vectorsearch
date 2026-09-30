#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# vs_test.sh - Simple TAP test framework for the shell script tests
#
# Basic usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/vs_test.sh"
#
#   expect_status "a bad version is refused" 1 ./script.sh 0.1
#   expect_eq "the file was written" 0.2.0-dev "$(cat VERSION)"
#   tap_finish
#
# Cases record into a plan that tap_finish prints, so a failure reports
# what it expected rather than only that it happened. $VS_WORKSPACE is
# a temporary directory removed on exit; fixtures build inside it.

[[ "${BASH_SOURCE[0]}" != "${0}" ]] ||
    { echo "vs_test.sh is sourced, not run" >&2; exit 2; }

# GIT_DIR and GIT_INDEX_FILE take precedence over `git -C <dir>`, so
# while they are set, a test that builds its own repository reads and
# writes the repository they point at instead. git sets them for hooks
# and for `git rebase --exec`, and the damage is silent: `git add -A`
# inside a fixture stages every file of that other repository as
# deleted.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY
unset GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_COMMON_DIR GIT_NAMESPACE

# A temporary directory for whatever the cases build, removed on exit.
VS_WORKSPACE="$(mktemp -d)"
trap 'rm -rf "$VS_WORKSPACE"' EXIT

VS_TESTS=0
VS_FAILED=0
VS_PLAN=()

ok() {
    VS_TESTS=$((VS_TESTS + 1))
    VS_PLAN+=("ok $VS_TESTS - $1")
}

nok() {
    VS_TESTS=$((VS_TESTS + 1))
    VS_FAILED=$((VS_FAILED + 1))
    VS_PLAN+=("not ok $VS_TESTS - $1")
    [[ -z "${2:-}" ]] || VS_PLAN+=("# $2")
}

# $1 description, $2 expected exit status, $3.. the command.
expect_status() {
    local desc="$1" want="$2" got out
    shift 2
    out="$("$@" 2>&1)"
    got=$?
    if [[ "$got" -eq "$want" ]]; then
        ok "$desc"
    else
        nok "$desc" "expected exit $want, got $got: $(head -1 <<<"$out")"
    fi
}

# $1 description, $2 expected value, $3 actual value.
expect_eq() {
    if [[ "$2" == "$3" ]]; then
        ok "$1"
    else
        nok "$1" "expected '$2', got '$3'"
    fi
}

# The TAP plan, then a status reflecting the failures. Call last.
tap_finish() {
    printf '1..%d\n' "$VS_TESTS"
    [[ "$VS_TESTS" -eq 0 ]] || printf '%s\n' "${VS_PLAN[@]}"
    [[ "$VS_FAILED" -eq 0 ]]
}
