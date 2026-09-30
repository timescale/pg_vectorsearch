#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# release_fixtures.sh - Shared fixtures for the release script tests
#
# Building a repository to run a release script against takes the same
# handful of steps in every test file, so they live here. A fixture is a
# local bare repository standing in for the remote with a clone pointing
# at it, and `gh` is a stub earlier on PATH that records its invocations
# in $GH_LOG and answers from the environment:
#
#   STUB_OPEN_PR     what `gh pr list` reports, empty for none
#   STUB_PR_URL      what `gh pr create` echoes
#   STUB_MERGE_RC    what `gh pr merge` exits with
#   STUB_COMMENT_ID  what a `gh api` read reports, empty for none
#
# So the git work stays real -- branches, file writes, commits, pushes
# -- while the GitHub calls become observable. What that cannot cover
# is GitHub's own behaviour, which needs a repository with a ruleset.
#
# Source vs_test.sh first: fixtures build inside its $VS_WORKSPACE.

[[ "${BASH_SOURCE[0]}" != "${0}" ]] ||
    { echo "release_fixtures.sh is sourced, not run" >&2; exit 2; }

[[ -n "${VS_WORKSPACE:-}" ]] ||
    { echo "source vs_test.sh before release_fixtures.sh" >&2; exit 2; }

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
        # Where the release checks look for install and upgrade scripts.
        mkdir -p sql src/pg
        printf 'ext_update_scripts = []\n' >src/pg/meson.build
        printf -- '-- canonical install script\n' >sql/pg_vectorsearch.sql
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
# A read of the API answers, a write records and says nothing. Matched
# before the pairwise cases below, since `gh api` takes its flags in
# whatever order the caller wrote them.
if [[ "$1" == api ]]; then
    case "$*" in
        *--method*) exit 0 ;;
        *) printf '%s\n' "${STUB_COMMENT_ID:-}"; exit 0 ;;
    esac
fi
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
