#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# TAP tests for scripts/upgrade-script-tool.sh.
#
# Usage: ./test/scripts/test_tool.sh
#
# Each case runs the command in a fixture repository and looks at what
# it left in sql/ and in the ext_update_scripts list.

set -uo pipefail

# shellcheck source=test/scripts/vs_test.sh
source "$(dirname "${BASH_SOURCE[0]}")/vs_test.sh"
# shellcheck source=test/scripts/release_fixtures.sh
source "$(dirname "${BASH_SOURCE[0]}")/release_fixtures.sh"

LIST=src/pg/meson.build

# Give a fixture an ext_update_scripts list holding the scripts named,
# in the form the build uses.
write_list() {
    local dir="$1" script
    shift
    {
        echo "ext_update_scripts = files("
        for script in "$@"; do
            echo "  '../../sql/$script',"
        done
        echo ")"
        echo "install_data(ext_update_scripts)"
    } >"$dir/$LIST"
}

# A fixture at version $1 with an empty list and the tags that follow.
fixture() {
    local dir version="$1" tag
    shift
    dir="$(new_fixture "$version")"
    write_list "$dir"
    for tag in "$@"; do
        git -C "$dir" tag "$tag"
    done
    echo "$dir"
}

tool() {
    local dir="$1"
    shift
    in_fixture "$dir" ./scripts/upgrade-script-tool.sh "$@"
}

# The sql/ files and list entries of a fixture, one per line.
scripts_in() {
    (cd "$1/sql" && ls pg_vectorsearch--*.sql 2>/dev/null)
}
listed_in() {
    sed -n "s|^  '../../sql/\(.*\)',$|\1|p" "$1/$LIST"
}

# ----------------------------------------------------------------
# create
# ----------------------------------------------------------------

dir="$(fixture 0.2.0-dev v0.1.0-rc1 v0.1.0)"
pending=pg_vectorsearch--0.1.0--0.2.0-dev.sql

expect_status "create: dry run succeeds" 0 tool "$dir" create --dry-run
expect_eq "create: dry run writes no script" "" "$(scripts_in "$dir")"
expect_eq "create: dry run lists nothing" "" "$(listed_in "$dir")"

expect_status "create: defaults to the last final tag and VERSION" 0 \
    tool "$dir" create
expect_eq "create: the script is named for both versions" "$pending" \
    "$(scripts_in "$dir")"
expect_eq "create: the script is listed" "$pending" "$(listed_in "$dir")"
expect_eq "create: the header names the step" \
    " * pg_vectorsearch 0.1.0 -> 0.2.0-dev" \
    "$(sed -n 2p "$dir/sql/$pending")"
expect_eq "create: the script refuses to run under psql" 1 \
    "$(grep -c '^\\echo Use "ALTER EXTENSION pg_vectorsearch UPDATE"' \
        "$dir/sql/$pending")"
expect_eq "create: the list still ends the files() call" \
    "install_data(ext_update_scripts)" "$(tail -1 "$dir/$LIST")"

echo "-- a statement" >>"$dir/sql/$pending"
expect_status "create: a second run succeeds" 0 tool "$dir" create
expect_eq "create: a second run keeps the script's contents" \
    "-- a statement" "$(tail -1 "$dir/sql/$pending")"
expect_eq "create: a second run lists it once" 1 \
    "$(listed_in "$dir" | grep -c .)"

write_list "$dir"
expect_status "create: an unlisted script is listed again" 0 \
    tool "$dir" create
expect_eq "create: the existing script is listed" "$pending" \
    "$(listed_in "$dir")"

expect_status "create: a second pending script is refused" 1 \
    tool "$dir" create --to 0.3.0-dev
expect_eq "create: the refusal writes nothing" "$pending" \
    "$(scripts_in "$dir")"

dir="$(fixture 0.3.0-dev)"
write_list "$dir" pg_vectorsearch--0.1.0--0.2.0.sql
expect_status "create: no tag and no --from is refused" 1 \
    tool "$dir" create
expect_status "create: --from and --to override the defaults" 0 \
    tool "$dir" create --from 0.2.0 --to 0.3.0-dev
expect_eq "create: a new entry goes after the existing ones" \
    "pg_vectorsearch--0.1.0--0.2.0.sql pg_vectorsearch--0.2.0--0.3.0-dev.sql" \
    "$(listed_in "$dir" | tr '\n' ' ' | sed 's/ $//')"

dir="$(fixture 0.2.0-dev v0.1.0)"
expect_status "create: a prerelease --from is refused" 1 \
    tool "$dir" create --from 0.1.0-rc1
expect_status "create: an invalid version is refused" 1 \
    tool "$dir" create --to 0.2
expect_status "create: the same version on both sides is refused" 1 \
    tool "$dir" create --from 0.2.0 --to 0.2.0
expect_status "create: an unknown argument is refused" 1 \
    tool "$dir" create --force

printf 'ext_update_scripts = []\n' >"$dir/$LIST"
expect_status "create: a list in another form is refused" 1 \
    tool "$dir" create

# ----------------------------------------------------------------
# rename
# ----------------------------------------------------------------

dir="$(fixture 0.2.0-dev v0.1.0)"
tool "$dir" create >/dev/null 2>&1
echo "-- a statement" >>"$dir/sql/$pending"
released=pg_vectorsearch--0.1.0--0.2.0.sql

expect_status "rename: dry run succeeds" 0 \
    tool "$dir" rename 0.2.0 --dry-run
expect_eq "rename: dry run renames nothing" "$pending" "$(scripts_in "$dir")"

expect_status "rename: a release candidate is refused" 1 \
    tool "$dir" rename 0.2.0-rc1
expect_status "rename: a development version is refused" 1 \
    tool "$dir" rename 0.2.0-dev
expect_status "rename: a missing version is refused" 1 \
    tool "$dir" rename
expect_eq "rename: a refusal renames nothing" "$pending" "$(scripts_in "$dir")"

expect_status "rename: a final release renames the script" 0 \
    tool "$dir" rename 0.2.0
expect_eq "rename: only the released name is left" "$released" \
    "$(scripts_in "$dir")"
expect_eq "rename: the list names the released script" "$released" \
    "$(listed_in "$dir")"
expect_eq "rename: the script keeps its statements" "-- a statement" \
    "$(tail -1 "$dir/sql/$released")"

expect_status "rename: nothing pending is not an error" 0 \
    tool "$dir" rename 0.2.0
expect_eq "rename: nothing pending changes nothing" "$released" \
    "$(scripts_in "$dir")"

# The version released need not be the one the cycle opened at.
dir="$(fixture 0.2.0-dev v0.1.0)"
tool "$dir" create >/dev/null 2>&1
expect_status "rename: another version than the cycle's is accepted" 0 \
    tool "$dir" rename 1.0.0
expect_eq "rename: the script is named for what was released" \
    pg_vectorsearch--0.1.0--1.0.0.sql "$(scripts_in "$dir")"

dir="$(fixture 0.2.0-dev v0.1.0)"
tool "$dir" create >/dev/null 2>&1
: >"$dir/sql/$released"
expect_status "rename: an existing target is refused" 1 \
    tool "$dir" rename 0.2.0

dir="$(fixture 0.2.0-dev v0.1.0)"
tool "$dir" create >/dev/null 2>&1
write_list "$dir"
expect_status "rename: an unlisted script is renamed" 0 \
    tool "$dir" rename 0.2.0
expect_eq "rename: and listed under its new name" "$released" \
    "$(listed_in "$dir")"

dir="$(fixture 0.2.0-dev v0.1.0)"
: >"$dir/sql/$pending"
: >"$dir/sql/pg_vectorsearch--0.1.5--0.2.0-dev.sql"
expect_status "rename: two pending scripts are refused" 1 \
    tool "$dir" rename 0.2.0

# ----------------------------------------------------------------
# check
# ----------------------------------------------------------------

# A fixture whose library defines the symbols given.
check_fixture() {
    local dir symbol
    dir="$(fixture 0.2.0-dev)"
    mkdir -p "$dir/src/pg"
    for symbol in "$@"; do
        echo "PG_FUNCTION_INFO_V1($symbol);"
    done >"$dir/src/pg/functions.c"
    echo "$dir"
}

c_function() {
    printf "CREATE FUNCTION %s() RETURNS int\n" "$1"
    printf "    AS 'MODULE_PATHNAME', '%s'\n    LANGUAGE C;\n" "$2"
}

dir="$(check_fixture vs_one vs_two vs_unused)"
c_function one vs_one >"$dir/sql/pg_vectorsearch.sql"
c_function two vs_two >"$dir/sql/pg_vectorsearch--0.1.0--0.2.0-dev.sql"
expect_status "check: every named symbol is defined" 0 \
    tool "$dir" check

dir="$(check_fixture vs_one)"
c_function one vs_one >"$dir/sql/pg_vectorsearch.sql"
c_function gone vs_gone >"$dir/sql/pg_vectorsearch--0.1.0--0.2.0.sql"
expect_status "check: a symbol only an old script names is required" 1 \
    tool "$dir" check
expect_eq "check: the missing symbol and its script are named" 1 \
    "$(tool "$dir" check 2>&1 |
        grep -c -- '--0.1.0--0.2.0.sql names the C symbol vs_gone')"

dir="$(check_fixture vs_one)"
{
    echo "CREATE FUNCTION one() RETURNS int"
    echo "    AS 'MODULE_PATHNAME',"
    echo "    'vs_one' LANGUAGE C;"
} >"$dir/sql/pg_vectorsearch.sql"
expect_status "check: a use it cannot parse is refused" 1 \
    tool "$dir" check

dir="$(check_fixture vs_one)"
{
    echo "-- MODULE_PATHNAME is resolved from the control file"
    c_function one vs_one
} >"$dir/sql/pg_vectorsearch.sql"
expect_status "check: a mention in a comment is not a use" 0 \
    tool "$dir" check

dir="$(fixture 0.2.0-dev)"
expect_status "check: a tree that defines and names no symbol passes" 0 \
    tool "$dir" check

expect_status "check: an argument is refused" 1 \
    tool "$dir" check sql/pg_vectorsearch.sql

# ----------------------------------------------------------------
# the command itself
# ----------------------------------------------------------------

expect_status "no command prints the usage and fails" 2 tool "$dir"
expect_status "an unknown command is refused" 1 tool "$dir" seed
expect_status "--help succeeds" 0 tool "$dir" --help

tap_finish
