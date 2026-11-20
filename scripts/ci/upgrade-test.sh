#!/bin/bash
# Test ALTER EXTENSION UPDATE from the previous release.
#
# Usage: ./scripts/ci/upgrade-test.sh [--previous-version]
#
# Installs the previous release next to the current build and runs the
# tests in test/pg/upgrade with pg_regress, followed by the regression
# tests against the updated extension. The current build has to
# be installed already (meson install, or make install).
#
# The previous release is the <from> side of the one upgrade script in sql/
# whose <to> side is the version in VERSION. --previous-version prints it
# and exits. Its files are taken from the first place that has them:
#
#   1. the PostgreSQL directories, if it is already installed there;
#   2. OLD_RELEASE_DIR, which build-release.sh fills on the first run.
#
# What gets installed depends on the release. Its install script always
# is. A release that has a library of its own (0.1.0 installed
# pg_vectorsearch-0.1.0.so) also gets that library and its control file.
# Later releases share the one library, so the current build serves their
# install script too.
#
# Environment:
#   PG_CONFIG         pg_config of the PostgreSQL to test against
#                     (default: pg_config on PATH).
#   OLD_RELEASE_DIR   Directory that keeps the previous release's build
#                     between runs (default: rebuilt every run).
#   UPGRADE_TEST_PORT Port of the scratch server (default: 54329).
#   GITHUB_STEP_SUMMARY
#                     Set by GitHub Actions. The result is appended to
#                     this file as a table for the run's summary page.
#   KEEP_WORKDIR      Set to 1 to keep the scratch directory, with the
#                     test results and server log, after the run.
#
# Installing into the PostgreSQL directories falls back to sudo. Files this
# run installed for the previous release are removed again on exit.

set -euo pipefail

cd "$(dirname "$0")/../.."

NAME=pg_vectorsearch
PG_CONFIG="$(command -v "${PG_CONFIG:-pg_config}")"
BINDIR="$("$PG_CONFIG" --bindir)"
PKGLIBDIR="$("$PG_CONFIG" --pkglibdir)"
EXTDIR="$("$PG_CONFIG" --sharedir)/extension"

to="$(tr -d '[:space:]' < VERSION)"
shopt -s nullglob
scripts=(sql/"$NAME"--*--"$to".sql)
shopt -u nullglob
if [[ ${#scripts[@]} -ne 1 ]]; then
    echo "ERROR: expected one upgrade script to $to in sql/, found" \
        "${#scripts[@]}" >&2
    exit 1
fi
from="$(basename "${scripts[0]}" .sql)"
from="${from#"$NAME"--}"
from="${from%--"$to"}"

if [[ "${1:-}" == --previous-version ]]; then
    echo "$from"
    exit 0
fi

for f in "$EXTDIR/$NAME--$to.sql" "$EXTDIR/$NAME--$from--$to.sql"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f is missing; install the current build first" >&2
        exit 1
    fi
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/pgvs-upgrade.XXXXXX")"
PORT="${UPGRADE_TEST_PORT:-54329}"
old_sql="$NAME--$from.sql"
old_lib="$NAME-$from.so"
old_control="$NAME--$from.control"
installed=()

# Run a command, with sudo if the PostgreSQL directories are not writable.
as_installer() {
    if [[ -w "$PKGLIBDIR" && -w "$EXTDIR" ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

cleanup() {
    "$BINDIR/pg_ctl" -D "$WORK/data" -m immediate stop >/dev/null 2>&1 || true
    if [[ ${#installed[@]} -gt 0 ]]; then
        as_installer rm -f "${installed[@]}" || true
    fi
    if [[ "${KEEP_WORKDIR:-0}" == 1 ]]; then
        echo "==> Kept $WORK"
    else
        rm -rf "$WORK"
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------
# The previous release
# ---------------------------------------------------------------------

old_release="${OLD_RELEASE_DIR:-$WORK/old-release}"
if [[ -f "$EXTDIR/$old_sql" ]]; then
    echo "==> Using the installed $NAME $from"
    old_source="already installed"
else
    old_source="downloaded and built"
    stamps=("$old_release"/.built-"$from"-*)
    [[ -e "${stamps[0]}" ]] && old_source="reused build"
    PG_CONFIG="$PG_CONFIG" ./scripts/ci/build-release.sh "$from" \
        "$old_release"

    echo "==> Installing $NAME $from"
    installed+=("$EXTDIR/$old_sql")
    as_installer install -m 644 "$old_release/$old_sql" "$EXTDIR/"
    if [[ -f "$old_release/$old_lib" ]]; then
        installed+=("$PKGLIBDIR/$old_lib" "$EXTDIR/$old_control")
        as_installer install -m 755 "$old_release/$old_lib" "$PKGLIBDIR/"
        as_installer install -m 644 "$old_release/$old_control" "$EXTDIR/"
    fi
fi

# Whether the previous release runs on a library of its own.
own_library=false
[[ -f "$PKGLIBDIR/$old_lib" ]] && own_library=true

# Append the result to the GitHub Actions run summary, if there is one.
#   $1  outcome of the upgrade tests
#   $2  outcome of the regression tests on the updated extension
summarize() {
    [[ -n "${GITHUB_STEP_SUMMARY:-}" ]] || return 0
    {
        echo "### Upgrade test: $NAME $from → $to"
        echo
        echo "| | |"
        echo "| --- | --- |"
        echo "| PostgreSQL | $("$PG_CONFIG" --version) |"
        echo "| Previous release | $old_source |"
        echo "| Previous release has its own library | $own_library |"
        echo "| Upgrade tests | $1 |"
        echo "| Regression tests on the updated extension | $2 |"
    } >>"$GITHUB_STEP_SUMMARY"
}

# ---------------------------------------------------------------------
# A scratch server
# ---------------------------------------------------------------------

echo "==> Starting a scratch server"
"$BINDIR/initdb" -D "$WORK/data" -A trust --no-locale >/dev/null
{
    echo "port = $PORT"
    echo "listen_addresses = ''"
    echo "unix_socket_directories = '$WORK'"
    echo "search_path = 'public'"
    echo "logging_collector = on"
} >>"$WORK/data/postgresql.conf"
"$BINDIR/pg_ctl" -D "$WORK/data" -l "$WORK/server.log" -w start >/dev/null

run_tests() {
    "$PKGLIBDIR/pgxs/src/test/regress/pg_regress" --bindir="$BINDIR" \
        --host="$WORK" --port="$PORT" "$@"
}

# ---------------------------------------------------------------------
# The upgrade tests in test/pg/upgrade. upgrade.sql updates the database;
# the other scripts run after it, in name order, against the updated
# extension.
# ---------------------------------------------------------------------

upgrade_tests=(upgrade)
for f in test/pg/upgrade/sql/*.sql; do
    name="$(basename "$f" .sql)"
    [[ "$name" == upgrade ]] || upgrade_tests+=("$name")
done

echo "==> Updating from $from to $to"
if ! PGVS_UPGRADE_FROM="$from" PGVS_UPGRADE_TO="$to" \
        PGVS_UPGRADE_OWN_LIBRARY="$own_library" run_tests \
        --inputdir=test/pg/upgrade --outputdir="$WORK/upgrade" \
        "${upgrade_tests[@]}"; then
    cat "$WORK/upgrade/regression.diffs" 2>/dev/null || true
    summarize "failed" "not run"
    exit 1
fi
upgrade_result="${#upgrade_tests[@]} passed"

# ---------------------------------------------------------------------
# The regression tests, as test/pg/meson.build lists them, in the same
# database
# ---------------------------------------------------------------------

mapfile -t regress_tests < <(
    sed -n "/^ *test_names = \[/,/^ *\]/s/^ *'\([a-z0-9_]*\)',$/\1/p" \
        test/pg/meson.build)
if [[ ${#regress_tests[@]} -eq 0 ]]; then
    echo "ERROR: found no test_names in test/pg/meson.build" >&2
    summarize "$upgrade_result" "not run"
    exit 1
fi

echo "==> Running the regression tests on the updated extension"
if ! run_tests --use-existing --inputdir=test/pg \
        --outputdir="$WORK/regress" "${regress_tests[@]}"; then
    cat "$WORK/regress/regression.diffs" 2>/dev/null || true
    summarize "$upgrade_result" "failed"
    exit 1
fi
summarize "$upgrade_result" "${#regress_tests[@]} passed"
echo "==> Upgrade from $from to $to passed"
