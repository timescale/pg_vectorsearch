#!/bin/bash
# pgvector compatibility & install-script security tests
#
# Builds pg_vectorsearch and pgvector against a throwaway PostgreSQL
# instance and runs two suites against it: the pgvector
# compatibility/lifecycle tests, and the install-script security
# (search_path / privilege-escalation) tests. Both need pgvector built
# and installed, so they share one build and one instance.
#
# Usage:
#   pgvector-compat.sh [command] [builddir]
#
# Commands (default: all):
#   all       build, start an instance, run both suites, tear down (local run)
#   setup     build pgvector + pg_vectorsearch, start a temp instance,
#             create the db
#   compat    run the pgvector compatibility suite       (after `setup`)
#   security  run the install-script security suite       (after `setup`)
#   teardown  stop the instance and remove its temp dir
#
# CI runs the phases as separate steps sharing one instance, so a failure is
# reported as compatibility vs security rather than one opaque "compat" check;
# `all` is the convenient local one-shot. Phase state (the instance's
# connection info) is passed between invocations via a file under the builddir.
#
# Environment:
#   PGVECTOR_DIR  Path to pgvector source (default: ../pgvector)
#   PG_CONFIG     Path to pg_config (auto-detected if not set)

set -euo pipefail

CMD="${1:-all}"
BUILDDIR="${2:-builddir}"
STATE_FILE="$BUILDDIR/.pgvector-compat-state"

# --------------------------------------------------------------------------
# setup: build pgvector + pg_vectorsearch, start a temp instance,
# create the test db
# --------------------------------------------------------------------------
do_setup() {
    PG_CONFIG="${PG_CONFIG:-$(command -v pg_config 2>/dev/null || true)}"
    if [[ -z "$PG_CONFIG" ]]; then
        echo "ERROR: pg_config not found. Install PostgreSQL or set PG_CONFIG." >&2
        exit 1
    fi
    PG_BINDIR="$("$PG_CONFIG" --bindir)"
    PGVECTOR_DIR="${PGVECTOR_DIR:-../pgvector}"

    echo "==> Using PostgreSQL: $PG_CONFIG"
    echo "==> pgvector source: $PGVECTOR_DIR"
    if [[ ! -f "$PGVECTOR_DIR/Makefile" ]]; then
        echo "ERROR: pgvector source not found at $PGVECTOR_DIR" >&2
        echo "  Set PGVECTOR_DIR or clone pgvector into ../pgvector" >&2
        exit 1
    fi

    echo "==> Building pgvector"
    make -C "$PGVECTOR_DIR" PG_CONFIG="$PG_CONFIG" clean
    make -C "$PGVECTOR_DIR" PG_CONFIG="$PG_CONFIG" -j"$(nproc 2>/dev/null || echo 4)"
    if ! make -C "$PGVECTOR_DIR" PG_CONFIG="$PG_CONFIG" install 2>/dev/null; then
        echo "    Retrying with sudo (system PostgreSQL)"
        sudo make -C "$PGVECTOR_DIR" PG_CONFIG="$PG_CONFIG" install
    fi

    echo "==> Building pg_vectorsearch"
    ./scripts/ci/build.sh "$BUILDDIR" -Dpostgresql=enabled

    # Uppercase names to match what's written to (and later sourced from) the
    # state file, so both halves read consistently.
    local TMPDIR_BASE PGDATA PGPORT PGLOG i
    TMPDIR_BASE=$(mktemp -d)
    PGDATA="$TMPDIR_BASE/data"
    PGPORT=$(( (RANDOM % 16384) + 49152 ))
    PGLOG="$TMPDIR_BASE/postgres.log"

    echo "==> Creating temporary PostgreSQL instance (port $PGPORT)"
    "$PG_BINDIR/initdb" -D "$PGDATA" --no-locale -E UTF8 -A trust > /dev/null
    "$PG_BINDIR/pg_ctl" -D "$PGDATA" -l "$PGLOG" \
        -o "-p $PGPORT -k $TMPDIR_BASE" start

    # Record connection info now (before waiting/CREATE DATABASE) so teardown
    # can stop the instance even if a later step here fails.
    {
        echo "PG_CONFIG='$PG_CONFIG'"
        echo "PG_BINDIR='$PG_BINDIR'"
        echo "TMPDIR_BASE='$TMPDIR_BASE'"
        echo "PGDATA='$PGDATA'"
        echo "PGPORT='$PGPORT'"
        echo "PGLOG='$PGLOG'"
    } > "$STATE_FILE"

    for i in $(seq 1 30); do
        if "$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d postgres \
            -c "SELECT 1" > /dev/null 2>&1; then
            break
        fi
        if [[ $i -eq 30 ]]; then
            echo "ERROR: PostgreSQL failed to start" >&2
            cat "$PGLOG" >&2
            exit 1
        fi
        sleep 0.5
    done

    echo "==> Creating test database"
    "$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d postgres \
        -c "CREATE DATABASE compat_test"
}

load_state() {
    if [[ ! -f "$STATE_FILE" ]]; then
        echo "ERROR: no instance state ($STATE_FILE); run 'setup' first." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$STATE_FILE"
}

# --------------------------------------------------------------------------
# compat: pgvector compatibility / lifecycle suite
# --------------------------------------------------------------------------
do_compat() {
    load_state
    echo "==> Running pgvector compatibility tests"
    "$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d compat_test \
        -f test/pg/compat/pgvector.sql
}

# --------------------------------------------------------------------------
# security: install-script search_path / privilege-escalation suite
# --------------------------------------------------------------------------
do_security() {
    load_state

    echo "==> Running install-script security (search_path hardening) tests"
    "$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d compat_test \
        -f test/pg/compat/security.sql

    # Event-trigger non-superuser guard.
    #
    # On managed platforms pgvector is marked "trusted", letting a
    # NON-superuser install it. pg_vectorsearch's event trigger fires on
    # that CREATE EXTENSION as the invoking (non-superuser) role; it
    # must NOT attempt its superuser-only compat DDL there, because that
    # would fail and roll back the whole pgvector install --
    # pg_vectorsearch's presence would break pgvector. The guard warns
    # and defers instead. This is the one path security.sql can't cover
    # (a plain psql script can't mark pgvector trusted), so drive it
    # here: flip the `trusted` flag in this run's pgvector control file
    # (restored below) to reproduce a trusted-pgvector deployment.
    echo "==> Running event-trigger non-superuser guard test"
    local vector_control vector_control_bak installed
    vector_control="$("$PG_CONFIG" --sharedir)/extension/vector.control"
    vector_control_bak="$TMPDIR_BASE/vector.control.bak"
    cp "$vector_control" "$vector_control_bak"
    if ! grep -q '^[[:space:]]*trusted' "$vector_control"; then
        printf 'trusted = true\n' >> "$vector_control" 2>/dev/null \
            || sudo sh -c "printf 'trusted = true\n' >> '$vector_control'"
    fi

    "$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d postgres \
        -c "CREATE DATABASE evt_test"

    # Superuser installs pg_vectorsearch; a non-superuser role is set up
    # to install pgvector (a trusted extension needs only CREATE on the
    # database).
    "$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d evt_test \
        -v ON_ERROR_STOP=1 <<'SQL'
CREATE EXTENSION pg_vectorsearch;
CREATE ROLE evt_nonsuper NOSUPERUSER;
GRANT CREATE ON DATABASE evt_test TO evt_nonsuper;
SQL

    # The non-superuser install must SUCCEED (guard warns and defers), not roll
    # back. Run without ON_ERROR_STOP so a regression surfaces as the assertion
    # below rather than a raw psql abort.
    "$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d evt_test \
        -c "SET ROLE evt_nonsuper; CREATE EXTENSION vector;" || true
    installed="$("$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d evt_test \
        -At -c "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='vector')")"

    # Restore the control file before the assertion can exit the script.
    cp "$vector_control_bak" "$vector_control" 2>/dev/null \
        || sudo cp "$vector_control_bak" "$vector_control" 2>/dev/null || true

    if [[ "$installed" == "t" ]]; then
        echo "    PASS: non-superuser pgvector install succeeded (compat deferred)"
    else
        echo "    FAIL: non-superuser pgvector install rolled back --" \
             "event-trigger guard regressed" >&2
        exit 1
    fi
}

# --------------------------------------------------------------------------
# teardown: stop the instance, save the log, remove the temp dir
# --------------------------------------------------------------------------
do_teardown() {
    [[ -f "$STATE_FILE" ]] || return 0
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    if [[ -f "${PGLOG:-}" && -d "$BUILDDIR" ]]; then
        cp "$PGLOG" "$BUILDDIR/postgres-compat.log" 2>/dev/null || true
    fi
    if [[ -n "${PGDATA:-}" && -d "$PGDATA" ]]; then
        echo "==> Stopping PostgreSQL"
        "$PG_BINDIR/pg_ctl" -D "$PGDATA" -m immediate stop 2>/dev/null || true
    fi
    if [[ -n "${TMPDIR_BASE:-}" && -d "$TMPDIR_BASE" ]]; then
        echo "==> Removing temp directory"
        rm -rf "$TMPDIR_BASE"
    fi
    rm -f "$STATE_FILE"
}

case "$CMD" in
    setup)    do_setup ;;
    compat)   do_compat ;;
    security) do_security ;;
    teardown) do_teardown ;;
    all)
        trap do_teardown EXIT
        do_setup
        do_compat
        do_security
        echo "==> All compatibility and security tests passed"
        ;;
    *)
        echo "ERROR: unknown command '$CMD'" \
             "(use setup|compat|security|teardown|all)" >&2
        exit 1
        ;;
esac
