#!/bin/bash
# pgvector compatibility tests
# Usage: ./scripts/ci/pgvector-compat.sh [builddir]
#
# Builds meerkat and pgvector, creates a temporary PostgreSQL instance,
# and runs the compatibility test suite to verify identical behavior.
#
# Environment:
#   PGVECTOR_DIR  Path to pgvector source (default: ../pgvector)
#   PG_CONFIG     Path to pg_config (auto-detected if not set)

set -euo pipefail

BUILDDIR="${1:-builddir}"
PGVECTOR_DIR="${PGVECTOR_DIR:-../pgvector}"
TMPDIR_BASE=""
PGDATA=""
PGLOG=""
PG_BINDIR=""
VECTOR_CONTROL=""       # set by the event-trigger test when it flips `trusted`
VECTOR_CONTROL_BAK=""

cleanup() {
    # Save postgres log to builddir for artifact upload
    if [[ -f "${PGLOG:-}" && -d "$BUILDDIR" ]]; then
        cp "$PGLOG" "$BUILDDIR/postgres-compat.log" 2>/dev/null || true
    fi
    # Restore pgvector's control file if the event-trigger test flipped it
    # (before removing TMPDIR_BASE, which holds the backup).
    if [[ -n "$VECTOR_CONTROL_BAK" && -f "$VECTOR_CONTROL_BAK" ]]; then
        cp "$VECTOR_CONTROL_BAK" "$VECTOR_CONTROL" 2>/dev/null \
            || sudo cp "$VECTOR_CONTROL_BAK" "$VECTOR_CONTROL" 2>/dev/null || true
    fi
    if [[ -n "$PGDATA" && -d "$PGDATA" ]]; then
        echo "==> Stopping PostgreSQL"
        "$PG_BINDIR/pg_ctl" -D "$PGDATA" -m immediate stop 2>/dev/null || true
    fi
    if [[ -n "$TMPDIR_BASE" && -d "$TMPDIR_BASE" ]]; then
        echo "==> Removing temp directory"
        rm -rf "$TMPDIR_BASE"
    fi
}
trap cleanup EXIT

# Resolve pg_config
PG_CONFIG="${PG_CONFIG:-$(command -v pg_config 2>/dev/null || true)}"
if [[ -z "$PG_CONFIG" ]]; then
    echo "ERROR: pg_config not found. Install PostgreSQL or set PG_CONFIG." >&2
    exit 1
fi
PG_BINDIR="$("$PG_CONFIG" --bindir)"

echo "==> Using PostgreSQL: $PG_CONFIG"
echo "==> pgvector source: $PGVECTOR_DIR"

# Validate pgvector directory
if [[ ! -f "$PGVECTOR_DIR/Makefile" ]]; then
    echo "ERROR: pgvector source not found at $PGVECTOR_DIR" >&2
    echo "  Set PGVECTOR_DIR or clone pgvector into ../pgvector" >&2
    exit 1
fi

# Build and install pgvector
echo "==> Building pgvector"
make -C "$PGVECTOR_DIR" PG_CONFIG="$PG_CONFIG" clean
make -C "$PGVECTOR_DIR" PG_CONFIG="$PG_CONFIG" -j"$(nproc 2>/dev/null || echo 4)"
if ! make -C "$PGVECTOR_DIR" PG_CONFIG="$PG_CONFIG" install 2>/dev/null; then
    echo "    Retrying with sudo (system PostgreSQL)"
    sudo make -C "$PGVECTOR_DIR" PG_CONFIG="$PG_CONFIG" install
fi

# Build and install meerkat
echo "==> Building meerkat"
./scripts/ci/build.sh "$BUILDDIR" -Dpostgresql=enabled

# Create temporary PostgreSQL instance
TMPDIR_BASE=$(mktemp -d)
PGDATA="$TMPDIR_BASE/data"
PGPORT=$(( (RANDOM % 16384) + 49152 ))
PGLOG="$TMPDIR_BASE/postgres.log"

echo "==> Creating temporary PostgreSQL instance (port $PGPORT)"
"$PG_BINDIR/initdb" -D "$PGDATA" --no-locale -E UTF8 -A trust > /dev/null

# Start PostgreSQL
"$PG_BINDIR/pg_ctl" -D "$PGDATA" -l "$PGLOG" \
    -o "-p $PGPORT -k $TMPDIR_BASE" start

# Wait for PostgreSQL to accept connections
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

# Create test database
echo "==> Creating test database"
"$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d postgres \
    -c "CREATE DATABASE compat_test"

# Run compatibility and lifecycle tests (test script loads extensions)
echo "==> Running pgvector compatibility tests"
"$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d compat_test \
    -f test/pg/compat/pgvector.sql

# Run search_path / privilege-escalation hardening tests (needs pgvector,
# which is trusted, plus superuser to exercise the event-trigger path)
echo "==> Running search_path hardening tests"
"$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d compat_test \
    -f test/pg/compat/security.sql

# Event-trigger non-superuser guard.
#
# On managed platforms pgvector is marked "trusted", letting a NON-superuser
# install it. meerkat's event trigger fires on that CREATE EXTENSION as the
# invoking (non-superuser) role; it must NOT attempt its superuser-only compat
# DDL there, because that would fail and roll back the whole pgvector install
# -- meerkat's presence would break pgvector. The guard warns and defers
# instead. This is the one path security.sql can't cover (a plain psql script
# can't mark pgvector trusted), so drive it here: flip the `trusted` flag in
# this run's pgvector control file (restored on exit, incl. failure, by the
# cleanup trap) to reproduce a trusted-pgvector deployment.
echo "==> Running event-trigger non-superuser guard test"
VECTOR_CONTROL="$("$PG_CONFIG" --sharedir)/extension/vector.control"
VECTOR_CONTROL_BAK="$TMPDIR_BASE/vector.control.bak"
cp "$VECTOR_CONTROL" "$VECTOR_CONTROL_BAK"
if ! grep -q '^[[:space:]]*trusted' "$VECTOR_CONTROL"; then
    printf 'trusted = true\n' >> "$VECTOR_CONTROL" 2>/dev/null \
        || sudo sh -c "printf 'trusted = true\n' >> '$VECTOR_CONTROL'"
fi

"$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d postgres \
    -c "CREATE DATABASE evt_test"

# Superuser installs meerkat; a non-superuser role is set up to install pgvector
# (a trusted extension needs only CREATE on the database).
"$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d evt_test -v ON_ERROR_STOP=1 \
    <<'SQL'
CREATE EXTENSION meerkat;
CREATE ROLE evt_nonsuper NOSUPERUSER;
GRANT CREATE ON DATABASE evt_test TO evt_nonsuper;
SQL

# The non-superuser install must SUCCEED (guard warns and defers), not roll
# back. Run without ON_ERROR_STOP so a regression surfaces as the assertion
# below rather than a raw psql abort.
"$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d evt_test \
    -c "SET ROLE evt_nonsuper; CREATE EXTENSION vector;" || true

installed="$("$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d evt_test -At \
    -c "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='vector')")"
if [[ "$installed" == "t" ]]; then
    echo "    PASS: non-superuser pgvector install succeeded (compat deferred)"
else
    echo "    FAIL: non-superuser pgvector install rolled back --" \
         "event-trigger guard regressed" >&2
    exit 1
fi

echo "==> All compatibility tests passed"
