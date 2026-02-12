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

cleanup() {
    # Save postgres log to builddir for artifact upload
    if [[ -f "${PGLOG:-}" && -d "$BUILDDIR" ]]; then
        cp "$PGLOG" "$BUILDDIR/postgres-compat.log" 2>/dev/null || true
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

# Create test database and load extensions
echo "==> Loading extensions"
"$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d postgres <<'SQL'
CREATE DATABASE compat_test;
SQL
"$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d compat_test <<'SQL'
CREATE EXTENSION vector;
CREATE EXTENSION meerkat;
SQL

# Run compatibility tests
echo "==> Running pgvector compatibility tests"
"$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d compat_test \
    -f test/pg/compat/pgvector.sql

echo "==> All compatibility tests passed"
