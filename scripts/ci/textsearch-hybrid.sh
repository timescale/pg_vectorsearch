#!/bin/bash
# Hybrid search (BM25 + vector) tests
# Usage: ./scripts/ci/textsearch-hybrid.sh [builddir]
#
# Builds pg_vectorsearch, ensures pg_textsearch is installed, creates a temporary
# PostgreSQL instance with both extensions, and runs the hybrid-search test
# suite (test/pg/compat/hybridsearch.sql) covering BM25+vector
# filter-then-rank queries in both directions and reciprocal rank fusion.
#
# Environment:
#   PGTEXTSEARCH_DIR     Path to pg_textsearch source (default: ../pg_textsearch).
#                         Used to build pg_textsearch from scratch when
#                         PGTEXTSEARCH_STAGE is not set.
#   PGTEXTSEARCH_STAGE    Path to a DESTDIR-staged pg_textsearch build (see
#                         build-pg-textsearch.sh). When set, that staged
#                         build is installed directly and PGTEXTSEARCH_DIR is
#                         ignored -- this is how CI reuses cached build
#                         output (see .github/actions/setup-pg-textsearch).
#                         Local runs typically leave this unset.
#   PG_CONFIG             Path to pg_config (auto-detected if not set)

set -euo pipefail

BUILDDIR="${1:-builddir}"
PGTEXTSEARCH_DIR="${PGTEXTSEARCH_DIR:-../pg_textsearch}"
PGTEXTSEARCH_STAGE="${PGTEXTSEARCH_STAGE:-}"
TMPDIR_BASE=""
PGDATA=""
PGLOG=""
PG_BINDIR=""

cleanup() {
    # Save postgres log to builddir for artifact upload
    if [[ -f "${PGLOG:-}" && -d "$BUILDDIR" ]]; then
        cp "$PGLOG" "$BUILDDIR/postgres-hybrid.log" 2>/dev/null || true
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

TMPDIR_BASE=$(mktemp -d)

# Install pg_textsearch: reuse a pre-staged build if one was handed to us
# (CI's cached path), otherwise build it from source here (local dev path).
if [[ -n "$PGTEXTSEARCH_STAGE" ]]; then
    echo "==> Using pre-built pg_textsearch stage: $PGTEXTSEARCH_STAGE"
else
    echo "==> pg_textsearch source: $PGTEXTSEARCH_DIR"
    PGTEXTSEARCH_STAGE="$TMPDIR_BASE/pg_textsearch-stage"
    PG_CONFIG="$PG_CONFIG" ./scripts/ci/build-pg-textsearch.sh \
        "$PGTEXTSEARCH_DIR" "$PGTEXTSEARCH_STAGE"
fi
./scripts/ci/install-pg-textsearch.sh "$PGTEXTSEARCH_STAGE"

# Build and install pg_vectorsearch
echo "==> Building pg_vectorsearch"
./scripts/ci/build.sh "$BUILDDIR" -Dpostgresql=enabled

# Create temporary PostgreSQL instance
PGDATA="$TMPDIR_BASE/data"
PGPORT=$(( (RANDOM % 16384) + 49152 ))
PGLOG="$TMPDIR_BASE/postgres.log"

echo "==> Creating temporary PostgreSQL instance (port $PGPORT)"
"$PG_BINDIR/initdb" -D "$PGDATA" --no-locale -E UTF8 -A trust > /dev/null

# Start PostgreSQL. pg_textsearch must be loaded via shared_preload_libraries
# (a postmaster-start-time-only setting), or CREATE EXTENSION pg_textsearch
# fails with "library not loaded".
"$PG_BINDIR/pg_ctl" -D "$PGDATA" -l "$PGLOG" \
    -o "-p $PGPORT -k $TMPDIR_BASE -c shared_preload_libraries=pg_textsearch" \
    start

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
    -c "CREATE DATABASE hybrid_test"

# Run hybrid search tests (test script loads both extensions)
echo "==> Running hybrid search tests"
"$PG_BINDIR/psql" -h "$TMPDIR_BASE" -p "$PGPORT" -d hybrid_test \
    -f test/pg/compat/hybridsearch.sql

echo "==> All hybrid search tests passed"
