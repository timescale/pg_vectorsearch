#!/bin/bash
# Build pg_textsearch and stage its install tree for caching.
#
# Usage: ./scripts/ci/build-pg-textsearch.sh <src_dir> <stage_dir>
#   <src_dir>    pg_textsearch source checkout (PGXS Makefile at its root)
#   <stage_dir>  DESTDIR-staged install tree; wiped and repopulated here so
#                it can be cached and later applied to a live PostgreSQL via
#                install-pg-textsearch.sh
#
# Environment:
#   PG_CONFIG   Path to pg_config (auto-detected if not set)

set -euo pipefail

SRC_DIR="${1:?usage: build-pg-textsearch.sh <src_dir> <stage_dir>}"
STAGE_DIR="${2:?usage: build-pg-textsearch.sh <src_dir> <stage_dir>}"

PG_CONFIG="${PG_CONFIG:-$(command -v pg_config 2>/dev/null || true)}"
if [[ -z "$PG_CONFIG" ]]; then
    echo "ERROR: pg_config not found. Install PostgreSQL or set PG_CONFIG." >&2
    exit 1
fi

if [[ ! -f "$SRC_DIR/Makefile" ]]; then
    echo "ERROR: pg_textsearch source not found at $SRC_DIR" >&2
    exit 1
fi

njobs="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

echo "==> Building pg_textsearch ($SRC_DIR) against $("$PG_CONFIG" --version)"
make -C "$SRC_DIR" PG_CONFIG="$PG_CONFIG" clean
make -C "$SRC_DIR" PG_CONFIG="$PG_CONFIG" -j"$njobs"

echo "==> Staging install to $STAGE_DIR"
rm -rf "$STAGE_DIR"
make -C "$SRC_DIR" PG_CONFIG="$PG_CONFIG" install DESTDIR="$STAGE_DIR"

echo "==> Staged $(find "$STAGE_DIR" -type f | wc -l) files"
