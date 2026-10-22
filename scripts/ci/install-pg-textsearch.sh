#!/bin/bash
# Apply a DESTDIR-staged pg_textsearch build (see build-pg-textsearch.sh) onto
# the live PostgreSQL installation.
#
# Usage: ./scripts/ci/install-pg-textsearch.sh <stage_dir>
#   <stage_dir>  staged install tree produced by build-pg-textsearch.sh

set -euo pipefail

STAGE_DIR="${1:?usage: install-pg-textsearch.sh <stage_dir>}"

if [[ ! -d "$STAGE_DIR" ]]; then
    echo "ERROR: staged pg_textsearch install not found at $STAGE_DIR" >&2
    echo "  Run build-pg-textsearch.sh first." >&2
    exit 1
fi

echo "==> Installing staged pg_textsearch build from $STAGE_DIR"
if ! cp -a "$STAGE_DIR"/. /; then
    echo "    Retrying with sudo (system PostgreSQL)"
    sudo cp -a "$STAGE_DIR"/. /
fi
