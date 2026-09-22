#!/bin/bash
# CI check for the Makefile wrapper: `make` must build the extension —
# and only the extension — against the default installed PostgreSQL,
# and `make install` must place the artifacts where pg_config says.
# Runnable locally (PG_CONFIG/CC are picked up from the environment;
# the install step needs sudo unless the prefix is writable).
#
# Usage: ./scripts/ci/makefile-check.sh

set -euo pipefail

cd "$(dirname "$0")/../.."

BUILDDIR="${BUILDDIR:-builddir-make}"
export BUILDDIR

make

ver="$(sed -n "s/^  version: '\([^']*\)',$/\1/p" meson.build)"
[[ -n "$ver" ]] || { echo "FAIL: cannot read version from meson.build"; exit 1; }

so="$BUILDDIR/src/pg/pg_vectorsearch-$ver.so"
[[ -f "$so" ]] || { echo "FAIL: extension library not built: $so"; exit 1; }

# The wrapper defaults to an extension-only build: no CLI, no
# standalone library, no unit tests.
tools="$(find "$BUILDDIR" \( -name 'mkt' -o -name 'run_tests' -o -name 'libpg_vectorsearch*' \) -type f | head -1)"
[[ -z "$tools" ]] || { echo "FAIL: developer tools built by default: $tools"; exit 1; }

# Install and verify the artifacts land in the PostgreSQL directories.
# sudo drops PATH additions (e.g. /usr/lib/postgresql/18/bin), so keep
# the caller's environment for pg_config and meson.
sudo env "PATH=$PATH" ${CC:+CC="$CC"} ${PG_CONFIG:+PG_CONFIG="$PG_CONFIG"} \
    make install BUILDDIR="$BUILDDIR"

pg_config_bin="${PG_CONFIG:-pg_config}"
pkglibdir="$("$pg_config_bin" --pkglibdir)"
sharedir="$("$pg_config_bin" --sharedir)"

for f in \
    "$pkglibdir/pg_vectorsearch-$ver.so" \
    "$sharedir/extension/pg_vectorsearch.control" \
    "$sharedir/extension/pg_vectorsearch--$ver.control" \
    "$sharedir/extension/pg_vectorsearch--$ver.sql"; do
    [[ -f "$f" ]] || { echo "FAIL: not installed: $f"; exit 1; }
done

# Optional live check (CI sets MAKE_CHECK_SMOKE=1): the installed
# extension must actually load in the running packaged PostgreSQL and
# report the version that was just built.
if [[ "${MAKE_CHECK_SMOKE:-0}" == 1 ]]; then
    # sudo -u postgres drops the environment; carry the port
    # explicitly (the CI cluster may not sit on 5432).
    # -q suppresses the DROP/CREATE command tags; keep only the last
    # line anyway in case a notice slips through.
    got="$(sudo -u postgres psql -X -q -A -t ${PGPORT:+-p "$PGPORT"} -c \
        "DROP EXTENSION IF EXISTS pg_vectorsearch;
         CREATE EXTENSION pg_vectorsearch;
         SELECT mkt.extension_version();" | tail -1)"
    [[ "$got" == "$ver" ]] ||
        { echo "FAIL: CREATE EXTENSION reports '$got', built '$ver'"; exit 1; }
fi

echo "makefile-check: OK (built and installed pg_vectorsearch-$ver)"
