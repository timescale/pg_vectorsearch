#!/bin/sh
# Run a TAP test with a clean data directory.
#
# PostgreSQL::Test names its cluster directories deterministically under
# TESTDATADIR, so a run that failed or bailed leaves them behind and the next
# one dies with "could not create data directory ... File exists". Clearing
# the directory first makes the test re-runnable, which matters most when it
# is failing and being iterated on.
set -eu

: "${TESTDATADIR:?must be set}"
: "${PG_PERL_DIR:?must be set}"

rm -rf "$TESTDATADIR"
mkdir -p "$TESTDATADIR"

exec perl -I "$PG_PERL_DIR" "$@"
