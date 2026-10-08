#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# TAP tests for scripts/ci/release-package.sh, which turns the source
# tree into the archive that ships and then proves that archive builds.
# It runs only on release pull requests and release pushes, so without
# these its archive handling reaches CI untested.
#
# meson is stubbed. What needs proving is which archive the script
# picks, what it refuses, and what it hands to the build script inside
# -- none of which a real `meson dist` would show, while costing
# minutes per case.
#
# Usage: ./test/scripts/test_release_package.sh

set -uo pipefail

# shellcheck source=test/scripts/vs_test.sh
source "$(dirname "${BASH_SOURCE[0]}")/vs_test.sh"

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$SCRIPT/scripts/ci/release-package.sh"

# A stub meson earlier on PATH. `meson dist` answers from $STUB_DIST:
#
#   ok        write an archive carrying a build.sh that logs its run
#   nobuild   write an archive with no scripts/ci/build.sh
#   decline   write nothing and exit 0, as meson does in a tree with
#             no VCS -- the case the script's assertion exists for
#   tiny      write an archive too small to be a real one
install_stub_meson() {
    mkdir -p "$VS_WORKSPACE/bin"
    cat >"$VS_WORKSPACE/bin/meson" <<'STUB'
#!/bin/bash
[[ "$1" == dist ]] || exit 0
# -C <builddir> is where the archive goes, as meson's own would.
builddir=.
for ((i = 1; i <= $#; i++)); do
    [[ "${!i}" == -C ]] && builddir="${@:i+1:1}"
done
out="$builddir/meson-dist"
mkdir -p "$out"
case "${STUB_DIST:-ok}" in
    decline)
        echo "Dist currently only works with Git or Mercurial repos"
        exit 0
        ;;
    tiny)
        : >"$out/pg_vectorsearch-$STUB_VERSION.tar.gz"
        exit 0
        ;;
esac
stage="$(mktemp -d)"
root="$stage/pg_vectorsearch-$STUB_VERSION"
mkdir -p "$root/scripts/ci"
if [[ "${STUB_DIST:-ok}" != nobuild ]]; then
    cat >"$root/scripts/ci/build.sh" <<'INNER'
#!/bin/bash
printf '%s\n' "$*" >>"$BUILD_LOG"
printf '%s\n' "$PWD" >>"$BUILD_CWD"
INNER
    chmod +x "$root/scripts/ci/build.sh"
fi
# Padded past the script's 1k floor, which exists to catch a
# truncated or empty archive. Incompressible, so the floor is cleared
# by the archive and not just by what went into it.
head -c 8192 /dev/urandom >"$root/filler"
tar -czf "$out/pg_vectorsearch-$STUB_VERSION.tar.gz" -C "$stage" \
    "pg_vectorsearch-$STUB_VERSION"
rm -rf "$stage"
STUB
    chmod +x "$VS_WORKSPACE/bin/meson"
    PATH="$VS_WORKSPACE/bin:$PATH"
    export PATH
}

install_stub_meson

# How many times the archive's build script ran: the fake build.sh logs
# one line per run. BSD wc right-aligns its count in an eight-column
# field, so the arithmetic expansion is what makes the number comparable
# to a plain integer on a macOS runner.
build_runs() {
    local lines
    lines="$(wc -l <"$1")"
    echo "$((lines))"
}

# $1 description, $2 expected status, $3 $STUB_DIST, $4.. extra args
run_package() {
    local desc="$1" want="$2" mode="$3"
    shift 3
    local dir="$VS_WORKSPACE/pk$RANDOM-$VS_TESTS"
    mkdir -p "$dir"
    BUILD_LOG="$dir.build.log"
    BUILD_CWD="$dir.build.cwd"
    : >"$BUILD_LOG"
    : >"$BUILD_CWD"
    export BUILD_LOG BUILD_CWD
    LAST_DIR="$dir"
    expect_status "package: $desc" "$want" \
        env STUB_DIST="$mode" STUB_VERSION=0.1.0 \
        BUILD_LOG="$BUILD_LOG" BUILD_CWD="$BUILD_CWD" \
        "$SCRIPT" "$dir" "$@"
}

# ----------------------------------------------------------------
# The archive is built and then put through its own build script
# ----------------------------------------------------------------

run_package "an archive that builds passes" 0 ok -Dpostgresql=enabled
expect_eq "package: the archive's build.sh ran once" \
    1 "$(build_runs "$LAST_DIR.build.log")"

# Whatever configures a normal build of this tree has to reach the
# build of the archive too, or the archive is built with other options
# than the ones under test.
expect_eq "package: meson options reach the archive's build" \
    "builddir -Dpostgresql=enabled" \
    "$(head -1 "$LAST_DIR.build.log")"

# The archive's own copy, not this checkout's: an archive that ships a
# broken or absent build script has to fail here.
expect_eq "package: the build runs inside the unpacked archive" \
    "pg_vectorsearch-0.1.0" "$(basename "$(head -1 "$LAST_DIR.build.cwd")")"

# ----------------------------------------------------------------
# What it refuses
# ----------------------------------------------------------------

# meson dist exits 0 when it declines to run at all, so success from it
# says nothing about an archive existing.
run_package "a declined dist is refused" 1 decline
expect_eq "package: nothing is built when no archive appears" \
    0 "$(build_runs "$LAST_DIR.build.log")"

run_package "an archive under the size floor is refused" 1 tiny
run_package "an archive with no build.sh is refused" 1 nobuild
expect_eq "package: nothing is built from an archive without one" \
    0 "$(build_runs "$LAST_DIR.build.log")"

# ----------------------------------------------------------------
# A stale archive is not mistaken for a fresh one
# ----------------------------------------------------------------

# The freshness assertion is the point of the size check above, and a
# real archive left in meson-dist by an earlier run satisfies it while
# saying nothing about the current one. Left to stand, the script
# unpacks that archive, builds it and reports success -- shipping a
# verdict about a tarball it did not produce.
stale="$VS_WORKSPACE/stale"
mkdir -p "$stale"
STALE_LOG="$stale.build.log"
STALE_CWD="$stale.build.cwd"
: >"$STALE_LOG"
: >"$STALE_CWD"
# The stub builds the leftover, so it is a genuine archive rather than
# something that would fail on its own.
env STUB_DIST=ok STUB_VERSION=0.0.9 BUILD_LOG="$STALE_LOG" \
    BUILD_CWD="$STALE_CWD" meson dist -C "$stale" --no-tests >/dev/null

expect_status "package: a leftover archive is not accepted" 1 \
    env STUB_DIST=decline STUB_VERSION=0.1.0 BUILD_LOG="$STALE_LOG" \
    BUILD_CWD="$STALE_CWD" "$SCRIPT" "$stale"
expect_eq "package: a leftover archive is never built" \
    0 "$(build_runs "$STALE_LOG")"

tap_finish
