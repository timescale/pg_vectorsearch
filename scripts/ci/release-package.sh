#!/bin/bash
# Build the source tarball and test what the tarball contains.
#
# Usage: ./scripts/ci/release-package.sh <builddir> [meson-options...]
#
# <builddir> is a configured build directory to run `meson dist` from.
# The remaining arguments configure the build of the unpacked archive,
# and want to match what a normal build of this tree uses.
#
# `meson dist` checks its own output by configuring, compiling and
# testing the unpacked tree. That check cannot cover this extension:
# its install step runs after those tests and stages into DESTDIR, so
# PostgreSQL never sees what was built. pg_regress resolves
# --load-extension from PostgreSQL's own directory, so the suite inside
# the check exercises whichever extension is installed there -- the
# tarball's contents never reach the database at all.
#
# So the archive is built with that check disabled, then unpacked and
# put through build.sh: the same configure, compile, install, test
# sequence every other job runs, with the install landing where
# PostgreSQL will look for it. The archive's own copy of the script is
# the one that runs, so a build script missing from it fails here.

set -euo pipefail

BUILDDIR="${1:?usage: $0 <builddir> [meson-options...]}"
shift
MESON_ARGS=("$@")

DISTDIR="$BUILDDIR/meson-dist"

# Emptied first, so what is found below came from this run. `meson
# dist` exits 0 when it declines to run at all, and an archive left by
# an earlier run would then satisfy every check here -- the script
# would unpack it, build it and report on a tarball it did not make.
rm -rf "$DISTDIR"

echo "==> Building the archive from $BUILDDIR"
meson dist -C "$BUILDDIR" --no-tests --formats gztar

ls -l "$DISTDIR" 2>/dev/null || true

# `meson dist` reports success when it declines to run at all -- in a
# tree with no VCS it prints "Dist currently only works with Git or
# Mercurial repos" and exits 0 -- so the archive is asserted rather
# than assumed. It writes the .sha256sum itself, beside the tarball.
TARBALL="$(find "$DISTDIR" -name '*.tar.gz' -size +1k -print -quit)"
[[ -n "$TARBALL" ]] || {
    echo "::error::meson dist produced no tarball"
    exit 1
}

echo "==> Unpacking $TARBALL"
UNPACK="$(mktemp -d)"
trap 'rm -rf "$UNPACK"' EXIT
tar -xzf "$TARBALL" -C "$UNPACK"

SRC="$(find "$UNPACK" -mindepth 1 -maxdepth 1 -type d -print -quit)"
[[ -n "$SRC" ]] || {
    echo "::error::$TARBALL unpacked to no directory"
    exit 1
}

[[ -x "$SRC/scripts/ci/build.sh" ]] || {
    echo "::error::$TARBALL ships no executable scripts/ci/build.sh"
    exit 1
}

echo "==> Building, installing and testing the unpacked archive"
(cd "$SRC" && ./scripts/ci/build.sh builddir \
    ${MESON_ARGS[@]+"${MESON_ARGS[@]}"})

echo "==> $(basename "$TARBALL") builds, installs and passes its tests"
