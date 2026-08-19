#!/bin/bash
# Build a debug PostgreSQL from source and install it, together with the
# injection_points test extension, into a prefix.
#
# This is the PostgreSQL flavor used by CI jobs that exercise assertions and
# the injection-point isolation tests. The packaged PostgreSQL is a production
# build (no assertions, injection points off, no test modules), so those jobs
# build their own server with this script.
#
# Usage: ./scripts/ci/build-pg-debug.sh <version> <prefix>
#   <version>  PostgreSQL version, e.g. 18.4
#   <prefix>   install prefix, e.g. "$HOME/pg-debug"
#
# Idempotent: if <prefix> already holds a PostgreSQL built with injection
# points, it does nothing, so it composes with a restore-or-build cache step.

set -euo pipefail

VERSION="${1:?usage: build-pg-debug.sh <version> <prefix>}"
PREFIX="${2:?usage: build-pg-debug.sh <version> <prefix>}"

njobs="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# Skip when the prefix already has an injection-points build (cache hit).
if [[ -x "$PREFIX/bin/pg_config" ]]; then
    incdir="$("$PREFIX/bin/pg_config" --includedir-server 2>/dev/null || true)"
    if [[ -n "$incdir" ]] &&
        grep -q '#define USE_INJECTION_POINTS' "$incdir/pg_config.h" 2>/dev/null; then
        echo "==> PostgreSQL $VERSION (injection points) already installed at $PREFIX"
        exit 0
    fi
fi

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

tarball="postgresql-${VERSION}.tar.bz2"
url="https://ftp.postgresql.org/pub/source/v${VERSION}/${tarball}"
echo "==> Downloading $url"
curl -fsSL "$url" -o "$workdir/$tarball"
curl -fsSL "${url}.sha256" -o "$workdir/${tarball}.sha256"

# Verify the tarball against PostgreSQL's published checksum before trusting
# it. TLS protects the download in transit but not the artifact at rest, so
# this guards against a tampered or corrupted tarball at the origin/mirror.
echo "==> Verifying checksum"
if ! (cd "$workdir" && sha256sum -c "${tarball}.sha256"); then
    echo "==> ERROR: checksum verification failed for $tarball" >&2
    exit 1
fi

tar -xf "$workdir/$tarball" -C "$workdir"

(
    cd "$workdir/postgresql-${VERSION}"

    # cassert already turns on MEMORY_CONTEXT_CHECKING (palloc chunk sentinels)
    # and CLOBBER_FREED_MEMORY; RANDOMIZE_ALLOCATED_MEMORY additionally fills
    # fresh allocations with garbage to catch reliance on uninitialized memory.
    echo "==> Configuring (debug + cassert + injection points + randomize)"
    ./configure \
        --prefix="$PREFIX" \
        --enable-debug \
        --enable-cassert \
        --enable-injection-points \
        --without-icu \
        --without-readline \
        --without-zlib \
        CPPFLAGS="-DRANDOMIZE_ALLOCATED_MEMORY"

    echo "==> Building"
    make -j"$njobs" -s

    echo "==> Installing"
    make -s install

    # The test modules are not installed by `make install`; install the
    # injection_points extension explicitly so injection points can be driven
    # from SQL (it is a normal in-tree module build).
    echo "==> Installing injection_points test module"
    make -s -C src/test/modules/injection_points install
)

echo "==> Installed to $PREFIX:"
"$PREFIX/bin/pg_config" --version
