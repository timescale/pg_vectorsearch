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

# Skip when the prefix already holds an injection-points build that also ships
# the TAP perl modules (cache hit). Both are required, so both must be present
# or we rebuild -- otherwise a prefix cached before TAP was added would be
# reused without PostgreSQL::Test.
if [[ -x "$PREFIX/bin/pg_config" ]]; then
    incdir="$("$PREFIX/bin/pg_config" --includedir-server 2>/dev/null || true)"
    pkglibdir="$("$PREFIX/bin/pg_config" --pkglibdir 2>/dev/null || true)"
    if [[ -n "$incdir" ]] &&
        grep -q '#define USE_INJECTION_POINTS' "$incdir/pg_config.h" 2>/dev/null &&
        [[ -f "$pkglibdir/pgxs/src/test/perl/PostgreSQL/Test/Cluster.pm" ]]; then
        echo "==> PostgreSQL $VERSION (injection points + TAP) already installed at $PREFIX"
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
    echo "==> Configuring (debug + cassert + injection points + randomize + tap)"
    # --enable-tap-tests builds and installs the PostgreSQL::Test perl modules
    # (needs IPC::Run at configure time; CI installs libipc-run-perl first).
    # Without it, `make install` ships no PostgreSQL::Test and the replication
    # TAP test's meson guard fails under -Dtap_tests=enabled.
    ./configure \
        --prefix="$PREFIX" \
        --enable-debug \
        --enable-cassert \
        --enable-injection-points \
        --enable-tap-tests \
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

    # PostgreSQL::Test (the TAP perl support) is likewise not shipped by the
    # top-level `make install`; install it explicitly so it lands in
    # <pkglibdir>/pgxs/src/test/perl, where the replication TAP test looks.
    echo "==> Installing TAP perl modules (PostgreSQL::Test)"
    make -s -C src/test/perl install
)

echo "==> Installed to $PREFIX:"
"$PREFIX/bin/pg_config" --version
