#!/bin/bash
# Build a PostgreSQL from source and install it, together with the
# injection_points test extension, into a prefix -- optionally instrumented
# with a sanitizer.
#
# This is the PostgreSQL flavor used by CI jobs that need more than the
# packaged PostgreSQL offers: assertions, randomized allocations, and
# injection points (the packaged build is a production build with none of
# that), or -- with a <sanitizer> argument -- a server built with the same
# sanitizer as the extension under test. The latter exists because a
# sanitizer-instrumented extension .so cannot be safely dlopen()'d into a
# non-instrumented backend (ASan/UBSan expect their runtime initialized
# before main()); testing pg_vectorsearch itself under a sanitizer needs
# both sides instrumented consistently.
#
# Usage: ./scripts/ci/build-pg.sh <version> <prefix> [sanitizer]
#   <version>    PostgreSQL version, e.g. 18.4
#   <prefix>     install prefix, e.g. "$HOME/pg-debug" or "$HOME/pg-sanitized-address"
#   <sanitizer>  optional; passed straight to -fsanitize= (e.g. address or
#                undefined). Omit for a plain debug build.
#
# Idempotent: if <prefix> already holds a build stamped for this exact
# version+sanitizer+compiler (see the marker file below), it does nothing,
# so it composes with a restore-or-build cache step.

set -euo pipefail

VERSION="${1:?usage: build-pg.sh <version> <prefix> [sanitizer]}"
PREFIX="${2:?usage: build-pg.sh <version> <prefix> [sanitizer]}"
SANITIZER="${3:-}"

njobs="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# A sanitized build forces clang: sanitizer instrumentation, flag names, and
# quality differ across compilers, and clang is what the rest of this
# project's sanitizer tooling (scripts/ci/sanitizers.sh) already assumes. A
# plain debug build leaves CC (and CFLAGS/LDFLAGS) alone entirely -- CI
# installs gcc for it, and configure picks its own default CFLAGS (-O2)
# only when CFLAGS is unset on its command line at all; even an explicitly
# empty CFLAGS="" suppresses that default (verified against this PG
# version), so the two extra variables are appended to the configure
# invocation only when a sanitizer is actually requested.
# A caller-provided CC (for example "gcc -m32" for a 32-bit build) is part
# of what the prefix was built as, so it goes into the stamp below.
cc_label="${CC:-default}"
configure_env=()
configure_extra_vars=()
if [[ -n "$SANITIZER" ]]; then
    cc_label="clang"
    configure_env+=("CC=clang")

    extra_cflags="-fsanitize=$SANITIZER -fno-omit-frame-pointer -g -O1"

    # UBSan-only: PostgreSQL's own dynahash.c stores a generic comparator
    # through a function pointer whose declared signature doesn't match how
    # it is called -- harmless on every real ABI (the extra trailing
    # argument is never read), but it is undefined behavior by the letter of
    # the C standard, and UBSan's function-type check (part of the default
    # "undefined" group) trips on it at the very first hash lookup, before
    # postgres does anything of ours. Not our bug to fix; excluded so the
    # check that matters (our own code) isn't drowned out by PostgreSQL's
    # own pre-existing UB.
    if [[ "$SANITIZER" == *undefined* ]]; then
        extra_cflags="$extra_cflags -fno-sanitize=function"
    fi

    configure_extra_vars+=("CFLAGS=$extra_cflags" "LDFLAGS=-fsanitize=$SANITIZER")

    # This script runs sanitizer-instrumented binaries itself (configure's
    # feature-detection probes, and the pg_config sanity check below) --
    # it can't assume scripts/ci/sanitizers.sh has already set these, since
    # in CI that's a separate, later step and GitHub Actions steps don't
    # share exported env vars. Without a default here, pg_config's own
    # pstrdup'd, held-until-exit config strings trip LeakSanitizer's
    # default-on leak detection and fail this script outright. Same
    # settings sanitizers.sh uses, for the same reasons (see there); the
    # ${VAR:-...} form leaves a caller-provided value alone.
    export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:detect_stack_use_after_return=0:abort_on_error=1}"
    export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}"
fi

# Marker records exactly what this prefix was built as. A plain grep of
# pg_config.h (e.g. for USE_INJECTION_POINTS) can't tell a debug build from
# a sanitized one, or one sanitizer from another, or catch a compiler
# change -- an explicit stamp is the only reliable cache-hit test.
stamp="$PREFIX/.build-pg-stamp"
stamp_content="version=$VERSION sanitizer=${SANITIZER:-none} cc=$cc_label"
if [[ -x "$PREFIX/bin/pg_config" ]] &&
    [[ "$(cat "$stamp" 2>/dev/null || true)" == "$stamp_content" ]]; then
    echo "==> PostgreSQL $VERSION (${SANITIZER:+$SANITIZER sanitizer, }cassert + injection points + tap) already installed at $PREFIX"
    exit 0
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

    # cassert already turns on MEMORY_CONTEXT_CHECKING (palloc chunk
    # sentinels) and CLOBBER_FREED_MEMORY; RANDOMIZE_ALLOCATED_MEMORY
    # additionally fills fresh allocations with garbage to catch reliance on
    # uninitialized memory (this is what first surfaced a real
    # uninitialized-field bug under the sanitizer build: an otherwise-silent
    # bug on any run where palloc happened to hand back zeroed memory).
    #
    # -O1, not -O0, for a sanitized build: -O0 inflates ASan's per-frame
    # stack redzone padding enormously (observed: bootstrap needing an
    # unreasonable, multi-hundred-MB max_stack_depth just to get past
    # initdb), which -O1 avoids while keeping full sanitizer diagnostics and
    # a debuggable build (-g). --enable-debug covers -g for a plain debug
    # build; a sanitized build asks for -g directly via CFLAGS above instead,
    # since it also needs -O1 in the same variable.
    #
    # --enable-tap-tests builds and installs the PostgreSQL::Test perl
    # modules (needs IPC::Run at configure time; CI installs
    # libipc-run-perl first). Without it, `make install` ships no
    # PostgreSQL::Test and the replication TAP test's meson guard fails
    # under -Dtap_tests=enabled.
    echo "==> Configuring (${SANITIZER:+$SANITIZER sanitizer + }cassert + injection points + randomize + tap)"
    # env, not a literal VAR=val prefix: the var=value pairs come from an
    # array built at runtime, and the "VAR=val cmd" prefix form only works
    # for literal, unexpanded tokens in the source -- not for words that
    # arrive via array/variable expansion.
    env "${configure_env[@]}" ./configure \
        --prefix="$PREFIX" \
        --enable-debug \
        --enable-cassert \
        --enable-injection-points \
        --enable-tap-tests \
        --without-icu \
        --without-readline \
        --without-zlib \
        CPPFLAGS="-DRANDOMIZE_ALLOCATED_MEMORY" \
        "${configure_extra_vars[@]}"

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

# Only after pg_config actually runs successfully: writing the stamp earlier
# would cache a failed or partial install as a hit on the next invocation.
echo "$stamp_content" > "$stamp"
