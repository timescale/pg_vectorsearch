#!/bin/bash
# Build and test with sanitizers
# Usage: ./scripts/ci/sanitizers.sh [sanitizer]
# Sanitizers: address, undefined, address,undefined (default), thread, memory
#
# Environment variables:
#   PG_CONFIG  - path to a pg_config built WITH THE SAME SANITIZER (e.g. via
#                scripts/ci/build-pg.sh). Unset (the default)
#                builds standalone/CLI only, same as always: a sanitized
#                extension .so cannot be safely dlopen()'d into a plain,
#                non-instrumented backend, so this is deliberately never
#                auto-detected from PATH the way scripts/ci/coverage.sh
#                detects an ordinary pg_config -- the wrong pg_config here
#                doesn't degrade gracefully, it crashes.

set -euo pipefail

# Allow a crashing test process to leave a core dump (CI analyzes them via the
# coredumps action). Harmless locally; cores go to the cwd.
ulimit -c unlimited 2>/dev/null || true

# ASan's stack redzones make PostgreSQL's own manual stack-depth check
# (a raw pointer-distance comparison between two functions' locals) demand
# far more real stack than usual; the default 8MB is not enough headroom
# even for ordinary queries under instrumentation.
ulimit -s unlimited 2>/dev/null || true

SANITIZER="${1:-address,undefined}"
BUILDDIR="builddir-san-${SANITIZER//,/-}"

echo "==> Setting up sanitizer build: $SANITIZER"
echo "    Build directory: $BUILDDIR"

# ASan: detect_leaks=0 for the same reason PostgreSQL's own buildfarm runs
# with it off -- palloc/memory contexts deliberately hold most allocations
# until a context resets or the process exits rather than freeing them
# individually, which LeakSanitizer's exit-time scan cannot distinguish from
# an actual leak; it would fail before a single real test ran.
# detect_stack_use_after_return=0 because ASan's "fake stack" (heap-backed
# storage for locals, used to catch use-after-return) breaks the stack-depth
# check above in a different way: it makes two functions' "local variable"
# addresses come from unrelated fake-stack chunks, so the raw pointer
# distance between them is meaningless and PostgreSQL trips its own
# excessively-deep-recursion guard on ordinary, shallow queries.
#
# UBSan: halt_on_error/abort_on_error so a violation fails the test instead
# of merely being logged and continuing.
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:detect_stack_use_after_return=0:abort_on_error=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}"

# Only enable the PostgreSQL extension build when the caller hands us a
# pg_config for a PostgreSQL built with a matching sanitizer (see the usage
# note above) -- see scripts/ci/build-pg.sh.
MESON_PG_ARGS=("-Dpostgresql=disabled")
if [[ -n "${PG_CONFIG:-}" ]]; then
    echo "==> PostgreSQL extension enabled (pg_config: $PG_CONFIG)"
    MESON_PG_ARGS=("-Dpostgresql=enabled" "-Dpg_config=$PG_CONFIG")

    # No need to repeat build-pg.sh's -fno-sanitize=function here: meson.build
    # already pulls it in via `pg_config --cflags` (VAL_CFLAGS reflects
    # whatever CFLAGS the sanitized PostgreSQL was configured with) into
    # pg_ext_cflags/vs_pg_cflags, which apply only to the PG-facing
    # translation units that actually compile PG's header inlines. A global
    # -Dc_args override here would apply project-wide instead, silently
    # dropping UBSan's function-type check for the standalone/CLI/unit-test
    # targets too, which have no reason to need it.
fi

echo "==> Configuring"
meson setup "$BUILDDIR" \
    -Db_sanitize="$SANITIZER" \
    "${MESON_PG_ARGS[@]}" \
    --wipe 2>/dev/null || \
    meson setup "$BUILDDIR" \
        -Db_sanitize="$SANITIZER" \
        "${MESON_PG_ARGS[@]}"

echo "==> Building"
meson compile -C "$BUILDDIR"

# The extension must actually be installed (not just built) for pg_regress's
# --load-extension / --temp-instance to find it -- it installs from the
# configured prefix, not the build directory.
if [[ "${MESON_PG_ARGS[0]}" != "-Dpostgresql=disabled" ]]; then
    echo "==> Installing extension (required for regression tests)"
    if ! meson install -C "$BUILDDIR" 2>/dev/null; then
        echo "    Retrying with sudo (system PostgreSQL)"
        sudo meson install -C "$BUILDDIR"
    fi
fi

echo "==> Running tests with $SANITIZER sanitizer"
TERM=xterm-256color meson test -C "$BUILDDIR" --verbose

echo "==> Sanitizer tests passed: $SANITIZER"
