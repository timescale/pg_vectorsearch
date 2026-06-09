#!/bin/bash
# Build and run tests
# Usage: ./scripts/ci/build.sh [builddir] [meson-options...]
#
# Extra arguments after builddir are passed directly to meson setup.
# When PostgreSQL is available (pg_config in PATH), the extension is built
# and installed automatically so that regression tests can run.
#
# Examples:
#   ./scripts/ci/build.sh                          # default build
#   ./scripts/ci/build.sh builddir                  # custom builddir
#   ./scripts/ci/build.sh builddir -Dpostgresql=disabled  # standalone only

set -euo pipefail

# Allow a crashing test process to leave a core dump (CI analyzes them via the
# coredumps action). Harmless locally; cores go to the cwd.
ulimit -c unlimited 2>/dev/null || true

BUILDDIR="${1:-builddir}"
shift || true
MESON_ARGS=("$@")

# Auto-detect PostgreSQL compiler to avoid compiler family mismatch.
# The meson build checks that the compiler family (gcc vs clang) matches
# pg_config --cc and skips the extension on mismatch.
# Honor -Dpg_config=<path> from meson args over the system pg_config.
if [[ -z "${PG_CONFIG:-}" ]]; then
    for arg in ${MESON_ARGS[@]+"${MESON_ARGS[@]}"}; do
        if [[ "$arg" =~ ^-Dpg_config=(.+)$ ]]; then
            PG_CONFIG="${BASH_REMATCH[1]}"
            break
        fi
    done
    PG_CONFIG="${PG_CONFIG:-$(command -v pg_config 2>/dev/null || true)}"
fi
if [[ -n "$PG_CONFIG" ]]; then
    PG_CC=$("$PG_CONFIG" --cc 2>/dev/null || true)
    if [[ -n "$PG_CC" ]]; then
        CC="${CC:-$PG_CC}"
        export CC
        echo "==> Using PostgreSQL compiler: $CC"
    fi
fi

echo "==> Setting up build directory: $BUILDDIR"
meson setup "$BUILDDIR" ${MESON_ARGS[@]+"${MESON_ARGS[@]}"} --wipe 2>/dev/null || \
    meson setup "$BUILDDIR" ${MESON_ARGS[@]+"${MESON_ARGS[@]}"}

echo "==> Building"
meson compile -C "$BUILDDIR"

# Install targets (extension .so and SQL files when PG is enabled; no-op
# when the extension is disabled). System PostgreSQL directories typically
# require elevated permissions.
echo "==> Installing"
if ! meson install -C "$BUILDDIR" 2>/dev/null; then
    echo "    Retrying with sudo (system PostgreSQL)"
    sudo meson install -C "$BUILDDIR"
fi

echo "==> Running tests"
meson test -C "$BUILDDIR"

echo "==> Build and tests passed"
