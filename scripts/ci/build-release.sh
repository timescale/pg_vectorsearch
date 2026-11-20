#!/bin/bash
# Build the extension of a released version from its official source
# tarball and collect the files it installs.
#
# Usage: ./scripts/ci/build-release.sh <version> <dir>
#   <version>  a released version, e.g. 0.1.0
#   <dir>      receives the extension library, control file(s) and install
#              script
#
# The tarball is downloaded from the GitHub release and checked against
# the checksum published with it. It is built with its own Makefile
# wrapper, the way a user builds it, so nothing here depends on that
# version's build options.
#
# Idempotent: if <dir> already holds a build of this version for this
# PostgreSQL major version, it does nothing, so it composes with a
# restore-or-build cache step.
#
# Environment:
#   PG_CONFIG           pg_config of the PostgreSQL to build against
#                       (default: pg_config on PATH).
#   GITHUB_REPOSITORY   owner/name of the repository that published the
#                       release (default: timescale/pg_vectorsearch).

set -euo pipefail

VERSION="${1:?usage: build-release.sh <version> <dir>}"
DIR="${2:?usage: build-release.sh <version> <dir>}"

cd "$(dirname "$0")/../.."

NAME=pg_vectorsearch
PG_CONFIG="$(command -v "${PG_CONFIG:-pg_config}")"
pg_major="$("$PG_CONFIG" --version | sed -E 's/^[^0-9]*([0-9]+).*/\1/')"
stamp="$DIR/.built-$VERSION-pg$pg_major"

if [[ -f "$stamp" ]]; then
    echo "==> $NAME $VERSION is already built in $DIR"
    exit 0
fi

tarball="$NAME-$VERSION.tar.gz"
url="https://github.com/${GITHUB_REPOSITORY:-timescale/$NAME}"
url="$url/releases/download/v$VERSION"

work="$(mktemp -d "${TMPDIR:-/tmp}/pgvs-release.XXXXXX")"
trap 'rm -rf "$work"' EXIT

echo "==> Downloading $tarball"
curl --fail --silent --show-error --location --retry 3 \
    -o "$work/$tarball" "$url/$tarball" \
    -o "$work/$tarball.sha256sum" "$url/$tarball.sha256sum"
(cd "$work" && sha256sum --check --quiet "$tarball.sha256sum")

echo "==> Building $NAME $VERSION"
mkdir "$work/src"
tar -xzf "$work/$tarball" -C "$work/src" --strip-components=1
if ! make -C "$work/src" BUILDDIR="$work/build" PG_CONFIG="$PG_CONFIG" \
        >"$work/build.log" 2>&1; then
    cat "$work/build.log" >&2
    exit 1
fi

mkdir -p "$DIR"
cp "$work/build/src/pg/$NAME"*.so "$work/build/src/pg/$NAME"*.control \
    "$work/build/src/pg/$NAME--$VERSION.sql" "$DIR/"
touch "$stamp"
echo "==> Collected $NAME $VERSION in $DIR"
