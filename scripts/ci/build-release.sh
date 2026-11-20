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
# A release that is still a draft has a tag but no downloadable files.
# The release workflow opens the next cycle in that state, so the source
# then comes from the tag v<version>.
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
#   GH_TOKEN            A token that can read that repository. With it
#                       the downloads go through gh, which a private
#                       repository needs. Without it they are anonymous.

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
repo="${GITHUB_REPOSITORY:-timescale/$NAME}"

work="$(mktemp -d "${TMPDIR:-/tmp}/pgvs-release.XXXXXX")"
trap 'rm -rf "$work"' EXIT

fetch() {
    curl --fail --silent --show-error --location --retry 3 "$@"
}

authenticated() {
    [[ -n "${GH_TOKEN:-}" ]] && command -v gh >/dev/null
}

# The release's tarball and checksum, into $work.
download_release() {
    if authenticated; then
        gh release download "v$VERSION" --repo "$repo" --dir "$work" \
            --pattern "$tarball" --pattern "$tarball.sha256sum"
    else
        local url="https://github.com/$repo/releases/download/v$VERSION"
        fetch -o "$work/$tarball" "$url/$tarball" \
            -o "$work/$tarball.sha256sum" "$url/$tarball.sha256sum"
    fi
}

# The source at the tag v<version>, unpacked into $work/src.
unpack_tag() {
    if git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null; then
        git archive "v$VERSION" | tar -x -C "$work/src"
        return
    fi
    if authenticated; then
        gh api "repos/$repo/tarball/refs/tags/v$VERSION" >"$work/tag.tar.gz"
    else
        fetch -o "$work/tag.tar.gz" \
            "https://github.com/$repo/archive/refs/tags/v$VERSION.tar.gz"
    fi
    tar -xzf "$work/tag.tar.gz" -C "$work/src" --strip-components=1
}

mkdir "$work/src"
echo "==> Downloading $tarball"
if download_release; then
    (cd "$work" && sha256sum --check --quiet "$tarball.sha256sum")
    tar -xzf "$work/$tarball" -C "$work/src" --strip-components=1
else
    echo "==> $tarball is not published; using the tag v$VERSION"
    unpack_tag
fi

echo "==> Building $NAME $VERSION"
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
