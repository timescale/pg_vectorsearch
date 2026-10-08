#!/bin/bash
# Point a release PR's reviewer at the source tarball the run packaged,
# and ask them to build the extension from it.
#
# Usage: ./scripts/ci/release-tarball-comment.sh
#
# Environment: GH_TOKEN, REPO, PR, VERSION, ARTIFACT_URL, HEAD_SHA.
#
# One comment, updated in place: a release PR is re-pushed whenever the
# notes are amended, and a fresh comment each time would bury the rest
# of the review. The packaged commit is named so a reviewer can tell
# whether the archive is the one their latest push produced.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

: "${REPO:?REPO is required}"
: "${PR:?PR is required}"
: "${VERSION:?VERSION is required}"
: "${ARTIFACT_URL:?ARTIFACT_URL is required}"
: "${HEAD_SHA:?HEAD_SHA is required}"

MARKER='<!-- release-tarball -->'

# The heredoc lives in a function rather than inside the command
# substitution below. bash 3.2, which is what macOS ships, scans a
# $( ) for quote characters before it handles a heredoc inside it, so
# an apostrophe in the prose below reads as an unterminated string and
# the script fails to parse.
comment_body() {
    cat <<BODY
$MARKER
### Build from the tarball before merging

Here's the source archive that will become the \`v$VERSION\` release
asset. CI unpacked it and put it through a full build: it configures,
compiles, installs and passes its own test suite from the archive.

As an extra check, download the tarball and compile it against a local
PostgreSQL:

1. Download **[release-tarball-$VERSION]($ARTIFACT_URL)** and unpack it.
2. Build and install it against your own PostgreSQL 18:

   \`\`\`bash
   tar -xzf pg_vectorsearch-$VERSION.tar.gz
   cd pg_vectorsearch-$VERSION
   meson setup builddir -Dpostgresql=enabled
   meson compile -C builddir && meson install -C builddir
   \`\`\`

3. \`CREATE EXTENSION pg_vectorsearch;\` and check that
   \`pg_vectorsearch_version()\` returns \`$VERSION\`.

Packaged from \`${HEAD_SHA:0:12}\`.
BODY
}

body="$(comment_body)"

existing="$(gh api --paginate \
    "repos/$REPO/issues/$PR/comments" \
    --jq "[.[] | select(.body | startswith(\"$MARKER\"))] | .[0].id // empty")"

if [[ -n "$existing" ]]; then
    gh api --method PATCH "repos/$REPO/issues/comments/$existing" \
        -f body="$body" >/dev/null
    log "updated the tarball comment on #$PR"
else
    gh api --method POST "repos/$REPO/issues/$PR/comments" \
        -f body="$body" >/dev/null
    log "commented on #$PR with the tarball"
fi
