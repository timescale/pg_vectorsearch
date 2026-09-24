#!/bin/bash
# Decide whether a pull request is a release and, if it is, assert that
# it is well formed.
#
# Usage: ./scripts/ci/release-check.sh [base-ref]
#
# A release PR is defined by what it does: it moves VERSION to a value
# with no prerelease suffix. Keying on the branch name would let a release
# cut from a differently-named branch skip its checks silently, on the one
# PR where that matters most.
#
# Anything else reports success without checking further -- a definite
# result either way is what lets this be a required status check.
#
# Without a base-ref it checks the current checkout instead of a PR.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

require_repo_root

BASE="${1:-}"

# Version-shaped strings in docs allowed to differ from the release
# version. One extended regex per line.
DOCS_VERSION_ALLOWLIST=(
    'PostgreSQL [0-9]+'
    'pgvector [0-9]+\.[0-9]+'
)

# ----------------------------------------------------------------
# Is this a release?
# ----------------------------------------------------------------

VERSION="$(project_version)"

# Validated before it is classified, not after. is_dev_version only looks
# at the suffix, so a malformed VERSION like "not-a-version" would read as
# a development version and skip every assertion below.
validate_version "$VERSION"

if [[ -n "$BASE" ]]; then
    git rev-parse -q --verify "$BASE" >/dev/null ||
        die "base ref '$BASE' not found (was the checkout shallow?)"

    if git diff --quiet "$BASE"...HEAD -- VERSION; then
        log "VERSION unchanged: not a release PR"
        exit 0
    fi

    # The PR introducing VERSION has a base without it, and `git show`
    # on a missing path exits 128 -- which would fail this check for a
    # reason unrelated to the release.
    base_version="$(git show "$BASE:VERSION" 2>/dev/null |
        tr -d '[:space:]')" || base_version=""
    : "${base_version:=(absent)}"

    if is_dev_version "$VERSION"; then
        log "VERSION moves $base_version -> $VERSION, a development" \
            "version: not a release PR"
        exit 0
    fi
    log "VERSION moves $base_version -> $VERSION: checking the release"
elif is_dev_version "$VERSION"; then
    # "Not a release" is success here too; a development checkout is the
    # expected case.
    log "VERSION is $VERSION, a development version: nothing to check"
    exit 0
else
    log "checking release $VERSION in the current checkout"
fi

# ----------------------------------------------------------------
# Assertions
# ----------------------------------------------------------------

log "check: tag absent"
require_tag_absent "$VERSION"

log "check: changelog entry"
changelog_has_section "$VERSION" ||
    die "CHANGELOG.md has no '## [$VERSION]' entry"

# Captured, not piped into grep: `grep -q` closes the pipe on its first
# match, and under `set -o pipefail` the SIGPIPE that kills
# changelog_section upstream makes the whole pipeline non-zero -- so the
# `if` would not fire and unfinished notes would pass this gate. Only
# once the section exceeds a pipe buffer, which a real one does.
section="$(changelog_section "$VERSION")"
if grep -q 'FILL-IN' <<<"$section"; then
    die "CHANGELOG.md entry for $VERSION still has FILL-IN" \
        "placeholders -- the release notes are unfinished"
fi

log "check: docs freshness"
name="$(meson_project_name)"
pattern="${name}[- ][0-9]+\\.[0-9]+\\.[0-9]+[a-z0-9.-]*"
hits="$(grep -rhoE "$pattern" README.md docs/ 2>/dev/null |
    grep -v -- "$VERSION" || true)"
for allowed in "${DOCS_VERSION_ALLOWLIST[@]}"; do
    hits="$(grep -vE "$allowed" <<<"$hits" || true)"
done
if [[ -n "$hits" ]]; then
    echo "$hits" >&2
    die "stale version references in README.md/docs/ (fix them, or" \
        "extend DOCS_VERSION_ALLOWLIST in $0)"
fi

log "check: on-disk format"
check_on_disk_format "$VERSION"

log "check: $VERSION looks releasable"
