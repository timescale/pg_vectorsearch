#!/bin/bash
# The release check: decide whether a pull request is a release, and if it
# is, assert that it is well formed.
#
# Usage: ./scripts/ci/release-check.sh [base-ref]
#
# Runs on every PR (as .github/workflows/release-check.yml) and locally
# against a checkout. A release PR is defined by what it does, not by what
# its branch is called: it changes VERSION, and the value it lands carries
# no prerelease suffix. Keying on a branch name instead would let a
# release PR cut from a differently-named branch skip its checks
# silently, on the one PR where that matters most.
#
# Anything else -- an ordinary PR, or one reopening a development cycle by
# moving VERSION back to a -dev value -- reports success without checking
# further. Reporting a definite result either way is what lets this be a
# required status check.
#
# With no base-ref it checks the current checkout rather than a PR, which
# is how to run the same assertions by hand.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

require_repo_root

BASE="${1:-}"

# Version-shaped strings in user-facing docs that are allowed to differ
# from the release version (historical or example mentions). One extended
# regex per line.
DOCS_VERSION_ALLOWLIST=(
    'PostgreSQL [0-9]+'
    'pgvector [0-9]+\.[0-9]+'
)

# ----------------------------------------------------------------
# Is this a release?
# ----------------------------------------------------------------

VERSION="$(project_version)"

if [[ -n "$BASE" ]]; then
    git rev-parse -q --verify "$BASE" >/dev/null ||
        die "base ref '$BASE' not found (was the checkout shallow?)"

    if git diff --quiet "$BASE"...HEAD -- VERSION; then
        log "VERSION unchanged: not a release PR"
        exit 0
    fi

    base_version="$(git show "$BASE:VERSION" | tr -d '[:space:]')"
    if is_prerelease "$VERSION"; then
        log "VERSION moves $base_version -> $VERSION, a prerelease:" \
            "not a release PR"
        exit 0
    fi
    log "VERSION moves $base_version -> $VERSION: checking the release"
elif is_prerelease "$VERSION"; then
    # Consistent with the PR case: "not a release" is success, not a
    # failure. Running this on a development checkout is expected.
    log "VERSION is $VERSION, a development version: nothing to check"
    exit 0
else
    log "checking release $VERSION in the current checkout"
fi

# ----------------------------------------------------------------
# Assertions
# ----------------------------------------------------------------

log "check: version string"
validate_version "$VERSION"

log "check: tag absent"
require_tag_absent "$VERSION"

log "check: changelog entry"
changelog_has_section "$VERSION" ||
    die "CHANGELOG.md has no '## [$VERSION]' entry"
if changelog_section "$VERSION" | grep -q 'FILL-IN'; then
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
