#!/bin/bash
# Release guards: the cheap assertions that decide whether a version is
# ready to be released. No build, no network writes -- CI runs this on the
# release PR, and prepare-release.sh runs it locally so the operator hits
# the same failures before pushing.
#
# Usage: ./scripts/ci/release-guards.sh [version]
#
# The version defaults to ./VERSION, which is what the release PR has
# already set. Pass one explicitly to check a version other than the
# checked-out one.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

require_repo_root

VERSION="${1:-$(project_version)}"

# Version-shaped strings in user-facing docs that are allowed to differ
# from the release version (historical or example mentions). One extended
# regex per line.
DOCS_VERSION_ALLOWLIST=(
    'PostgreSQL [0-9]+'
    'pgvector [0-9]+\.[0-9]+'
)

log "guards: version string"
validate_version "$VERSION"

log "guards: VERSION matches"
mv="$(project_version)"
[[ "$mv" == "$VERSION" ]] ||
    die "VERSION is '$mv', expected '$VERSION'"

log "guards: tag absent"
require_tag_absent "$VERSION"

log "guards: changelog entry"
changelog_has_section "$VERSION" ||
    die "CHANGELOG.md has no '## [$VERSION]' entry"
if changelog_section "$VERSION" | grep -q 'FILL-IN'; then
    die "CHANGELOG.md entry for $VERSION still has FILL-IN placeholders"
fi

log "guards: docs freshness"
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

log "guards: $VERSION looks releasable"
