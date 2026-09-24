#!/bin/bash
# Decide whether a pull request is a release PR, and if it is, run the
# release guards against it.
#
# Usage: ./scripts/ci/release-check.sh <base-ref>
#
# A release PR is defined by what it does, not by what its branch is
# called: it changes VERSION, and the value it lands is a release version
# (no prerelease suffix). Keying on the branch name instead would let a
# release PR cut from a differently-named branch skip its guards without
# anyone noticing.
#
# Anything else -- an ordinary PR, or the PR that reopens a development
# cycle by moving VERSION back to a -dev value -- exits successfully
# without running the guards. Reporting a definite result either way is
# what lets this be a required status check.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

require_repo_root

BASE="${1:-}"
[[ -n "$BASE" ]] || die "usage: $0 <base-ref>"

git rev-parse -q --verify "$BASE" >/dev/null ||
    die "base ref '$BASE' not found (was the checkout shallow?)"

if git diff --quiet "$BASE"...HEAD -- VERSION; then
    log "VERSION unchanged: not a release PR"
    exit 0
fi

version="$(project_version)"
base_version="$(git show "$BASE:VERSION" | tr -d '[:space:]')"

if is_prerelease "$version"; then
    log "VERSION moves $base_version -> $version, which is a" \
        "prerelease: not a release PR"
    exit 0
fi

log "VERSION moves $base_version -> $version: this is a release PR"
exec "$(dirname "$0")/release-guards.sh" "$version"
