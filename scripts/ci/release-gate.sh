#!/bin/bash
# Decide whether the checked-out VERSION should be tagged and released,
# and if so assert that it is fit to be.
#
# Usage: ./scripts/ci/release-gate.sh [expected-version]
#
# Eligibility and readiness are separate questions. Not being a release
# is reported as release=false and exit 0, because most pushes that touch
# VERSION are not releases; a release with unfinished notes fails loudly,
# because that one is broken.
#
# The split also means releasing does not rely on release-check having
# blocked the PR: an unfinished release that reached the branch anyway
# stops here.
#
# expected-version guards a dispatched run against the wrong commit,
# and DISPATCHED=true adds the check that the commit is merged.
# Writes release= and version= to $GITHUB_OUTPUT when set.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

require_repo_root

EXPECTED="${1:-}"
VERSION="$(project_version)"

emit() {
    log "verdict: release=$1 version=$VERSION"
    [[ -z "${GITHUB_OUTPUT:-}" ]] ||
        printf 'release=%s\nversion=%s\n' "$1" "$VERSION" >>"$GITHUB_OUTPUT"
}

validate_version "$VERSION"

if [[ -n "$EXPECTED" && "$EXPECTED" != "$VERSION" ]]; then
    die "dispatched for $EXPECTED but VERSION is $VERSION -- check the" \
        "commit this run was started from"
fi

# A dispatched run names its own commit, so it has to be established
# that the commit is one a release may come from. Without this, a
# dispatch from an unmerged chore/release-* branch would tag and publish
# it: VERSION there has no -dev suffix and the notes are finished, so
# every other check passes.
if [[ "${DISPATCHED:-}" == true ]]; then
    head="$(git rev-parse HEAD)"
    on_release_branch=false
    for branch in $(git for-each-ref --format='%(refname)' \
        'refs/remotes/origin/main' \
        'refs/remotes/origin/[0-9]*.[0-9]*.x'); do
        if git merge-base --is-ancestor "$head" "$branch"; then
            log "$head is on ${branch#refs/remotes/origin/}"
            on_release_branch=true
            break
        fi
    done
    [[ "$on_release_branch" == true ]] ||
        die "$head is not on main or an X.Y.x branch -- a release may" \
            "only come from a merged commit, and dispatching from an" \
            "unmerged release branch would publish it"
fi

# ----------------------------------------------------------------
# Eligibility -- not a release is not a failure
# ----------------------------------------------------------------

if is_dev_version "$VERSION"; then
    log "$VERSION is a development version: nothing to release"
    emit false
    exit 0
fi

if tag_exists_local "$VERSION"; then
    log "v$VERSION already exists: $VERSION has already been released"
    emit false
    exit 0
fi

# ----------------------------------------------------------------
# Readiness -- a broken release fails loudly
# ----------------------------------------------------------------

# The commit must be the one prepare-release.sh made, not merely a
# commit sitting where one did. Between a release and the dev-cycle bump,
# main carries a released VERSION, so any commit merged in that window
# looks like a release to every other check here. The Next-Version
# trailer is what only a release commit has.
log "checking this is a release commit"
next_version="$(git log -1 --format='%(trailers:key=Next-Version,valueonly)' \
    | tr -d '[:space:]')"
[[ -n "$next_version" ]] ||
    die "$(git rev-parse --short HEAD) has no Next-Version trailer, so it" \
        "is not a release commit -- VERSION says $VERSION, but only the" \
        "commit prepare-release.sh writes declares the next cycle"
log "release commit confirmed; next cycle is $next_version"

log "checking CHANGELOG.md has a finished entry for $VERSION"
changelog_has_section "$VERSION" ||
    die "releasing $VERSION, but CHANGELOG.md has no '## [$VERSION]'" \
        "entry -- nothing to use as release notes"

# Captured, not piped: `grep -q` closes the pipe on its first match, and
# pipefail turns that SIGPIPE into a non-zero pipeline, inverting this
# test on any realistically sized section.
section="$(changelog_section "$VERSION")"
if grep -q 'FILL-IN' <<<"$section"; then
    die "CHANGELOG.md entry for $VERSION still has FILL-IN" \
        "placeholders -- the release notes are unfinished"
fi
[[ -n "${section//[[:space:]]/}" ]] ||
    die "CHANGELOG.md entry for $VERSION is empty"
log "release notes are finished ($(wc -l <<<"$section") lines)"

log "checking whether the index format changed since the last release"
check_on_disk_format "$VERSION"

log "$VERSION is ready to tag and release"
emit true
