#!/bin/bash
# Create the git tag v<version> in this repository, and a GitHub
# Releases entry for it carrying the CHANGELOG.md notes.
#
# Usage: ./scripts/ci/release-create.sh <version> <sha>
#
# Creates exactly two things and uploads no files: one git ref
# (refs/tags/v<version>) and one GitHub Releases entry. Attaching
# artifacts is a separate step, so a failure there cannot damage a
# release that already exists.
#
# <sha> is the commit the build job tested, and must be what is checked
# out: a release may only tag a verified commit, and passing the sha in
# makes that checkable rather than assumed.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

require_repo_root

[[ $# -eq 2 ]] || die "usage: $0 <version> <sha>"

VERSION="$1"
TARGET_SHA="$2"
TAG="v$VERSION"

validate_version "$VERSION"

[[ "$VERSION" == "$(project_version)" ]] ||
    die "asked to release $VERSION but VERSION says $(project_version)"

HEAD_SHA="$(git rev-parse HEAD)"
[[ "$HEAD_SHA" == "$TARGET_SHA" ]] ||
    die "HEAD is $HEAD_SHA but the tested commit was $TARGET_SHA --" \
        "refusing to tag a commit that was not built and tested"

# The tag is created as its own step, and its sha verified, before the
# release is made. `gh release create --target` only places the tag when
# it does not already exist: if one appears in between, the release
# silently attaches to that tag instead and can point somewhere other
# than the commit that was tested.
#
# Creating the ref through the API fails when it already exists, which
# is what makes the check and the creation one decision rather than two.
existing="$(gh api "repos/$(repo_slug)/git/ref/tags/$TAG" \
    --jq .object.sha 2>/dev/null)" || existing=""

if [[ -n "$existing" ]]; then
    # Resumable: a run that tagged and then failed before creating the
    # release must be able to finish. Any other sha is fatal.
    [[ "$existing" == "$TARGET_SHA" ]] ||
        die "tag $TAG already exists at $existing, not the tested" \
            "commit $TARGET_SHA -- refusing to release from it"
    warn "tag $TAG already exists at $TARGET_SHA: resuming"
else
    log "creating tag $TAG at $TARGET_SHA"
    gh api -X POST "repos/$(repo_slug)/git/refs" \
        -f "ref=refs/tags/$TAG" -f "sha=$TARGET_SHA" \
        --jq '"created \(.ref)"'

    # Read it back rather than trusting the write: the release must
    # describe the commit that was tested, not whatever the ref holds.
    placed="$(gh api "repos/$(repo_slug)/git/ref/tags/$TAG" \
        --jq .object.sha)"
    [[ "$placed" == "$TARGET_SHA" ]] ||
        die "tag $TAG resolved to $placed, not $TARGET_SHA"
fi

NOTES="$(mktemp)"
trap 'rm -f "$NOTES"' EXIT
changelog_section "$VERSION" >"$NOTES"
[[ -s "$NOTES" ]] || die "no release notes for $VERSION in CHANGELOG.md"
log "notes: $(wc -l <"$NOTES") lines from the CHANGELOG.md entry"

# --verify-tag, not --target: the tag exists by now and was checked
# against TARGET_SHA, so the release must attach to it or fail rather
# than create one of its own.
args=(--repo "$(repo_slug)"
    --verify-tag
    --title "$(meson_project_name) $VERSION"
    --notes-file "$NOTES")

# A prerelease is marked as one on GitHub, so it does not become the
# "latest release" that people and install scripts land on.
if is_prerelease "$VERSION"; then
    log "$VERSION is a prerelease: marking the release accordingly"
    args+=(--prerelease)
fi

log "creating the GitHub Releases entry for $TAG"
gh release create "$TAG" "${args[@]}"

log "created: $(gh release view "$TAG" --repo "$(repo_slug)" \
    --json url --jq .url)"

# Reported, never acted on: closing a milestone that still has open work
# is a judgement call, not something a release script should make.
if milestone="$(resolve_milestone "$VERSION")"; then
    open_issues="$(gh api --paginate \
        "repos/$(repo_slug)/milestones" \
        --jq ".[] | select(.title == \"$milestone\") | .open_issues" \
        2>/dev/null | head -1)"
    if [[ -n "$open_issues" && "$open_issues" != 0 ]]; then
        warn "milestone \"$milestone\" still has $open_issues open" \
            "issue(s) -- closing it is a human decision"
    else
        log "milestone \"$milestone\" has no open issues left"
    fi
fi
