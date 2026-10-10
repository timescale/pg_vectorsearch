#!/bin/bash
# Attach the packaged assets to the draft release for v<version>, then
# publish it.
#
# Usage: ./scripts/ci/release-publish.sh <version> <distdir>
#            [--keep-draft]
#
# --keep-draft attaches the files and stops, leaving the entry
# unpublished for someone to look over and publish by hand. It holds
# back only the publish: the tag is pushed before the draft is even
# created, so the version is cut either way, and everything downstream
# -- the development-cycle bump included -- still runs. Finishing is
# then one click in the Releases UI.
#
# Publishing is what makes a release immutable. GitHub freezes an
# immutable release's assets and its tag at that moment and rejects
# every later upload with "Cannot upload assets to an immutable
# release", so a release published before its files are attached can
# never acquire them -- and deleting it does not help, because a tag
# that has carried an immutable release can never be reused.
#
# One script therefore owns the invariant that nothing is published
# until every file is attached: the publish is unreachable here without
# the verification immediately before it. It reads the upload back from
# the API rather than trusting the upload's exit status, since a missing
# asset is only correctable while the entry is still a draft.
#
# The pipeline runs it twice -- once with --keep-draft to stage the
# assets, and once without, as its final job, to publish. The second run
# re-attaches and re-verifies, so what is sealed is checked by the job
# that seals it.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

require_repo_root

KEEP_DRAFT=""
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --keep-draft) KEEP_DRAFT=1 ;;
        -*) die "unknown option: $arg" ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

[[ ${#POSITIONAL[@]} -eq 2 ]] ||
    die "usage: $0 <version> <distdir> [--keep-draft]"

VERSION="${POSITIONAL[0]}"
DISTDIR="${POSITIONAL[1]}"
TAG="v$VERSION"

# Whether to leave the release unpublished, taken from the most
# specific source that expresses an opinion:
#
#   --keep-draft         the staging call in the release job, which
#                        always holds the draft open
#   KEEP_DRAFT_INPUT     a dispatched run's own choice, yes or no --
#                        whoever started this run outranks any standing
#                        setting, including an explicit no
#   Keep-Draft: trailer  on the release commit, so the decision is made
#                        and reviewed in the pull request that carries
#                        it, and reaches here by the same route the
#                        next version does
#   RELEASE_KEEP_DRAFT   repository variable, the standing policy
#
# Silence at every level publishes. Holding a release back is the
# deliberate act, so it is the one that has to be said out loud.
resolve_keep_draft() {
    local trailer
    case "${KEEP_DRAFT_INPUT:-}" in
        yes) log "this run was dispatched asking for a draft"
            return 0 ;;
        no) log "this run was dispatched asking to publish"
            return 1 ;;
    esac

    trailer="$(commit_trailer Keep-Draft HEAD)"
    if [[ -n "$trailer" ]]; then
        if is_truthy "$trailer"; then
            log "the release commit asks for a draft (Keep-Draft:" \
                "$trailer)"
            return 0
        fi
        log "the release commit asks to publish (Keep-Draft: $trailer)"
        return 1
    fi

    if is_truthy "${KEEP_DRAFT_DEFAULT:-}"; then
        log "RELEASE_KEEP_DRAFT holds every release as a draft"
        return 0
    fi
    return 1
}

if [[ -z "$KEEP_DRAFT" ]] && resolve_keep_draft; then
    KEEP_DRAFT=1
fi

validate_version "$VERSION"

[[ "$VERSION" == "$(project_version)" ]] ||
    die "asked to publish $VERSION but VERSION says $(project_version)"

[[ -d "$DISTDIR" ]] || die "no such directory: $DISTDIR"

# What the build job packaged, by name. These are what must be on the
# release before it is sealed.
EXPECTED=()
while IFS= read -r file; do
    EXPECTED+=("$(basename "$file")")
done < <(find "$DISTDIR" -maxdepth 1 -type f | sort)

[[ ${#EXPECTED[@]} -gt 0 ]] ||
    die "no files to attach in $DISTDIR"

log "attaching ${#EXPECTED[@]} file(s): ${EXPECTED[*]}"

# Asserts every expected name is attached to the release, naming the
# ones that are not. Called before publishing, where a shortfall is
# still fixable, and again for an already-published release, where it
# can only be reported.
require_assets_attached() {
    local attached missing=()
    attached="$(gh release view "$TAG" --repo "$(repo_slug)" \
        --json assets --jq '.assets[].name')"

    local want
    for want in "${EXPECTED[@]}"; do
        grep -qxF "$want" <<<"$attached" || missing+=("$want")
    done

    [[ ${#missing[@]} -eq 0 ]] ||
        die "release $TAG is missing ${#missing[@]} asset(s):" \
            "${missing[*]}"
}

# What this run did and to which files, in the job summary, so the
# contents of a release are recoverable from the run that made it
# without opening the Releases page.
summarize() {
    [[ -n "${GITHUB_STEP_SUMMARY:-}" ]] || return 0
    {
        printf '### %s %s\n\n' "$TAG" "$1"
        printf '%s\n\n' "$(gh release view "$TAG" \
            --repo "$(repo_slug)" --json url --jq .url)"
        printf -- "- \`%s\`\n" "${EXPECTED[@]}"
    } >>"$GITHUB_STEP_SUMMARY"
}

draft="$(gh release view "$TAG" --repo "$(repo_slug)" \
    --json isDraft --jq .isDraft 2>/dev/null)" ||
    die "no release entry for $TAG -- release-create.sh makes the" \
        "draft this step publishes"

if [[ "$draft" == "false" ]]; then
    # Resumable: a run that published and then failed afterwards is
    # finished, and re-running it must not report an error. A published
    # release that is missing files is not resumable at all, and
    # require_assets_attached says so.
    warn "release $TAG is already published: verifying it is complete"
    require_assets_attached
else
    # --clobber so a re-run replaces what a previous attempt attached
    # rather than failing on the name clash. A draft is mutable, which
    # is the whole reason the upload happens before the publish.
    gh release upload "$TAG" "$DISTDIR"/* --clobber --repo "$(repo_slug)"
    require_assets_attached

    if [[ -n "$KEEP_DRAFT" ]]; then
        log "--keep-draft: $TAG keeps its assets but stays unpublished"
        summarize "held as a draft"
        exit 0
    fi

    log "all assets attached: publishing $TAG"
    gh release edit "$TAG" --repo "$(repo_slug)" --draft=false
fi

summarize published

log "published: $(gh release view "$TAG" --repo "$(repo_slug)" \
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
