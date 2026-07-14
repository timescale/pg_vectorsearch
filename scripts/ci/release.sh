#!/bin/bash
# The release pipeline. One script, identical in CI and locally: the
# release workflow's jobs each run a single step of it, and running it
# with no step arguments performs the whole release from the current
# checkout (the local fallback when GitHub Actions is unavailable).
#
# Usage: ./scripts/ci/release.sh <version> [step...]
#
# Steps (default: all, in this order):
#   verify    guards (version match, tag absent, docs freshness) and
#             a full build + test run via scripts/ci/build.sh
#   package   source tarball + sha256 + release-notes file into dist/
#   publish   THE RELEASE: gh release create (tag + release + notes,
#             nothing else — follow-up failures can't break it)
#   upload    attach dist/ artifacts to the release page (idempotent)
#   announce  best-effort propagation to enabled surfaces
#             (discussion/wiki); skipped surfaces are logged, and a
#             failure here fails only this step, never the release
#
# Environment:
#   RELEASE_TARGET_SHA  commit to tag (default: HEAD) — CI passes the
#                       SHA its verify job tested, so publish can never
#                       tag a different commit than was verified
#   BUILDDIR            build directory for verify (default:
#                       builddir-release)
#   PG_CONFIG           forwarded to scripts/ci/build.sh

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

require_repo_root

VERSION="${1:-}"
[[ -n "$VERSION" ]] || die "usage: $0 <version> [step...]"
shift
STEPS=("$@")
[[ ${#STEPS[@]} -gt 0 ]] || STEPS=(verify package publish upload announce)

TAG="v$VERSION"
DIST="dist"
EXTENSION_NAME="$(meson_project_name)"
TARBALL="$DIST/$EXTENSION_NAME-$VERSION.tar.gz"
NOTES="$DIST/release-notes-$VERSION.md"
TARGET_SHA="${RELEASE_TARGET_SHA:-$(git rev-parse HEAD)}"
BUILDDIR="${BUILDDIR:-builddir-release}"

# Version-shaped strings in user-facing docs that are allowed to differ
# from the release version (historical or example mentions). One
# extended regex per line.
DOCS_VERSION_ALLOWLIST=(
    'PostgreSQL [0-9]+'
    'pgvector [0-9]+\.[0-9]+'
)

step_verify() {
    log "verify: version and tag"
    validate_version "$VERSION"
    local mv
    mv="$(meson_version)"
    [[ "$mv" == "$VERSION" ]] ||
        die "meson.build version is '$mv', expected '$VERSION'" \
            "(release PR not merged?)"
    git rev-parse -q --verify "refs/tags/$TAG" >/dev/null &&
        die "tag $TAG already exists"

    log "verify: changelog entry"
    changelog_has_section "$VERSION" ||
        die "CHANGELOG.md has no '## [$VERSION]' entry"
    if changelog_section "$VERSION" | grep -q 'FILL-IN'; then
        die "CHANGELOG.md entry for $VERSION has unfilled FILL-IN" \
            "placeholders"
    fi

    log "verify: docs freshness"
    local pattern="${EXTENSION_NAME}[- ][0-9]+\\.[0-9]+\\.[0-9]+[a-z0-9.-]*"
    local hits
    hits="$(grep -rhoE "$pattern" README.md docs/ 2>/dev/null |
        grep -v -- "$VERSION" || true)"
    for allowed in "${DOCS_VERSION_ALLOWLIST[@]}"; do
        hits="$(grep -vE "$allowed" <<<"$hits" || true)"
    done
    if [[ -n "$hits" ]]; then
        echo "$hits" >&2
        die "stale version references in README.md/docs/ (fix them or" \
            "extend DOCS_VERSION_ALLOWLIST in $0)"
    fi

    log "verify: full build and test run"
    ./scripts/ci/build.sh "$BUILDDIR" -Dtools=enabled
}

step_package() {
    log "package: source tarball"
    mkdir -p "$DIST"
    git archive --format=tar.gz \
        --prefix="$EXTENSION_NAME-$VERSION/" \
        -o "$TARBALL" "$TARGET_SHA"
    (cd "$DIST" && sha256sum "$(basename "$TARBALL")" \
        >"$(basename "$TARBALL").sha256")

    log "package: release notes from CHANGELOG.md"
    changelog_section "$VERSION" >"$NOTES"
    [[ -s "$NOTES" ]] || die "empty release notes for $VERSION"
    log "package: wrote $TARBALL and $NOTES"
}

step_publish() {
    log "publish: creating tag $TAG and GitHub release at $TARGET_SHA"
    local args=(
        "$TAG"
        --target "$TARGET_SHA"
        --title "$EXTENSION_NAME $VERSION"
        --notes-file "$NOTES"
    )
    if is_prerelease "$VERSION"; then
        args+=(--prerelease)
    fi
    gh release create "${args[@]}"
    log "publish: released $TAG"
}

step_upload() {
    log "upload: attaching artifacts to $TAG"
    gh release upload "$TAG" --clobber \
        "$TARBALL" "$TARBALL.sha256"
}

# Announce failures must not fail the release; each surface is
# feature-detected and skipped with a log line when absent.
step_announce() {
    local slug
    slug="$(repo_slug)"

    log "announce: release discussion"
    if [[ "$(gh api "repos/$slug" --jq .has_discussions)" == "true" ]]
    then
        local cat_id repo_id
        repo_id="$(gh api "repos/$slug" --jq .node_id)"
        # shellcheck disable=SC2016  # GraphQL $vars, not shell vars
        cat_id="$(gh api graphql -f query='
            query($owner: String!, $name: String!) {
              repository(owner: $owner, name: $name) {
                discussionCategories(first: 25) {
                  nodes { id name }
                }
              }
            }' -f owner="${slug%/*}" -f name="${slug#*/}" \
            --jq '.data.repository.discussionCategories.nodes[]
                  | select(.name == "Announcements") | .id' || true)"
        if [[ -n "$cat_id" ]]; then
            # shellcheck disable=SC2016  # GraphQL $vars, not shell vars
            gh api graphql -f query='
                mutation($repo: ID!, $cat: ID!, $title: String!,
                         $body: String!) {
                  createDiscussion(input: {repositoryId: $repo,
                      categoryId: $cat, title: $title, body: $body}) {
                    discussion { url }
                  }
                }' -f repo="$repo_id" -f cat="$cat_id" \
                -f title="$EXTENSION_NAME $VERSION released" \
                -f body="$(cat "$NOTES")" \
                --jq .data.createDiscussion.discussion.url ||
                warn "creating the announcement discussion failed"
        else
            log "announce: no 'Announcements' category — skipping" \
                "discussion"
        fi
    else
        log "announce: discussions not enabled — skipping"
    fi

    log "announce: wiki release-notes page"
    if [[ "$(gh api "repos/$slug" --jq .has_wiki)" == "true" ]]; then
        local wikidir
        wikidir="$(mktemp -d)"
        if git clone --depth 1 "https://github.com/$slug.wiki.git" \
            "$wikidir" 2>/dev/null; then
            {
                echo "# $EXTENSION_NAME $VERSION"
                echo
                cat "$NOTES"
            } >"$wikidir/Release-$VERSION.md"
            git -C "$wikidir" add "Release-$VERSION.md"
            git -C "$wikidir" -c user.name="release-bot" \
                -c user.email="noreply@github.com" \
                commit -m "Release notes for $VERSION" >/dev/null
            git -C "$wikidir" push ||
                warn "pushing the wiki release page failed"
        else
            log "announce: wiki enabled but not initialized — skipping"
        fi
        rm -rf "$wikidir"
    else
        log "announce: wiki not enabled — skipping"
    fi

    log "announce: GitHub Pages"
    if gh api "repos/$slug/pages" >/dev/null 2>&1; then
        log "announce: Pages is enabled but has no automated" \
            "release-notes integration yet — publish manually if wanted"
    else
        log "announce: Pages not enabled — skipping"
    fi
}

for step in "${STEPS[@]}"; do
    case "$step" in
        verify)   step_verify ;;
        package)  step_package ;;
        publish)  step_publish ;;
        upload)   step_upload ;;
        announce) step_announce ;;
        *) die "unknown step '$step'" ;;
    esac
done

log "done: ${STEPS[*]}"
