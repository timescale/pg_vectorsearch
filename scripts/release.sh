#!/bin/bash
# Operator-facing release driver. Walks the release process phase by
# phase; the actual release pipeline lives in scripts/ci/release.sh
# (run by CI, or locally as fallback). The full manual runbook is
# docs/release.md.
#
# Usage: ./scripts/release.sh <command> [options] <version>
#
# Commands:
#   prepare <version>   start the release branch: bump the version,
#                       add the CHANGELOG.md entry from the template,
#                       then (re-run after filling it in) run checks
#                       and commit. Merging the resulting PR is the
#                       human approval that publishes the release.
#   publish <version>   fallback only — normally the release workflow
#                       fires on the release PR's merge. Re-dispatches
#                       the workflow (--local: run the pipeline here
#                       instead, for when Actions is unavailable).
#   post <version>      open the next development cycle
#                       (e.g. post 0.2.0-dev) on a new branch
#   verify <version>    download the published artifacts, check them,
#                       and build from the tarball
#
# Options:
#   --dry-run   print what would happen without changing anything
#   --local     (publish) run scripts/ci/release.sh here instead of
#               dispatching the release workflow

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/release-lib.sh"

require_repo_root

DRY_RUN=0
LOCAL=0
COMMAND=""
VERSION=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --local) LOCAL=1 ;;
        -*) die "unknown option '$arg'" ;;
        *)
            if [[ -z "$COMMAND" ]]; then COMMAND="$arg"
            elif [[ -z "$VERSION" ]]; then VERSION="$arg"
            else die "unexpected argument '$arg'"
            fi
            ;;
    esac
done
[[ -n "$COMMAND" && -n "$VERSION" ]] ||
    die "usage: $0 <prepare|publish|post|verify> [options] <version>"

TAG="v$VERSION"
EXTENSION_NAME="$(meson_project_name)"

run() {
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "would run: $*"
    else
        "$@"
    fi
}

require_clean_tree() {
    [[ -z "$(git status --porcelain)" ]] ||
        die "working tree is not clean"
}

require_main_up_to_date() {
    [[ "$(git branch --show-current)" == "main" ]] ||
        die "not on branch 'main'"
    git fetch origin main --quiet
    [[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] ||
        die "main is not up to date with origin/main"
}

require_tag_absent() {
    git rev-parse -q --verify "refs/tags/$TAG" >/dev/null &&
        die "tag $TAG already exists"
    return 0
}

# All check runs on HEAD must have completed successfully. Matters
# because the CI workflows use paths: filters — the release PR alone
# may not have triggered every suite.
require_ci_green() {
    local sha slug pending failed
    sha="$(git rev-parse HEAD)"
    slug="$(repo_slug)"
    pending="$(gh api "repos/$slug/commits/$sha/check-runs" --paginate \
        --jq '[.check_runs[] | select(.status != "completed")] | length')"
    [[ "$pending" == 0 ]] || die "$pending CI check(s) still running"
    failed="$(gh api "repos/$slug/commits/$sha/check-runs" --paginate \
        --jq '[.check_runs[] | select(.conclusion != null)
               | select([.conclusion] | inside(["success", "neutral",
                                               "skipped"]) | not)]
              | length')"
    [[ "$failed" == 0 ]] || die "$failed CI check(s) not green on HEAD"
}

set_meson_version() {
    local new="$1"
    run sed -i "s/^  version: '[^']*',\$/  version: '$new',/" \
        meson.build
    [[ "$DRY_RUN" == 1 || "$(meson_version)" == "$new" ]] ||
        die "failed to set version in meson.build"
}

# Add the "## [<version>]" changelog entry heading. git-cliff
# generates it together with the "### Changes" delta since the
# previous release tag — except for the first release, which has no
# previous release to delta against (and the full pre-release history
# would drown readers): its entry is the hand-written notes alone,
# under a heading written here (pgvector-style "First release").
add_changelog_entry() {
    if [[ -n "$(git tag --list 'v[0-9]*')" ]]; then
        git-cliff --unreleased --tag "$TAG" --prepend CHANGELOG.md
        return
    fi
    log "first release: no previous release tag, skipping git-cliff"
    awk -v ver="$VERSION" -v date="$(date +%Y-%m-%d)" '
        { print }
        NR == 1 && /^# Changelog$/ {
            print ""
            print "## [" ver "] - " date
        }
    ' CHANGELOG.md >CHANGELOG.md.new
    mv CHANGELOG.md.new CHANGELOG.md
}

# Insert the release-notes template (guidance comments stripped)
# directly under the "## [<version>]" heading git-cliff generated.
insert_notes_template() {
    local tmpl
    tmpl="$(mktemp)"
    # Strip <!-- ... --> guidance comments (possibly multi-line).
    perl -0pe 's/<!--.*?-->\n?//gs' .release-notes-template.md >"$tmpl"
    awk -v ver="$VERSION" -v tmpl="$tmpl" '
        { print }
        $0 ~ "^## \\[" ver "\\]" {
            print ""
            while ((getline line < tmpl) > 0) print line
        }
    ' CHANGELOG.md >CHANGELOG.md.new
    mv CHANGELOG.md.new CHANGELOG.md
    rm -f "$tmpl"
}

cmd_prepare() {
    validate_version "$VERSION"
    require_tag_absent
    local branch="chore/release-$VERSION"

    # Second invocation: the release branch exists and the entry is
    # filled in — run the checks and commit.
    if [[ "$(git branch --show-current)" == "$branch" ]]; then
        require_clean_tree_except_release
        if changelog_section "$VERSION" | grep -q 'FILL-IN'; then
            die "CHANGELOG.md entry for $VERSION still has FILL-IN" \
                "placeholders — fill them in, then re-run"
        fi
        log "running the release verify step (build, tests, docs)"
        run ./scripts/ci/release.sh "$VERSION" verify
        run git add meson.build CHANGELOG.md
        run git commit -m "chore: release $VERSION"
        log "release branch ready — push it and open a PR:"
        log "  git push -u origin $branch"
        log "MERGING THE RELEASE PR PUBLISHES THE RELEASE: the release"
        log "workflow fires on the version flip landing on main."
        return
    fi

    # First invocation: set the release branch up. git-cliff is only
    # needed when there is a previous release to delta against.
    require_clean_tree
    require_main_up_to_date
    if [[ -n "$(git tag --list 'v[0-9]*')" ]]; then
        command -v git-cliff >/dev/null ||
            die "git-cliff not found (see cliff.toml for the" \
                "pinned version)"
    fi

    log "creating $branch"
    run git checkout -b "$branch"
    log "setting meson.build version to $VERSION"
    set_meson_version "$VERSION"
    log "adding the CHANGELOG.md entry for $VERSION"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "would add the CHANGELOG.md entry (git-cliff --unreleased" \
            "--tag $TAG --prepend; heading only for a first release)" \
            "and insert the notes template"
        return
    fi
    add_changelog_entry
    insert_notes_template

    log "next: fill in the FILL-IN sections of the new CHANGELOG.md"
    log "entry (Highlights, upgrade notes, ...), then re-run:"
    log "  $0 prepare $VERSION"
}

# On the release branch the version bump and changelog entry are
# expected to be dirty; anything else is not.
require_clean_tree_except_release() {
    local dirty
    dirty="$(git status --porcelain | grep -vE ' (meson\.build|CHANGELOG\.md)$' || true)"
    [[ -z "$dirty" ]] || {
        echo "$dirty" >&2
        die "unexpected uncommitted changes on the release branch"
    }
}

cmd_publish() {
    validate_version "$VERSION"
    require_clean_tree
    require_main_up_to_date
    require_tag_absent
    local mv
    mv="$(meson_version)"
    [[ "$mv" == "$VERSION" ]] ||
        die "meson.build version is '$mv', expected '$VERSION'" \
            "(release PR not merged?)"
    require_ci_green

    if [[ "$LOCAL" == 1 ]]; then
        log "running the release pipeline locally (fallback path)"
        run ./scripts/ci/release.sh "$VERSION"
        return
    fi

    log "dispatching the release workflow for $VERSION"
    log "(normally unnecessary — merging the release PR triggers it;"
    log "use this to re-run after a transient failure)"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "would run: gh workflow run release.yml -f version=$VERSION"
        return
    fi
    read -r -p "Dispatch the release workflow for $VERSION? [y/N] " a
    [[ "$a" == y || "$a" == Y ]] || die "aborted"
    gh workflow run release.yml -f version="$VERSION"
    sleep 3
    gh run watch "$(gh run list --workflow=release.yml --limit 1 \
        --json databaseId --jq '.[0].databaseId')" || true
}

cmd_post() {
    validate_version "$VERSION"
    is_prerelease "$VERSION" ||
        warn "post-release version '$VERSION' has no -dev suffix"
    require_clean_tree
    require_main_up_to_date
    local branch="chore/open-$VERSION"

    log "creating $branch"
    run git checkout -b "$branch"
    set_meson_version "$VERSION"
    run git add meson.build
    run git commit -m "chore: open $VERSION development"
    log "next: git push -u origin $branch and open a PR"
}

cmd_verify() {
    validate_version "$VERSION"
    local dir="dist/verify-$VERSION"
    local tarball="$EXTENSION_NAME-$VERSION.tar.gz"

    log "downloading release artifacts for $TAG"
    run rm -rf "$dir"
    run mkdir -p "$dir"
    run gh release download "$TAG" --dir "$dir"
    if [[ "$DRY_RUN" == 1 ]]; then return; fi

    log "checking artifact checksum"
    (cd "$dir" && sha256sum -c "$tarball.sha256")

    log "building from the tarball (extension-only by default)"
    tar -C "$dir" -xzf "$dir/$tarball"
    local src="$dir/$EXTENSION_NAME-$VERSION"
    meson setup "$src/build" "$src"
    meson compile -C "$src/build"
    meson test -C "$src/build"

    log "tarball build OK; manual smoke test:"
    log "  meson install -C $src/build"
    log "  psql: CREATE EXTENSION $EXTENSION_NAME;"
    log "        SELECT mkt.extension_version(), mkt.git_commit();"
}

case "$COMMAND" in
    prepare) cmd_prepare ;;
    publish) cmd_publish ;;
    post)    cmd_post ;;
    verify)  cmd_verify ;;
    *) die "unknown command '$COMMAND'" ;;
esac
