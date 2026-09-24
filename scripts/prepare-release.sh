#!/bin/bash
# Cut a release PR. Merging that PR is what publishes the release, so this
# script stops at opening it -- a human reviews the notes and merges.
#
# Usage: ./scripts/prepare-release.sh [--dry-run] [--next <version>] <version>
#
# Run it twice. The first run creates the release branch, sets VERSION,
# and writes the CHANGELOG entry with the notes template spliced in. Fill
# in the FILL-IN sections, then run it again on that branch: it checks the
# entry, commits, and opens the PR.
#
#   ./scripts/prepare-release.sh 0.2.0     # branch + changelog scaffold
#   $EDITOR CHANGELOG.md                   # fill in the FILL-IN sections
#   ./scripts/prepare-release.sh 0.2.0     # checks, commit, open the PR
#
# --next sets the development version the cycle reopens at after the
# release lands. It defaults to a minor bump (a patch bump on an X.Y.x
# maintenance branch) and is recorded in the PR body, where a reviewer can
# still change it -- that is how a major bump gets requested.
#
# The full process is documented in docs/release.md.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/release-lib.sh"

require_repo_root

DRY_RUN=0
VERSION=""
NEXT_VERSION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --next)
            [[ $# -ge 2 ]] || die "--next needs a version"
            NEXT_VERSION="$2"; shift 2 ;;
        -*) die "unknown option '$1'" ;;
        *)
            [[ -z "$VERSION" ]] || die "unexpected argument '$1'"
            VERSION="$1"; shift ;;
    esac
done

[[ -n "$VERSION" ]] ||
    die "usage: $0 [--dry-run] [--next <version>] <version>"

validate_version "$VERSION"
! is_prerelease "$VERSION" ||
    log "note: $VERSION is a prerelease; it will be marked as one"

BRANCH="chore/release-$VERSION"
TAG="v$VERSION"
RELEASE_LABEL="release"

run() {
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "would run: $*"
    else
        "$@"
    fi
}

# ----------------------------------------------------------------
# Changelog
# ----------------------------------------------------------------

# git-cliff is only needed when there is a previous release to delta
# against. The first release has none -- and its whole pre-release history
# would drown the entry -- so that one gets the heading alone, written by
# add_changelog_entry itself.
needs_git_cliff() {
    [[ -n "$(git tag --list 'v[0-9]*')" ]]
}

# The version cliff.toml's template was written against, so the error
# below and the mismatch warning cannot drift from the config.
cliff_pinned_version() {
    sed -n 's/^# Pinned version: git-cliff \([0-9][0-9.]*[0-9]\).*$/\1/p' \
        cliff.toml | head -1
}

require_changelog_tool() {
    needs_git_cliff || return 0

    local pin
    pin="$(cliff_pinned_version)"

    if ! command -v git-cliff >/dev/null; then
        cat >&2 <<EOF
ERROR: git-cliff not found.

Releasing $VERSION needs it: every release after the first generates its
CHANGELOG.md commit list from the conventional commit history, and
cliff.toml holds the template that does it.

  Install:  https://git-cliff.org/docs/installation
  Version:  ${pin:-see cliff.toml} (what cliff.toml targets)

The first release needs no generated list, so this is only required once
a v<version> tag exists.
EOF
        exit 1
    fi

    # A different version is usually fine, but the template is written
    # against the pin -- worth saying so rather than silently rendering
    # something else.
    local have
    have="$(git-cliff --version 2>/dev/null | awk '{print $2}')"
    if [[ -n "$pin" && -n "$have" && "$have" != "$pin" ]]; then
        warn "git-cliff $have found, but cliff.toml targets $pin;" \
            "the generated changelog may differ"
    fi
}

# Add the "## [<version>]" heading and the commit delta below it.
add_changelog_entry() {
    if needs_git_cliff; then
        run git-cliff --unreleased --tag "$TAG" --prepend CHANGELOG.md
        return
    fi

    log "first release: no previous tag to delta against, so no" \
        "generated commit list"
    if [[ "$DRY_RUN" == 1 ]]; then
        return 0
    fi
    awk -v ver="$VERSION" -v date="$(date +%Y-%m-%d)" '
        { print }
        NR == 1 && /^# Changelog$/ {
            print ""
            print "## [" ver "] - " date
        }
    ' CHANGELOG.md >CHANGELOG.md.new
    mv CHANGELOG.md.new CHANGELOG.md
}

# Splice the notes template in directly under the version heading,
# stripping its guidance comments.
insert_notes_template() {
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "would insert .release-notes-template.md under the heading"
        return 0
    fi
    local tmpl
    tmpl="$(mktemp)"
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

# ----------------------------------------------------------------
# Second run: check, commit, open the PR
# ----------------------------------------------------------------

# On the release branch, VERSION and CHANGELOG.md are expected to be
# modified; nothing else is.
require_only_release_changes() {
    local dirty
    dirty="$(git status --porcelain |
        grep -vE ' (VERSION|CHANGELOG\.md)$' || true)"
    [[ -z "$dirty" ]] || {
        echo "$dirty" >&2
        die "unexpected uncommitted changes on the release branch"
    }
}

pr_body() {
    local next="$1"
    cat <<EOF
Releases \`$VERSION\`. Merging this PR publishes it: the version landing
on \`$(git rev-parse --abbrev-ref "$BASE_BRANCH")\` without a prerelease
suffix is what triggers the release pipeline.

The release notes are the \`## [$VERSION]\` entry in \`CHANGELOG.md\` --
review them there, since they are published verbatim to the GitHub
release page.

<!-- next-version: the development version the cycle reopens at once
     this merges. Change it to request a different bump, for example a
     major one. Keep the backticks. -->
Next version: \`$next\`
EOF
}

open_pr() {
    local next="$1" milestone args=()

    ensure_label "$RELEASE_LABEL" 'bfd4f2' \
        'Release PR: merging it publishes a release'

    args=(--base "$BASE_BRANCH" --head "$BRANCH"
        --title "chore: release $VERSION"
        --label "$RELEASE_LABEL")

    if milestone="$(resolve_milestone "$VERSION")"; then
        log "milestone: $milestone"
        args+=(--milestone "$milestone")
    else
        warn "no milestone matches $VERSION (tried \"Release $VERSION\"" \
            "and \"Release $(strip_prerelease "$VERSION")\") -- opening" \
            "the PR without one"
    fi

    run git push -u origin "$BRANCH"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "would run: gh pr create ${args[*]} --body <body>"
        return 0
    fi
    gh pr create "${args[@]}" --body "$(pr_body "$next")"
}

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

BASE_BRANCH="$(git branch --show-current)"

if [[ "$BASE_BRANCH" == "$BRANCH" ]]; then
    # Second run: the branch exists and the notes should be filled in.
    BASE_BRANCH="$(git config "branch.$BRANCH.release-base" ||
        echo main)"
    require_only_release_changes

    if changelog_section "$VERSION" | grep -q 'FILL-IN'; then
        die "CHANGELOG.md entry for $VERSION still has FILL-IN" \
            "placeholders -- fill them in, then re-run"
    fi

    [[ -n "$NEXT_VERSION" ]] ||
        NEXT_VERSION="$(next_dev_version "$VERSION" "$BASE_BRANCH")"
    validate_version "$NEXT_VERSION"
    is_prerelease "$NEXT_VERSION" ||
        die "next version '$NEXT_VERSION' must carry a prerelease" \
            "suffix (e.g. ${NEXT_VERSION}-dev)"
    log "next development version: $NEXT_VERSION (recorded in the PR" \
        "body; edit it there to change the bump)"

    # Guards are read-only, so they run even under --dry-run: that is
    # what makes a dry run a real preflight rather than an echo.
    log "running the release guards"
    ./scripts/ci/release-guards.sh "$VERSION"

    run git add VERSION CHANGELOG.md
    run git commit -m "chore: release $VERSION"
    open_pr "$NEXT_VERSION"
    log "release PR opened. Merging it publishes $VERSION."
    exit 0
fi

# First run: set the branch up.
require_clean_tree
require_branch_up_to_date "$BASE_BRANCH"
require_tag_absent "$VERSION"
# Before creating the branch or touching VERSION: a missing tool here
# would otherwise leave a half-made release branch to clean up by hand.
require_changelog_tool

current="$(project_version)"
[[ "$current" != "$VERSION" ]] ||
    die "VERSION is already $VERSION -- nothing to bump"
is_prerelease "$current" ||
    die "VERSION is '$current', which is not a development version;" \
        "expected a prerelease suffix to release from"

log "creating $BRANCH from $BASE_BRANCH"
run git checkout -b "$BRANCH"
run git config "branch.$BRANCH.release-base" "$BASE_BRANCH"

log "setting VERSION to $VERSION (was $current)"
if [[ "$DRY_RUN" == 1 ]]; then
    echo "would write VERSION=$VERSION"
else
    set_version "$VERSION"
fi

log "adding the CHANGELOG.md entry for $VERSION"
add_changelog_entry
insert_notes_template

log "next: fill in the FILL-IN sections of the new CHANGELOG.md entry,"
log "then re-run to check, commit and open the PR:"
log "  $0 $VERSION"
