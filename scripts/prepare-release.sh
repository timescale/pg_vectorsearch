#!/bin/bash
# Cut a release PR.
#
# Usage: ./scripts/prepare-release.sh [options]
#
#   --version X.Y.Z         the version to release
#   --next-version X.Y.Z-s  the development version the cycle reopens at
#   --dry-run               show what would happen, change nothing
#
# Both versions are optional and are proposed from ./VERSION: the release
# is that version with its prerelease suffix removed, and the next cycle
# is a minor bump (a patch bump on an X.Y.x maintenance branch). On a
# terminal you are asked to confirm or edit them; without one the
# defaults are taken, so automation needs no flags.
#
# The release notes are *not* completed here. The PR is opened with the
# template's FILL-IN placeholders still in it, to be finished in the PR
# where they get reviewed -- and .github/workflows/release-check.yml
# fails while any remain, so an unfinished release cannot merge.
#
# See docs/release.md.

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
        --version)
            [[ $# -ge 2 ]] || die "--version needs a value"
            VERSION="$2"; shift 2 ;;
        --next-version)
            [[ $# -ge 2 ]] || die "--next-version needs a value"
            NEXT_VERSION="$2"; shift 2 ;;
        -h | --help)
            sed -n '2,21p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) die "unexpected argument '$1' (see --help)" ;;
    esac
done

run() {
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "would run: $*"
    else
        "$@"
    fi
}

# Ask for a value, offering a default. Without a terminal the default is
# taken silently, which is what makes this usable from automation. The
# prompt goes to stderr, so the answer can be captured from stdout.
ask() {
    local prompt="$1" default="$2" reply
    if [[ ! -t 0 ]]; then
        echo "$default"
        return 0
    fi
    read -r -p "$prompt [$default]: " reply </dev/tty
    echo "${reply:-$default}"
}

confirm() {
    [[ -t 0 ]] || return 0
    local reply
    read -r -p "$1 [y/N] " reply </dev/tty
    [[ "$reply" == [yY]* ]] || die "aborted"
}

# git-cliff is only needed when there is a previous release to delta
# against; the first release's entry is written below instead.
needs_git_cliff() {
    [[ -n "$(previous_release_tag)" ]]
}

# The version cliff.toml's template targets, so the messages below cannot
# drift from the config.
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

    local have
    have="$(git-cliff --version 2>/dev/null | awk '{print $2}')"
    if [[ -n "$pin" && -n "$have" && "$have" != "$pin" ]]; then
        warn "git-cliff $have found, but cliff.toml targets $pin;" \
            "the generated changelog may differ"
    fi
}

# ----------------------------------------------------------------
# Changelog
# ----------------------------------------------------------------

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

# Splice the notes template in under the version heading, stripping its
# guidance comments. The FILL-IN placeholders stay: they are completed in
# the PR, and release-check.yml fails while any remain.
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

# A format bump means every existing index is rejected at open, which the
# notes have to say. Written here rather than left to the author: the
# script can see it, and it is the one upgrade note that is never
# optional.
note_format_change() {
    local prev now was note
    prev="$(previous_release_tag)"
    [[ -n "$prev" ]] || return 0
    now="$(meta_format_version)"
    was="$(meta_format_version "$prev")"
    [[ -n "$now" && -n "$was" && "$now" != "$was" ]] || return 0

    log "on-disk format changed since $prev: adding a reindex note"
    if [[ "$DRY_RUN" == 1 ]]; then
        return 0
    fi
    note="- **Indexes must be rebuilt.** The on-disk index format changed"
    note="$note (0x$was -> 0x$now), so indexes built by earlier versions"
    note="$note are rejected at open and have to be recreated."
    awk -v note="$note" '
        /^FILL-IN: breaking changes/ { print note; print "" }
        { print }
    ' CHANGELOG.md >CHANGELOG.md.new
    mv CHANGELOG.md.new CHANGELOG.md
}

# ----------------------------------------------------------------
# The PR
# ----------------------------------------------------------------

pr_body() {
    cat <<EOF
Releases \`$VERSION\`.

**The release notes are unfinished.** Complete the \`## [$VERSION]\` entry
in \`CHANGELOG.md\` in this PR: every \`FILL-IN\` placeholder has to go,
and the generated \`### Changes\` list is raw material -- trim it to the
notable items. The \`release-check\` job fails while any placeholder
remains, so this cannot merge until the notes are done. What lands here
is published verbatim as the release notes.

Merging this PR publishes \`$VERSION\`: \`VERSION\` landing on
\`$BASE_BRANCH\` without a prerelease suffix is what triggers the release
pipeline.

<!-- next-version: the development version the cycle reopens at once this
     merges. Change it to request a different bump, for example a major
     one. Keep the backticks. -->
Next version: \`$NEXT_VERSION\`
EOF
}

open_pr() {
    local milestone args=()

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
    gh pr create "${args[@]}" --body "$(pr_body)"
}

# ----------------------------------------------------------------
# Resolve the versions
# ----------------------------------------------------------------

BASE_BRANCH="$(git branch --show-current)"
CURRENT="$(project_version)"

is_prerelease "$CURRENT" ||
    die "VERSION is '$CURRENT', which is not a development version;" \
        "expected a prerelease suffix to release from"

[[ -n "$VERSION" ]] ||
    VERSION="$(ask "Release version" "$(strip_prerelease "$CURRENT")")"
validate_version "$VERSION"
[[ "$VERSION" != "$CURRENT" ]] ||
    die "VERSION is already $VERSION -- nothing to release"

[[ -n "$NEXT_VERSION" ]] ||
    NEXT_VERSION="$(ask "Next development version" \
        "$(next_dev_version "$VERSION" "$BASE_BRANCH")")"
validate_version "$NEXT_VERSION"
is_prerelease "$NEXT_VERSION" ||
    die "next version '$NEXT_VERSION' must carry a prerelease suffix" \
        "(e.g. ${NEXT_VERSION}-dev)"

BRANCH="chore/release-$VERSION"
TAG="v$VERSION"
RELEASE_LABEL="release"

# ----------------------------------------------------------------
# Preconditions -- all of them before anything is mutated
# ----------------------------------------------------------------

require_clean_tree
require_branch_up_to_date "$BASE_BRANCH"
require_tag_absent "$VERSION"
require_changelog_tool

# An incompatible format change cannot ship in a patch release; catching
# it here rather than in CI saves cutting a PR that cannot merge.
check_on_disk_format "$VERSION"

log "releasing   $CURRENT -> $VERSION on $BASE_BRANCH"
log "next cycle  $NEXT_VERSION"
log "branch      $BRANCH"
confirm "Create the release branch and open its PR?"

# ----------------------------------------------------------------
# Go
# ----------------------------------------------------------------

log "creating $BRANCH"
run git checkout -b "$BRANCH"

log "setting VERSION to $VERSION"
if [[ "$DRY_RUN" == 1 ]]; then
    echo "would write VERSION=$VERSION"
else
    set_version "$VERSION"
fi

log "writing the CHANGELOG.md entry"
add_changelog_entry
insert_notes_template
note_format_change

run git add VERSION CHANGELOG.md
run git commit -m "chore: release $VERSION"
open_pr

log "done. Complete the release notes in the PR; it cannot merge while"
log "any FILL-IN placeholder remains."
