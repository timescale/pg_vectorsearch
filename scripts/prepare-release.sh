#!/bin/bash
# Draft a release: the changelog entry and the pull request body.
#
# Usage: ./scripts/prepare-release.sh [options]
#
#   --version X.Y.Z           the version to release
#   --next-version X.Y.Z-s    the next development version
#   --ignore-untracked-files  do not ask about untracked files
#   --repo owner/name         release to this repository instead
#
# Both versions default from ./VERSION: the release is it without its
# prerelease suffix, the next cycle a minor bump (patch on an X.Y.x
# branch). Passing all three flags makes the run consult no terminal at
# all, which is what automation needs.
#
# Nothing in the checkout changes. The run checks that a release can
# be cut from it, then writes two files under a temporary directory:
# CHANGELOG.md with the new entry on top -- the commit list git-cliff
# generates, and FILL-IN placeholders for the notes a person writes --
# and body.md, the pull request description. Finish the notes in
# those files, then open the pull request with open-release-pr.sh;
# the run prints the command.
#
# See docs/release.md.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/release-lib.sh"

require_repo_root

IGNORE_UNTRACKED_FILES=0
VERSION=""
NEXT_VERSION=""
REPO_GIVEN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ignore-untracked-files) IGNORE_UNTRACKED_FILES=1; shift ;;
        --repo)
            [[ $# -ge 2 ]] || die "--repo needs an owner/name"
            [[ "$2" == */* ]] || die "--repo wants owner/name, got '$2'"
            RELEASE_REPO="$2"
            VS_RELEASE_REMOTE=""  # drop any remote cached for the old one
            REPO_GIVEN=1
            shift 2 ;;
        --version)
            [[ $# -ge 2 ]] || die "--version needs a value"
            VERSION="$2"; shift 2 ;;
        --next-version)
            [[ $# -ge 2 ]] || die "--next-version needs a value"
            NEXT_VERSION="$2"; shift 2 ;;
        -h | --help)
            sed -n '2,24p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) die "unexpected argument '$1' (see --help)" ;;
    esac
done

# Prompt for a value, defaulting without a terminal. read -p writes to
# stderr, leaving stdout for the answer.
ask() {
    local prompt="$1" default="$2" reply
    if [[ ! -t 0 ]]; then
        echo "$default"
        return 0
    fi
    read -r -p "$prompt [$default]: " reply </dev/tty
    echo "${reply:-$default}"
}

# Yes/no, defaulting to no; non-zero without a terminal rather than
# hanging.
confirm() {
    [[ -t 0 ]] || return 1
    local reply
    read -r -p "$1 [y/N] " reply </dev/tty
    [[ "$reply" == [yY]* ]]
}

# The git-cliff version cliff.toml targets, so messages cannot drift
# from the config.
cliff_pinned_version() {
    sed -n 's/^# Pinned version: git-cliff \([0-9][0-9.]*[0-9]\).*$/\1/p' \
        cliff.toml | head -1
}

require_changelog_tool() {
    local pin
    pin="$(cliff_pinned_version)"

    if ! command -v git-cliff >/dev/null; then
        cat >&2 <<EOF
ERROR: git-cliff not found.

Releasing $VERSION needs it: the CHANGELOG.md commit list is generated
from the conventional commit history, and cliff.toml holds the template
that does it.

  Install:  https://git-cliff.org/docs/installation
  Version:  ${pin:-see cliff.toml} (what cliff.toml targets)
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
# The changelog draft, written to $1
# ----------------------------------------------------------------

# --unreleased means "not reachable from any tag", so with no tags yet
# this covers the whole history -- the first release included.
add_changelog_entry() {
    if [[ -f CHANGELOG.md ]]; then
        cp CHANGELOG.md "$1"
    else
        printf '# Changelog\n' >"$1"
    fi
    git-cliff --unreleased --tag "$TAG" --prepend "$1"
}

# Splice the notes template under the version heading, stripping its
# guidance comments. FILL-IN placeholders stay; the release check fails
# while any remain.
insert_notes_template() {
    local tmpl
    tmpl="$(mktemp)"
    perl -0pe 's/<!--.*?-->\n?//gs' .release-notes-template.md >"$tmpl"
    awk -v ver="$VERSION" -v tmpl="$tmpl" '
        { print }
        $0 ~ "^## \\[" ver "\\]" {
            print ""
            while ((getline line < tmpl) > 0) print line
        }
    ' "$1" >"$1.new"
    mv "$1.new" "$1"
    rm -f "$tmpl"
}

# A format bump makes every existing index unreadable. Written here
# rather than left to the author because the script can detect it.
note_format_change() {
    local prev now was note
    prev="$(previous_release_tag)"
    [[ -n "$prev" ]] || return 0
    now="$(meta_format_version)"
    was="$(meta_format_version "$prev")"
    [[ -n "$now" && -n "$was" && "$now" != "$was" ]] || return 0

    log "on-disk format changed since $prev: adding a reindex note"
    note="- **Indexes must be rebuilt.** The on-disk index format changed"
    note="$note (0x$was -> 0x$now), so indexes built by earlier versions"
    note="$note are rejected at open and have to be recreated."
    awk -v note="$note" '
        /^FILL-IN: breaking changes/ { print note; print "" }
        { print }
    ' "$1" >"$1.new"
    mv "$1.new" "$1"
}

# ----------------------------------------------------------------
# The pull request body
# ----------------------------------------------------------------

# What the PR does, not how releasing works -- that is docs/release.md's
# job. The highlights go in by hand once the notes exist, above the
# trailer.
pr_body() {
    # Backticks here are markdown, not command substitution.
    # shellcheck disable=SC2016
    printf 'Releases `%s`.\n\n' "$VERSION"
    create_next_version_trailer
}

# The next development version, as a git trailer the release pipeline
# reads back with `git interpret-trailers`.
#
# Emitted into both the release commit and the PR body, because which one
# reaches main depends on the merge: rebase keeps the commit, squash
# builds the message from the body (squash_merge_commit_message=PR_BODY).
#
# Must stay the last paragraph and unquoted -- git parses a trailer block
# only at the end of a message, and takes the value literally.
create_next_version_trailer() {
    printf 'Next-Version: %s\n' "$NEXT_VERSION"
}

# ----------------------------------------------------------------
# Repository state -- before anything is asked, so a dirty tree surfaces
# before you pick versions
# ----------------------------------------------------------------

BASE_BRANCH="$(git branch --show-current)"
CURRENT="$(project_version)"

require_clean_tree
review_untracked "$IGNORE_UNTRACKED_FILES" confirm
require_branch_up_to_date "$BASE_BRANCH"

# A rehearsal cut from elsewhere is fine, but its PR is based there too,
# and every workflow filters on `branches: [main]` -- so it gets almost
# no checks, which reads as a green release.
if [[ "$BASE_BRANCH" != main && ! "$BASE_BRANCH" =~ ^[0-9]+\.[0-9]+\.x$ ]]; then
    warn "releasing from '$BASE_BRANCH', not main or an X.Y.x branch:" \
        "the PR will be based there, and CI workflows only run on PRs" \
        "against main -- expect the release check not to run"
fi

# ----------------------------------------------------------------
# Resolve the versions
# ----------------------------------------------------------------

is_dev_version "$CURRENT" ||
    die "VERSION is '$CURRENT', which is not a development version;" \
        "expected a -dev suffix to release from"

[[ -n "$VERSION" ]] ||
    VERSION="$(ask "Release version" "$(strip_prerelease "$CURRENT")")"
validate_version "$VERSION"
[[ "$VERSION" != "$CURRENT" ]] ||
    die "VERSION is already $VERSION -- nothing to release"

[[ -n "$NEXT_VERSION" ]] ||
    NEXT_VERSION="$(ask "Next development version" \
        "$(next_dev_version "$VERSION" "$BASE_BRANCH")")"
validate_version "$NEXT_VERSION"
is_dev_version "$NEXT_VERSION" ||
    die "next version '$NEXT_VERSION' must carry a -dev suffix" \
        "(e.g. $(strip_prerelease "$NEXT_VERSION")-dev)"

BRANCH="chore/release-$VERSION"
TAG="v$VERSION"

# ----------------------------------------------------------------
# Version-dependent preconditions
# ----------------------------------------------------------------

require_tag_absent "$VERSION"
require_changelog_tool

# Caught here as well as in CI, to save drafting a release that cannot
# merge.
check_on_disk_format "$VERSION"

# Said now rather than discovered at the push: recreating a branch is
# the opener's --force, and the one who runs it should know it is
# coming.
if branch_exists_local "$BRANCH" || branch_exists_remote "$BRANCH"; then
    warn "$BRANCH already exists -- open-release-pr.sh recreates it" \
        "from $BASE_BRANCH only with --force"
fi

log "releasing   $CURRENT -> $VERSION on $BASE_BRANCH"
log "next cycle  $NEXT_VERSION"
log "branch      $BRANCH"

# ----------------------------------------------------------------
# Write the draft
# ----------------------------------------------------------------

DRAFT_DIR="$(mktemp -d \
    "${TMPDIR:-/tmp}/pg_vectorsearch-release-$VERSION.XXXXXX")"
# A half-written draft is worse than none: a later run would find two.
# An EXIT trap, because a failure inside a function below ends the
# script without an ERR trap firing.
trap 'rm -rf "$DRAFT_DIR"' EXIT

log "writing the CHANGELOG.md entry"
add_changelog_entry "$DRAFT_DIR/CHANGELOG.md"
insert_notes_template "$DRAFT_DIR/CHANGELOG.md"
note_format_change "$DRAFT_DIR/CHANGELOG.md"
pr_body >"$DRAFT_DIR/body.md"
trap - EXIT

cat <<EOF
==> Draft written to $DRAFT_DIR. The checkout is unchanged.

    Finish the notes in CHANGELOG.md there (every FILL-IN) and say what
    the release is in body.md, keeping Next-Version as its last line.
    Then cut the branch and open the pull request with:

EOF
# Quoted for the shell that runs it: a branch name can carry anything
# but whitespace, and the versions are validated.
printf '      ./scripts/open-release-pr.sh'
[[ "$REPO_GIVEN" == 0 ]] || printf ' --repo %q' "$RELEASE_REPO"
printf ' %s %s %q \\\n' "$VERSION" "$NEXT_VERSION" "$BASE_BRANCH"
printf '          %q \\\n          %q\n' \
    "$DRAFT_DIR/CHANGELOG.md" "$DRAFT_DIR/body.md"
