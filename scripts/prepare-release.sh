#!/bin/bash
# Cut a release PR.
#
# Usage: ./scripts/prepare-release.sh [options]
#
#   --version X.Y.Z           the version to release
#   --next-version X.Y.Z-s    the next development version
#   --create-pr               push and open the PR without asking
#   --ignore-untracked-files  do not ask about untracked files
#   --repo owner/name         release to this repository instead
#
# Passing the first four makes the run fully non-interactive, which is
# what automation wants: it then never consults a terminal, so it cannot
# stall on a runner that happens to allocate one.
#
# --repo is for rehearsing the flow somewhere harmless -- a fork, say.
# Releases go to timescale/pg_vectorsearch by default, stated rather than
# read off a remote so that a differently-named origin cannot aim one at
# the wrong place. Whichever repository is chosen, the branch is pushed to
# the remote pointing at it, found by URL rather than by being called
# "origin".
#
# Both versions are optional and are proposed from ./VERSION: the release
# is that version with its prerelease suffix removed, and the next cycle
# is a minor bump (a patch bump on an X.Y.x maintenance branch). On a
# terminal you are asked to confirm or edit them; without one the
# defaults are taken, so automation needs no flags.
#
# Preparing the branch is local and reversible, so it happens without
# asking. Pushing it and opening the PR is not, so that is asked
# separately -- or answered up front with --create-pr, which is what
# automation wants. Without a terminal and without the flag, the branch is
# prepared and the PR is left for you, along with a script that opens it.
#
# To preview a release, run it and answer no: that leaves the real branch,
# the real changelog entry and a working PR script to read, and prints the
# two commands that undo it.
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

CREATE_PR=0
IGNORE_UNTRACKED_FILES=0
VERSION=""
NEXT_VERSION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --create-pr) CREATE_PR=1; shift ;;
        --ignore-untracked-files) IGNORE_UNTRACKED_FILES=1; shift ;;
        --repo)
            [[ $# -ge 2 ]] || die "--repo needs an owner/name"
            [[ "$2" == */* ]] || die "--repo wants owner/name, got '$2'"
            RELEASE_REPO="$2"
            VS_RELEASE_REMOTE=""  # drop any remote cached for the old one
            shift 2 ;;
        --version)
            [[ $# -ge 2 ]] || die "--version needs a value"
            VERSION="$2"; shift 2 ;;
        --next-version)
            [[ $# -ge 2 ]] || die "--next-version needs a value"
            NEXT_VERSION="$2"; shift 2 ;;
        -h | --help)
            sed -n '2,42p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) die "unexpected argument '$1' (see --help)" ;;
    esac
done

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

# Yes/no, defaulting to no. Returns non-zero without a terminal, so a
# caller can fall back rather than hang.
confirm() {
    [[ -t 0 ]] || return 1
    local reply
    read -r -p "$1 [y/N] " reply </dev/tty
    [[ "$reply" == [yY]* ]]
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
        git-cliff --unreleased --tag "$TAG" --prepend CHANGELOG.md
        return
    fi

    log "first release: no previous tag to delta against, so no" \
        "generated commit list"
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

# Write the commands that open the PR to a script, and run *that* rather
# than doing it inline. Declining the PR, or a failure part-way, then
# leaves something runnable instead of instructions to retype -- and
# because the file is what gets executed, it cannot drift from what the
# script would have done itself.
PR_DIR=""

write_pr_script() {
    local milestone args=()

    PR_DIR="$(mktemp -d \
        "${TMPDIR:-/tmp}/pg_vectorsearch-release-$VERSION.XXXXXX")"

    # --repo explicitly: with more than one GitHub remote gh has no
    # default repository and `gh pr create` fails rather than guessing.
    args=(--repo "$(repo_slug)"
        --base "$BASE_BRANCH" --head "$BRANCH"
        --title "chore: release $VERSION"
        --label "$RELEASE_LABEL")

    if milestone="$(resolve_milestone "$VERSION")"; then
        log "milestone: $milestone"
        args+=(--milestone "$milestone")
    else
        warn "no milestone matches $VERSION (tried \"Release $VERSION\"" \
            "and \"Release $(strip_prerelease "$VERSION")\") -- the PR" \
            "gets none"
    fi

    # The body goes in its own file so nothing has to survive two rounds
    # of shell quoting.
    pr_body >"$PR_DIR/body.md"

    {
        echo '#!/bin/bash'
        echo "# Opens the release PR for $VERSION."
        echo '#'
        echo '# Generated by scripts/prepare-release.sh. Safe to re-run:'
        echo '# the push is idempotent and gh refuses a duplicate PR.'
        echo 'set -euo pipefail'
        printf 'cd %q\n' "$PWD"
        echo '# shellcheck source=scripts/release-lib.sh'
        echo 'source scripts/release-lib.sh'
        # A script kept after its branch was deleted must say so
        # rather than fail inside git push.
        printf 'git rev-parse --verify --quiet refs/heads/%q >/dev/null ||\n' \
            "$BRANCH"
        printf '    die %q\n' \
            "branch $BRANCH does not exist -- run scripts/prepare-release.sh first"
        printf 'ensure_label %q %q %q\n' "$RELEASE_LABEL" 'bfd4f2' \
            'Release PR: merging it publishes a release'
        printf 'git push -u %q %q\n' "$(release_remote)" "$BRANCH"
        printf 'gh pr create'
        printf ' %q' "${args[@]}"
        printf ' --body-file %q\n' "$PR_DIR/body.md"
    } >"$PR_DIR/open-pr.sh"
    chmod +x "$PR_DIR/open-pr.sh"

    log "PR command written to $PR_DIR/open-pr.sh"
}

open_pr() {
    "$PR_DIR/open-pr.sh"
}

# ----------------------------------------------------------------
# Repository state -- checked before anything is asked, so a dirty tree
# or a surprising untracked file surfaces before you pick versions
# ----------------------------------------------------------------

BASE_BRANCH="$(git branch --show-current)"
CURRENT="$(project_version)"

require_clean_tree
review_untracked "$IGNORE_UNTRACKED_FILES" confirm
require_branch_up_to_date "$BASE_BRANCH"

# ----------------------------------------------------------------
# Resolve the versions
# ----------------------------------------------------------------

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
# Version-dependent preconditions -- still all before anything is mutated
# ----------------------------------------------------------------

require_tag_absent "$VERSION"
require_branch_absent "$BRANCH"
require_changelog_tool

# An incompatible format change cannot ship in a patch release; catching
# it here rather than in CI saves cutting a PR that cannot merge.
check_on_disk_format "$VERSION"

log "releasing   $CURRENT -> $VERSION on $BASE_BRANCH"
log "next cycle  $NEXT_VERSION"
log "branch      $BRANCH"

# ----------------------------------------------------------------
# Go
# ----------------------------------------------------------------

log "creating $BRANCH"
git checkout -b "$BRANCH"

log "setting VERSION to $VERSION"
set_version "$VERSION"

log "writing the CHANGELOG.md entry"
add_changelog_entry
insert_notes_template
note_format_change

git add VERSION CHANGELOG.md
git commit -m "chore: release $VERSION"

log "$BRANCH is ready: VERSION is $VERSION and CHANGELOG.md has its"
log "entry (notes still unfinished)."

write_pr_script

if [[ "$CREATE_PR" == 1 ]] ||
    confirm "Push $BRANCH and open the release PR?"; then
    open_pr
    log "done. Complete the release notes in the PR; it cannot merge"
    log "while any FILL-IN placeholder remains."
    exit 0
fi

cat <<EOF
==> Branch prepared, PR not opened. To open it, run:

      $PR_DIR/open-pr.sh

    It pushes $BRANCH and opens the PR with the title, label, milestone
    and body this run worked out. To undo instead:

      git checkout $BASE_BRANCH && git branch -D $BRANCH
EOF
