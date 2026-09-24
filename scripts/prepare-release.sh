#!/bin/bash
# Cut a release PR.
#
# Usage: ./scripts/prepare-release.sh [options]
#
#   --version X.Y.Z           the version to release
#   --next-version X.Y.Z-s    the next development version
#   --create-pr               push and open the PR without asking
#   --ignore-untracked-files  do not ask about untracked files
#   --force                   overwrite an existing release branch
#   --repo owner/name         release to this repository instead
#
# Both versions default from ./VERSION: the release is it without its
# prerelease suffix, the next cycle a minor bump (patch on an X.Y.x
# branch). Passing all five flags makes the run consult no terminal at
# all, which is what automation needs.
#
# Preparing the branch happens without asking, being local and
# reversible. Pushing it is the one question, because that publishes the
# release; answer no to preview, then run the printed script or drop the
# branch.
#
# The release notes are left unfinished on purpose: the PR opens with
# FILL-IN placeholders and release-check.yml fails while any remain.
#
# See docs/release.md.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/release-lib.sh"

require_repo_root

CREATE_PR=0
IGNORE_UNTRACKED_FILES=0
FORCE=0
VERSION=""
NEXT_VERSION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --create-pr) CREATE_PR=1; shift ;;
        --ignore-untracked-files) IGNORE_UNTRACKED_FILES=1; shift ;;
        --force) FORCE=1; shift ;;
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
            sed -n '2,26p' "$0" | sed 's/^# \?//'; exit 0 ;;
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
# Changelog
# ----------------------------------------------------------------

# --unreleased means "not reachable from any tag", so with no tags yet
# this covers the whole history -- the first release included.
add_changelog_entry() {
    git-cliff --unreleased --tag "$TAG" --prepend CHANGELOG.md
}

# Splice the notes template under the version heading, stripping its
# guidance comments. FILL-IN placeholders stay; CI fails while any remain.
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
    ' CHANGELOG.md >CHANGELOG.md.new
    mv CHANGELOG.md.new CHANGELOG.md
}

# ----------------------------------------------------------------
# An existing release branch
# ----------------------------------------------------------------

# An existing release branch is overwritten, not refused: rehearsing a
# release means re-cutting the same version. Safe to offer only because
# require_tag_absent runs first, so a shipped version never reaches here.
#
# REUSE_BRANCH selects `git checkout -B`; REMOTE_BRANCH_SHA becomes the
# --force-with-lease value.
REUSE_BRANCH=0
REMOTE_BRANCH_SHA=""
REMOTE_BRANCH_SHA_SHORT=""

resolve_branch_reuse() {
    local where="" prs=""

    branch_exists_local "$BRANCH" && where="locally"
    if branch_exists_remote "$BRANCH"; then
        where="${where:+$where and }on $(release_remote)"
        REMOTE_BRANCH_SHA="$(remote_branch_sha "$BRANCH")"
        REMOTE_BRANCH_SHA_SHORT="${REMOTE_BRANCH_SHA:0:8}"
    fi
    [[ -n "$where" ]] || return 0

    warn "$BRANCH already exists $where"

    # A PR follows its head branch by name, so a push rewrites the head
    # of every open PR on it -- including one against another base,
    # whose diff then becomes meaningless with nothing else to say so.
    prs="$(open_prs_on_branch "$BRANCH")"
    if [[ -n "$prs" ]]; then
        warn "pushing later would force over the commit these open PRs" \
            "are reviewing:"
        while IFS= read -r line; do
            [[ -n "$line" ]] && echo "    $line" >&2
        done <<<"$prs"
        if ! grep -q " into $BASE_BRANCH " <<<"$prs"; then
            warn "none of them targets $BASE_BRANCH, so this run opens a" \
                "new PR and leaves the ones above pointing at a branch" \
                "whose history it rewrote -- close them first unless you" \
                "mean that"
        fi
    fi

    # Deliberately says nothing about pushing: this answer only recreates
    # the local branch, and the push has its own question further down.
    if [[ "$FORCE" == 1 ]]; then
        warn "recreating it as asked (--force)"
    elif confirm "Recreate $BRANCH from $BASE_BRANCH?"; then
        :
    else
        # Carrying on would destroy work, so no terminal means no.
        die "aborted -- delete $BRANCH to start over, or pass --force" \
            "to overwrite it"
    fi

    REUSE_BRANCH=1
}

# ----------------------------------------------------------------
# The PR
# ----------------------------------------------------------------

# What the PR does, not how releasing works -- that is docs/release.md's
# job. Highlights come from the entry this run just wrote, so the summary
# cannot contradict the notes.
pr_body() {
    # Backticks here are markdown, not command substitution.
    # shellcheck disable=SC2016
    printf 'Releases `%s`.\n\n' "$VERSION"

    # Omitted while still a placeholder: nothing refreshes the body when
    # the notes are completed by ordinary commits, and a squash merge
    # turns the body into main's commit message -- so a FILL-IN here
    # would ship in the history. Re-running once the notes are written
    # picks them up.
    local highlights
    highlights="$(highlights_section)"
    if [[ -n "$highlights" ]] && ! grep -q 'FILL-IN' <<<"$highlights"; then
        printf '## Highlights\n\n%s\n\n' "$highlights"
    fi
    next_version_trailer
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
next_version_trailer() {
    printf 'Next-Version: %s\n' "$NEXT_VERSION"
}

# This version's Highlights subsection, or a pointer if the template
# shape ever changes and it is missing.
#
# The awk stops printing rather than exiting: an early exit would SIGPIPE
# changelog_section upstream, and pipefail would fail the assignment.
highlights_section() {
    local out
    out="$(changelog_section "$VERSION" | awk '
        /^### Highlights$/ { on = 1; next }
        on && /^#/         { on = 0; next }
        on {
            if (!seen && $0 ~ /^[[:space:]]*$/) next
            seen = 1
            print
        }')"
    if [[ -n "$out" ]]; then
        printf '%s\n' "$out"
    else
        printf '%s\n' "See the \`$VERSION\` entry in \`CHANGELOG.md\`."
    fi
}

# The PR is opened by a generated script, which is also what this run
# executes -- so declining leaves something runnable that cannot drift
# from what would have happened.
PR_DIR=""

write_pr_script() {
    local milestone args=()

    PR_DIR="$(mktemp -d \
        "${TMPDIR:-/tmp}/pg_vectorsearch-release-$VERSION.XXXXXX")"

    # --repo explicitly: with two GitHub remotes gh has no default and
    # `gh pr create` fails rather than guessing.
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

    # Its own file, so nothing survives two rounds of shell quoting.
    pr_body >"$PR_DIR/body.md"

    {
        echo '#!/bin/bash'
        echo "# Opens the release PR for $VERSION."
        echo '#'
        echo '# Generated by scripts/prepare-release.sh. Safe to re-run:'
        echo '# the push is idempotent and gh refuses a duplicate PR.'
        echo 'set -euo pipefail'
        printf 'cd %q\n' "$PWD"
        # Set before sourcing, so the library's default does not win.
        # Without it every helper here -- the open-PR lookup, the body
        # refresh, the label -- would act on the default repository
        # while gh pr create acted on --repo's.
        printf 'RELEASE_REPO=%q\n' "$(repo_slug)"
        echo '# shellcheck source=scripts/release-lib.sh'
        echo 'source scripts/release-lib.sh'
        # Say so, rather than failing inside git push, if the branch is
        # gone by the time this runs.
        printf 'git rev-parse --verify --quiet refs/heads/%q >/dev/null ||\n' \
            "$BRANCH"
        printf '    die %q\n' \
            "branch $BRANCH does not exist -- run scripts/prepare-release.sh first"
        printf 'ensure_label %q %q %q\n' "$RELEASE_LABEL" 'bfd4f2' \
            'Release PR: merging it publishes a release'
        # A force is needed whenever the branch is already on the
        # remote -- this cut overwrote one, or an earlier run pushed it
        # and the notes have since been amended in.
        #
        # A force is needed only when the branch was already on the
        # remote when this run looked, and the lease value is the sha it
        # saw then -- so the push refuses if the branch moved in the
        # meantime.
        #
        # Not git's default lease: that reads the remote-tracking ref,
        # which is absent when someone else pushed the branch, and an
        # absent expected value makes `--force-with-lease` reject a
        # perfectly good push as "stale info". Not the sha read at push
        # time either, which would always match and be a bare --force
        # wearing a lease.
        if [[ -n "$REMOTE_BRANCH_SHA" ]]; then
            printf 'git push --force-with-lease=%q -u %q %q\n' \
                "$BRANCH:$REMOTE_BRANCH_SHA" "$(release_remote)" "$BRANCH"
        else
            printf 'git push -u %q %q\n' "$(release_remote)" "$BRANCH"
        fi

        # A push updates an open PR, and `gh pr create` fails when one
        # exists -- so create is conditional and the body is refreshed
        # instead. This is what makes re-running safe.
        #
        # $pr is the generated script's, so it must reach the file
        # unexpanded.
        # shellcheck disable=SC2016
        printf 'if pr=$(open_pr_url %q %q) && [[ -n "$pr" ]]; then\n' \
            "$BRANCH" "$BASE_BRANCH"
        # shellcheck disable=SC2016
        printf '    log "PR already open, updated by the push: $pr"\n'
        # shellcheck disable=SC2016
        printf '    update_pr_body "$pr" %q %q %q\n' \
            "$PR_DIR/body.md" "$NEXT_VERSION" "$PR_DIR/body-previous.md"
        printf 'else\n'
        printf '    gh pr create'
        printf ' %q' "${args[@]}"
        printf ' --body-file %q\n' "$PR_DIR/body.md"
        printf 'fi\n'
    } >"$PR_DIR/open-pr.sh"
    chmod +x "$PR_DIR/open-pr.sh"

    log "PR command written to $PR_DIR/open-pr.sh"
}

open_pr() {
    "$PR_DIR/open-pr.sh"
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
RELEASE_LABEL="release"

# ----------------------------------------------------------------
# Version-dependent preconditions -- still all before anything is mutated
# ----------------------------------------------------------------

require_tag_absent "$VERSION"
resolve_branch_reuse
require_changelog_tool

# Caught here as well as in CI, to save cutting a PR that cannot merge.
check_on_disk_format "$VERSION"

log "releasing   $CURRENT -> $VERSION on $BASE_BRANCH"
log "next cycle  $NEXT_VERSION"
log "branch      $BRANCH"

# ----------------------------------------------------------------
# Go
# ----------------------------------------------------------------

if [[ "$REUSE_BRANCH" == 1 ]]; then
    log "recreating $BRANCH"
    git checkout -B "$BRANCH"
else
    log "creating $BRANCH"
    git checkout -b "$BRANCH"
fi

log "setting VERSION to $VERSION"
set_version "$VERSION"

log "writing the CHANGELOG.md entry"
add_changelog_entry
insert_notes_template
note_format_change

git add VERSION CHANGELOG.md
git commit -m "chore: release $VERSION" -m "$(next_version_trailer)"

log "$BRANCH is ready: VERSION is $VERSION and CHANGELOG.md has its"
log "entry (notes still unfinished)."

write_pr_script

# Name what the push will actually do; overwriting force-pushes and
# updates the open PR rather than opening one.
if [[ "$REUSE_BRANCH" == 1 && -n "$REMOTE_BRANCH_SHA" ]]; then
    push_question="Force-push $BRANCH over $REMOTE_BRANCH_SHA_SHORT and update the release PR?"
else
    push_question="Push $BRANCH and open the release PR?"
fi

if [[ "$CREATE_PR" == 1 ]] || confirm "$push_question"; then
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
