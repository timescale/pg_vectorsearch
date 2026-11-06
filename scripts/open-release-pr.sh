#!/bin/bash
# Open a release pull request from a finished changelog and body.
#
# Usage: ./scripts/open-release-pr.sh [--force] [--repo owner/name]
#            [--disable-check <name>]...
#            <version> <next-dev> <base-branch> <changelog-file> <body-file>
#
# The second half of cutting a release. prepare-release.sh writes the
# two files as drafts. Once the notes in them are written, this cuts
# chore/release-<version> from the base, commits CHANGELOG.md and
# VERSION, runs the release check, pushes the branch and opens the
# pull request. A person runs it from a checkout; the "Cut a release
# PR" workflow runs it in a job of its own, on a fresh checkout, after
# the job that wrote the notes. Both run it the same way.
#
# The checkout must be clean and sit on a commit the remote's base
# branch has. Nothing else in it is read: the release commit is built
# from the two files alone, so whatever wrote them cannot put anything
# else into the release. A run that is refused leaves the checkout as
# it found it.
#
# next-dev is the version the development cycle reopens at, blank for
# the default bump. The body must end with exactly one Next-Version
# trailer naming it, and carry no FILL-IN: a squash merge turns the
# body into main's commit message, and the trailer is what the release
# pipeline reads back. The changelog entry must be finished too, which
# the release check enforces.
#
# --disable-check turns one of the release check's assertions off, as
# a Disable-check trailer appended to the body. The body file itself
# may not carry one: turning a check off is the decision of whoever
# opens the pull request, never something in the notes.
#
# --force recreates a branch that already exists, locally or on the
# remote, and force-pushes with a lease on the commit the remote had,
# so the push refuses if someone moved the branch meanwhile. An open
# pull request on the branch is kept and its body refreshed.
#
# With APP_SLUG set, the commit is attributed to that App's bot account
# and the remote gets a credential built from GH_TOKEN. Both are
# skipped when it is unset, so a run by hand neither rewrites the
# operator's git identity nor puts a token in their remote.
#
# Writes url= to $GITHUB_OUTPUT when set.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/release-lib.sh"

require_repo_root

FORCE=0
DISABLED=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --disable-check)
            [[ $# -ge 2 ]] || die "--disable-check needs a check name"
            DISABLED+=("$2"); shift 2 ;;
        --repo)
            [[ $# -ge 2 ]] || die "--repo needs an owner/name"
            [[ "$2" == */* ]] || die "--repo wants owner/name, got '$2'"
            RELEASE_REPO="$2"
            VS_RELEASE_REMOTE=""  # drop any remote cached for the old one
            shift 2 ;;
        -h | --help)
            sed -n '2,44p' "$0" | sed 's/^# \?//'; exit 0 ;;
        --*) die "unexpected option '$1' (see --help)" ;;
        *) break ;;
    esac
done

[[ $# -eq 5 ]] ||
    die "usage: $0 [--force] [--repo owner/name] [--disable-check name]" \
        "<version> <next-dev> <base-branch> <changelog-file> <body-file>"

VERSION="$1"
NEXT="$2"
BASE="$3"
CHANGELOG="$4"
BODY_FILE="$5"

validate_version "$VERSION"
! is_dev_version "$VERSION" ||
    die "$VERSION is a development version, not a release"

if [[ -z "$NEXT" ]]; then
    NEXT="$(next_dev_version "$VERSION" "$BASE")"
    log "next development version defaults to $NEXT"
fi
validate_version "$NEXT"
is_dev_version "$NEXT" ||
    die "next version $NEXT carries no -dev suffix, so the gate would" \
        "read the bump that lands it as another release"

BRANCH="chore/release-$VERSION"
REMOTE="$(release_remote)"
BASE_REF="$REMOTE/$BASE"

# ----------------------------------------------------------------
# The notes, checked before anything else so a typo costs one re-run
# ----------------------------------------------------------------

[[ -f "$CHANGELOG" ]] || die "no changelog at $CHANGELOG"
[[ -f "$BODY_FILE" ]] || die "no pull request body at $BODY_FILE"

changelog_has_section "$VERSION" "$CHANGELOG" ||
    die "the changelog has no '## [$VERSION]' entry"
section="$(changelog_section "$VERSION" "$CHANGELOG")"
if grep -q 'FILL-IN' <<<"$section"; then
    die "the changelog entry for $VERSION still has FILL-IN placeholders"
fi
# The gate refuses an empty entry after the merge, when main already
# carries the released VERSION.
[[ -n "${section//[[:space:]]/}" ]] ||
    die "the changelog entry for $VERSION is empty"

! grep -q 'FILL-IN' "$BODY_FILE" ||
    die "the pull request body still has FILL-IN placeholders"
# Read as written and as rendered: a comment can hide the line from
# one reading and not the other, and the check honours the rendered
# one. The rendered body is captured whole first, so the grep's early
# exit cannot turn into a pipe failure that reads as no match.
rendered="$(strip_html_comments <"$BODY_FILE")"
if grep -qi '^disable-check:' "$BODY_FILE" ||
    grep -qi '^disable-check:' <<<"$rendered"; then
    die "the pull request body carries a Disable-check trailer -- pass" \
        "--disable-check instead, so the override is this run's"
fi

# Exactly one trailer, read the way git reads a squashed message:
# from the last paragraph only. Two would be concatenated by the
# gate's trailer read into a version nothing can parse. --no-divider,
# because the gate's read does not stop at a --- line either.
trailers="$(git interpret-trailers --parse --no-divider <"$BODY_FILE" |
    grep -i '^next-version:' || true)"
case "$(grep -c . <<<"$trailers")" in
    0) die "the pull request body does not end with a Next-Version" \
        "trailer" ;;
    1) ;;
    *) die "the pull request body carries more than one Next-Version" \
        "trailer" ;;
esac
body_next="${trailers#*:}"
body_next="${body_next//[[:space:]]/}"
[[ "$body_next" == "$NEXT" ]] ||
    die "the body declares Next-Version: $body_next, this run $NEXT"

# Where a refresh keeps the description it replaces: beside the
# notes, so it must be neither of them.
PREVIOUS_BODY="$(dirname "$BODY_FILE")/body-previous.md"
for notes in "$BODY_FILE" "$CHANGELOG"; do
    [[ ! "$PREVIOUS_BODY" -ef "$notes" ]] ||
        die "body-previous.md is where the replaced description goes --" \
            "name $notes something else"
done

# The override joins the trailer paragraph, where git reads it from
# the squashed message too. Trailing whitespace goes first: a line of
# spaces after the trailer would start a new paragraph and leave
# Next-Version outside the one git reads.
if [[ ${#DISABLED[@]} -gt 0 ]]; then
    with_override="$(mktemp)"
    body="$(cat "$BODY_FILE")"
    body="${body%"${body##*[![:space:]]}"}"
    {
        printf '%s\n' "$body"
        printf 'Disable-check: %s\n' "${DISABLED[*]}"
    } >"$with_override"
    BODY_FILE="$with_override"
fi

# ----------------------------------------------------------------
# The checkout: the base branch, untouched
# ----------------------------------------------------------------

require_clean_tree

git rev-parse -q --verify "$BASE_REF^{commit}" >/dev/null ||
    die "no $BASE_REF -- fetch $BASE from $REMOTE first"

git merge-base --is-ancestor HEAD "$BASE_REF" ||
    die "$(git rev-parse --short HEAD) is not on $BASE_REF -- switch to" \
        "$BASE first"

is_dev_version "$(project_version)" ||
    die "VERSION is $(project_version), which is not a development" \
        "version to release from"

# The cut writes CHANGELOG.md from the draft and a refusal takes the
# cut back, so the checkout's copy is either tracked, and restored
# from the commit, or absent. One the base does not track would be
# written over and then dropped, and the draft itself must not be it.
had_changelog="$(git ls-files CHANGELOG.md)"
[[ -n "$had_changelog" || ( ! -e CHANGELOG.md && ! -L CHANGELOG.md ) ]] ||
    die "the checkout has a CHANGELOG.md that $BASE does not track --" \
        "move it out of the way; the release writes its own from" \
        "$CHANGELOG"
[[ ! "$CHANGELOG" -ef CHANGELOG.md ]] ||
    die "the changelog argument is the checkout's own CHANGELOG.md --" \
        "pass the draft prepare-release.sh wrote"
# The body is read again after the cut, so it must not be a file the
# cut writes. -ef compares inodes, so a link counts too.
for written in CHANGELOG.md VERSION; do
    [[ ! "$BODY_FILE" -ef "$written" ]] ||
        die "the pull request body is the checkout's own $written --" \
            "pass the draft prepare-release.sh wrote"
done

# First, so every remote query below goes out with the credential the
# push will use.
configure_app_identity

require_tag_absent "$VERSION"

# A branch that is already there is recreated only when asked. The
# lease is the commit the remote had when this looked, so a push over
# a branch someone moved meanwhile is refused as stale.
LEASE=""
if branch_exists_remote "$BRANCH"; then
    LEASE="$(remote_branch_sha "$BRANCH")"
fi
if [[ -n "$LEASE" ]] || branch_exists_local "$BRANCH"; then
    [[ "$FORCE" == 1 ]] ||
        die "$BRANCH already exists -- pass --force to recreate it" \
            "from $BASE"
    warn "recreating $BRANCH from $BASE"
    # A pull request follows its head branch by name, so the push
    # rewrites every open one on it, whatever base it targets.
    prs="$(open_prs_on_branch "$BRANCH")"
    if [[ -n "$prs" ]]; then
        warn "the push will force over the commit these open pull" \
            "requests are reviewing:"
        while IFS= read -r line; do
            [[ -n "$line" ]] && echo "    $line" >&2
        done <<<"$prs"
        if ! grep -q " into $BASE " <<<"$prs"; then
            warn "none of them targets $BASE, so this run opens a new" \
                "pull request and leaves the ones above pointing at a" \
                "branch whose history it rewrote -- close them first" \
                "unless you mean that"
        fi
    fi
fi

# ----------------------------------------------------------------
# The release commit
# ----------------------------------------------------------------

# Where to return to if a step below refuses: the branch, or the
# commit when detached. The tree was clean, so whatever the cut left
# in it is the cut's own and is thrown away. A branch the cut
# recreated goes back to its old tip, one it created is dropped, and
# a re-run starts from the same place this one did.
start="$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)"
[[ "$start" != "$BRANCH" ]] ||
    die "the checkout is on $BRANCH itself -- switch to $BASE first"
old_tip="$(git rev-parse -q --verify "refs/heads/$BRANCH" || true)"
# Hooks are for the operator's own checkouts. A post-checkout hook
# cannot stop a switch, only fail its exit status after the fact,
# which would read here as a refusal with the branch already moved.
nohooks=(-c core.hooksPath=/dev/null)
restore_checkout() {
    git reset -q --hard || return 1
    [[ -n "$had_changelog" ]] || rm -f CHANGELOG.md
    git "${nohooks[@]}" checkout -q "$start" || return 1
    if [[ -n "$old_tip" ]]; then
        git branch -q -f "$BRANCH" "$old_tip"
    elif git rev-parse -q --verify "refs/heads/$BRANCH" >/dev/null; then
        git branch -q -D "$BRANCH"
    fi
}
undo_cut() {
    local rc=$?
    [[ "$rc" != 0 ]] || return 0
    warn "refused after cutting $BRANCH: putting the checkout back on" \
        "$start"
    restore_checkout ||
        warn "could not put $BRANCH back -- the checkout is as the" \
            "failure left it"
    exit "$rc"
}

log "cutting $BRANCH from $(git rev-parse --short HEAD)"
trap undo_cut EXIT
git "${nohooks[@]}" switch -q -C "$BRANCH"

cp "$CHANGELOG" CHANGELOG.md
set_version "$VERSION"

git add CHANGELOG.md VERSION
git commit -q -m "chore: release $VERSION" -m "Next-Version: $NEXT"

# The one fact the push relies on, so asserted rather than assumed
# from the two files staged above.
changed="$(git diff --name-only HEAD~1 HEAD | sort | tr '\n' ' ')"
[[ "$changed" == "CHANGELOG.md VERSION " ]] ||
    die "the release commit changes '${changed% }'"

# The same check the pull request runs, before anything is pushed.
# The body is the pull request description, so a Disable-check
# trailer in it counts here as it will there. The base branch is
# named the way the pull request names it, without the remote.
BODY="$(cat "$BODY_FILE")" GITHUB_BASE_REF="$BASE" \
    ./scripts/ci/release-check.sh "$BASE_REF"

# ----------------------------------------------------------------
# Push and open
# ----------------------------------------------------------------

ensure_label "$RELEASE_LABEL" bfd4f2 \
    'Release PR: merging it publishes a release'

if [[ -n "$LEASE" ]]; then
    git push --force-with-lease="$BRANCH:$LEASE" -u "$REMOTE" "$BRANCH"
else
    git push -u "$REMOTE" "$BRANCH"
fi
# On the remote now, so it stays whatever happens below.
trap - EXIT

# The push has already updated an open pull request, and gh refuses
# a duplicate, so the body is refreshed instead of creating one.
url="$(open_pr_url "$BRANCH" "$BASE")"
if [[ -n "$url" ]]; then
    log "pull request already open, updated by the push: $url"
    update_pr_body "$url" "$BODY_FILE" "$NEXT" "$PREVIOUS_BODY"
else
    args=(--repo "$(repo_slug)"
        --base "$BASE" --head "$BRANCH"
        --title "chore: release $VERSION"
        --label "$RELEASE_LABEL"
        --body-file "$BODY_FILE")
    if milestone="$(resolve_milestone "$VERSION")"; then
        log "milestone: $milestone"
        args+=(--milestone "$milestone")
    else
        warn "no milestone matches $VERSION -- the pull request gets none"
    fi
    url="$(gh pr create "${args[@]}")"
    log "opened $url"
fi

[[ -z "${GITHUB_OUTPUT:-}" ]] || printf 'url=%s\n' "$url" >>"$GITHUB_OUTPUT"
