#!/bin/bash
# Shared helpers for the release scripts: the operator-facing ones in
# scripts/ and the pipeline stages in scripts/ci/. Sourced, not executed.
#
# The version comes from ./VERSION -- the single source of truth, which
# meson.build reads for project(version:) -- and the extension name from
# meson.build's project() call. Nothing here hardcodes either.

# shellcheck shell=bash

log()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }

require_repo_root() {
    [[ -f meson.build && -f VERSION ]] ||
        die "run from the repository root"
}

# ----------------------------------------------------------------
# Version
# ----------------------------------------------------------------

# The project version, from VERSION. Tolerates a trailing newline and
# stray surrounding whitespace; meson itself requires the file to hold
# exactly one line.
project_version() {
    tr -d '[:space:]' <VERSION
}

set_version() {
    printf '%s\n' "$1" >VERSION
}

# Project name from meson.build's project() call (first quoted string).
meson_project_name() {
    sed -n "s/^project('\([^']*\)'.*/\1/p" meson.build | head -1
}

# Validate a version string: semver-ish with an optional prerelease
# suffix, and legal as a PostgreSQL extension version (no '--', no
# leading or trailing '-').
validate_version() {
    local v="$1"
    [[ "$v" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-z0-9.]+)?$ ]] ||
        die "invalid version '$v' (expected X.Y.Z or X.Y.Z-suffix)"
    [[ "$v" != *--* ]] || die "version must not contain '--'"
}

# A prerelease is any version with a -suffix (-alpha1, -rc1, -dev, ...).
is_prerelease() {
    [[ "$1" == *-* ]]
}

# The release part of a version: 0.2.0-rc1 -> 0.2.0.
strip_prerelease() {
    echo "${1%%-*}"
}

# The version a development cycle should open at after releasing $1.
#
# Minor bump by default; patch bump on an X.Y.x maintenance branch, where
# by definition only patch releases happen. $2 is the branch being
# released from (default: the current branch).
next_dev_version() {
    local released="$1"
    local branch="${2:-$(git branch --show-current)}"
    local base major minor patch
    base="$(strip_prerelease "$released")"
    IFS=. read -r major minor patch <<<"$base"

    if [[ "$branch" =~ ^[0-9]+\.[0-9]+\.x$ ]]; then
        echo "$major.$minor.$((patch + 1))-dev"
    else
        echo "$major.$((minor + 1)).0-dev"
    fi
}

# ----------------------------------------------------------------
# Changelog
# ----------------------------------------------------------------

# Print the CHANGELOG.md entry for a version: everything from its
# "## [<version>]" heading (exclusive) to the next "## [" heading.
changelog_section() {
    local version="$1"
    awk -v ver="$version" '
        $0 ~ "^## \\[" ver "\\]" { found = 1; next }
        found && /^## \[/ { exit }
        found { print }
    ' CHANGELOG.md
}

changelog_has_section() {
    grep -q "^## \[$1\]" CHANGELOG.md
}

# ----------------------------------------------------------------
# Git state
# ----------------------------------------------------------------

# Uncommitted changes to *tracked* files. These are refused: they ride
# along onto the release branch, so building or testing there to check
# the release would be testing them rather than what the PR contains.
# Untracked files are handled separately -- see review_untracked.
require_clean_tree() {
    local dirty
    dirty="$(git status --porcelain --untracked-files=no)"
    [[ -z "$dirty" ]] || {
        echo "$dirty" >&2
        die "uncommitted changes to tracked files -- commit or stash" \
            "them first"
    }
}

# Untracked files do not block a release: having them is normal, and they
# cannot reach the release commit (only VERSION and CHANGELOG.md are
# staged) or the tagged commit. But one of them might be a file that
# should have been committed -- a new source file, a new upgrade script --
# and shipping without it is worse than a prompt. So they are listed and
# confirmed rather than passed over silently.
#
# $1 is 1 to skip the question and proceed; $2 is a callback that asks a
# yes/no question and returns non-zero for no. They are always listed
# either way -- knowing what was passed over costs a line and is worth
# having in a release log. With nobody to ask, stray files in a checkout
# must not fail a release, so it warns and continues.
review_untracked() {
    local ignore="$1" confirm_fn="$2" untracked f
    untracked="$(git status --porcelain --untracked-files=normal |
        sed -n 's/^?? //p')"
    [[ -n "$untracked" ]] || return 0

    warn "untracked files present -- check none belong in the release:"
    while IFS= read -r f; do
        [[ -n "$f" ]] && echo "    $f" >&2
    done <<<"$untracked"

    if [[ "$ignore" == 1 ]]; then
        warn "ignoring them as asked (--ignore-untracked-files)"
    elif [[ -t 0 ]]; then
        "$confirm_fn" "Release anyway?" ||
            die "aborted -- commit what belongs in the release, move the" \
                "rest aside, or pass --ignore-untracked-files"
    else
        warn "no terminal to ask on; continuing"
    fi
}

require_tag_absent() {
    local tag="v$1" remote
    remote="$(release_remote)"
    git rev-parse -q --verify "refs/tags/$tag" >/dev/null &&
        die "tag $tag already exists locally"
    git ls-remote --exit-code --tags "$remote" "refs/tags/$tag" \
        >/dev/null 2>&1 &&
        die "tag $tag already exists on $remote"
    return 0
}

# The release branch must not exist yet, locally or on the release remote.
#
# Without this, a second run of prepare-release.sh fails inside
# `git checkout -b` with git's own "a branch named ... already exists",
# after the version prompts and with no hint about which of the two
# situations it is: a release already in flight, or a branch left behind
# by a run that was abandoned. Both have an obvious next step, but only
# once you know which one you are in.
require_branch_absent() {
    local branch="$1" remote
    remote="$(release_remote)"
    git rev-parse -q --verify "refs/heads/$branch" >/dev/null &&
        die "branch $branch already exists locally -- this release may" \
            "already be in flight (check for an open PR on" \
            "$(repo_slug)); if it was an abandoned attempt, delete it" \
            "with 'git branch -D $branch' and run again"
    git ls-remote --exit-code --heads "$remote" "refs/heads/$branch" \
        >/dev/null 2>&1 &&
        die "branch $branch already exists on $remote -- the release is" \
            "already in flight (look for its PR on $(repo_slug)); to" \
            "start over, close that PR and delete the branch with" \
            "'git push $remote --delete $branch'"
    return 0
}

# The release branch must be cut from an up-to-date base, or the
# generated changelog delta and the tagged commit disagree with what
# everyone else sees.
require_branch_up_to_date() {
    local branch="${1:-$(git branch --show-current)}" remote
    remote="$(release_remote)"
    git fetch -q "$remote" "$branch" ||
        die "cannot fetch $remote/$branch"
    local local_sha remote_sha
    local_sha="$(git rev-parse HEAD)"
    remote_sha="$(git rev-parse FETCH_HEAD)"
    [[ "$local_sha" == "$remote_sha" ]] ||
        die "$branch is not in sync with $remote/$branch" \
            "(local $local_sha, remote $remote_sha)"
}

# ----------------------------------------------------------------
# GitHub
# ----------------------------------------------------------------

# The repository releases are published to, stated rather than guessed.
#
# Not from `gh repo view` or a remote: gh has no default repository when a
# checkout has more than one GitHub remote, and `gh pr create` then fails
# with "No default remote repository has been set" -- while deriving it
# from a remote would let a misnamed origin aim a release at a fork. Which
# repository this project releases to is a fact about the project, so it
# is written down. RELEASE_REPO overrides it for testing against a fork.
: "${RELEASE_REPO:=timescale/pg_vectorsearch}"

repo_slug() {
    printf '%s\n' "$RELEASE_REPO"
}

# The slug a remote URL points at, for all three spellings:
# git@host:owner/repo, https://host/owner/repo and
# ssh://git@host/owner/repo, with or without a .git suffix.
remote_slug() {
    sed -E -e 's#\.git$##' -e 's#^.*github\.com[:/]##' <<<"$1"
}

# The remote that points at RELEASE_REPO.
#
# Not assumed to be "origin": a fork workflow commonly has origin as the
# fork and the canonical repository under another name, and pushing the
# release branch to a fork while opening the PR against the canonical
# repository would fail with the branch missing. Found by URL, so the
# remote's name does not matter.
VS_RELEASE_REMOTE=""
release_remote() {
    [[ -z "$VS_RELEASE_REMOTE" ]] || {
        printf '%s\n' "$VS_RELEASE_REMOTE"
        return 0
    }
    local name url
    while read -r name url; do
        if [[ "$(remote_slug "$url")" == "$RELEASE_REPO" ]]; then
            VS_RELEASE_REMOTE="$name"
            printf '%s\n' "$name"
            return 0
        fi
    done < <(git remote -v | awk '$3 == "(push)" { print $1, $2 }')

    die "no remote points at $RELEASE_REPO (have:" \
        "$(git remote | tr '\n' ' ')) -- add one, or pick another" \
        "repository with --repo"
}

# Create a label if it does not exist. Idempotent; never fatal, because a
# missing label must not stop a release.
ensure_label() {
    local name="$1" color="$2" description="$3" slug
    slug="$(repo_slug)"
    gh label create "$name" --repo "$slug" --color "$color" \
        --description "$description" >/dev/null 2>&1 ||
        gh label list --repo "$slug" --json name --jq '.[].name' \
            2>/dev/null | grep -qxF "$name" ||
        warn "could not create or find the '$name' label"
    return 0
}

# The milestone tracking a version, following the repo's "Release X.Y.Z"
# naming. Falls back to the release part of a prerelease version
# (0.2.0-rc1 -> "Release 0.2.0"), then to nothing: a milestone is
# metadata and must never block a release.
resolve_milestone() {
    local version="$1" titles candidate
    titles="$(gh api --paginate \
        "repos/$(repo_slug)/milestones?state=all" \
        --jq '.[].title' 2>/dev/null)" || titles=""

    for candidate in "Release $version" \
        "Release $(strip_prerelease "$version")"; do
        if grep -qxF "$candidate" <<<"$titles"; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# ----------------------------------------------------------------
# On-disk format
# ----------------------------------------------------------------

# The index format version: the low byte of PRISM_META_MAGIC, bumped on
# any incompatible metapage or layout change so an older index is
# rejected at open. $1 is a git ref, or empty for the working tree.
# Prints nothing if it cannot be read -- callers must treat that as
# "unknown", never as "changed".
meta_format_version() {
    local ref="${1:-}" src
    if [[ -n "$ref" ]]; then
        src="$(git show "$ref:src/pg/meta.h" 2>/dev/null)" || return 0
    else
        src="$(cat src/pg/meta.h 2>/dev/null)" || return 0
    fi
    sed -n \
        's/^#define PRISM_META_MAGIC.*0x[0-9A-Fa-f]\{6\}\([0-9A-Fa-f]\{2\}\).*/\1/p' \
        <<<"$src" | head -1
}

# The newest release tag reachable from HEAD, or nothing.
previous_release_tag() {
    git tag --list 'v[0-9]*' --merged HEAD --sort=-v:refname | head -1
}

# True when $2 is only a patch bump away from $1 (same major.minor).
is_patch_bump() {
    local from to
    from="$(strip_prerelease "$1")"
    to="$(strip_prerelease "$2")"
    [[ "${from%.*}" == "${to%.*}" && "$from" != "$to" ]]
}

# An incompatible on-disk format change makes every existing index
# unreadable, so it cannot ship in a patch release: users upgrading a
# patch expect not to reindex. Comparing against the previous release is
# only possible once there is one, and only when the magic can be read at
# both ends -- an unreadable end is "unknown", not "changed".
check_on_disk_format() {
    local version="$1" prev now was
    prev="$(previous_release_tag)"
    if [[ -z "$prev" ]]; then
        log "on-disk format: no previous release to compare against"
        return 0
    fi

    now="$(meta_format_version)"
    was="$(meta_format_version "$prev")"
    if [[ -z "$now" || -z "$was" ]]; then
        local where=""
        [[ -n "$now" ]] || where="HEAD"
        [[ -n "$was" ]] || where="${where:+$where and }$prev"
        warn "cannot read the index format version at $where --" \
            "skipping the on-disk format check"
        return 0
    fi

    if [[ "$now" == "$was" ]]; then
        log "on-disk format: unchanged since $prev (0x$now)"
        return 0
    fi

    log "on-disk format: changed since $prev (0x$was -> 0x$now)"
    if is_patch_bump "${prev#v}" "$version"; then
        die "$version is a patch release, but the index format changed" \
            "since $prev (0x$was -> 0x$now): existing indexes cannot be" \
            "read, so this needs a minor or major release"
    fi
    log "on-disk format: $version is not a patch release, so the" \
        "format change is allowed"
}
