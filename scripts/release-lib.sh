#!/bin/bash
# Shared helpers for the release scripts, operator-facing and CI alike.
# Sourced, not executed.
#
# The version is read from ./VERSION and the extension name from
# meson.build's project(); nothing here hardcodes either.

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

# Whitespace-tolerant, though meson requires VERSION to be one line.
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

# Semver-ish, and legal as a PostgreSQL extension version.
validate_version() {
    local v="$1"
    [[ "$v" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-z0-9.]+)?$ ]] ||
        die "invalid version '$v' (expected X.Y.Z or X.Y.Z-suffix)"
    [[ "$v" != *--* ]] || die "version must not contain '--'"
}

# Any -suffix: -alpha1, -rc1, -dev.
is_prerelease() {
    [[ "$1" == *-* ]]
}

# A development version -- the state main sits in between releases.
#
# Distinct from is_prerelease because -rc1 and -alpha1 are versions that
# get *released*: they are tagged, published and checked like any other.
# Only -dev means "not a release".
is_dev_version() {
    [[ "$1" == *-dev ]]
}

# The release part of a version: 0.2.0-rc1 -> 0.2.0.
strip_prerelease() {
    echo "${1%%-*}"
}

# The version the next cycle opens at after releasing $1: a minor bump,
# or a patch bump when $2 (default: current branch) is X.Y.x, where only
# patch releases happen.
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

# A version's CHANGELOG.md entry, heading exclusive.
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

# Refuse modified tracked files: they would ride onto the release branch,
# so testing there would not be testing what the PR contains. Untracked
# files are review_untracked's business.
require_clean_tree() {
    local dirty
    dirty="$(git status --porcelain --untracked-files=no)"
    [[ -z "$dirty" ]] || {
        echo "$dirty" >&2
        die "uncommitted changes to tracked files -- commit or stash" \
            "them first"
    }
}

# Untracked files cannot reach the release commit, so they do not block
# one -- but one of them might be a file that should have been committed,
# so they are listed and confirmed rather than passed over. Always
# listed; with nobody to ask, warn and continue.
#
# $1 is 1 to skip the question; $2 is a yes/no callback.
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

# Fails closed: `git ls-remote --exit-code` returns 2 for "no such ref",
# but any other non-zero is a transport or auth failure. Reading that as
# "tag absent" would let a release proceed without ever establishing it.
require_tag_absent() {
    local tag="v$1" remote rc=0
    remote="$(release_remote)"
    git rev-parse -q --verify "refs/tags/$tag" >/dev/null &&
        die "tag $tag already exists locally"
    git ls-remote --exit-code --tags "$remote" "refs/tags/$tag" \
        >/dev/null 2>&1 || rc=$?
    case "$rc" in
        0) die "tag $tag already exists on $remote" ;;
        2) return 0 ;;
        *) die "cannot check for tag $tag on $remote (git ls-remote" \
            "exited $rc) -- refusing to assume it is absent" ;;
    esac
}

# Asked, not asserted: re-cutting a version is legitimate, so the caller
# decides whether to overwrite. What must not happen is a raw
# `git checkout -b` failure after the version prompts.
branch_exists_local() {
    git rev-parse -q --verify "refs/heads/$1" >/dev/null
}

# Fails closed for the same reason as require_tag_absent: a transport
# error read as "no such branch" would skip the --force-with-lease value.
branch_exists_remote() {
    local rc=0
    git ls-remote --exit-code --heads "$(release_remote)" \
        "refs/heads/$1" >/dev/null 2>&1 || rc=$?
    case "$rc" in
        0) return 0 ;;
        2) return 1 ;;
        *) die "cannot check for branch $1 on $(release_remote)" \
            "(git ls-remote exited $rc)" ;;
    esac
}

# The remote's current commit for a branch, or nothing. Reused as a
# --force-with-lease value, so an overwrite refuses if it moved since.
remote_branch_sha() {
    git ls-remote --heads "$(release_remote)" "refs/heads/$1" |
        cut -f1
}

# Is *the* PR for this release open? Head $1 into base $2 -- GitHub
# allows one open PR per head-and-base pair, so matching the head alone
# would skip creating a PR into a different base. A closed PR is an
# abandoned attempt and does not count. Metadata, so never fatal.
open_pr_url() {
    gh pr list --repo "$(repo_slug)" --head "$1" --base "$2" \
        --state open --json url --jq '.[0].url // empty' 2>/dev/null ||
        true
}

# What will pushing this branch disturb? Every open PR on head $1,
# base deliberately unfiltered: a PR follows its head by name, so the
# push rewrites all of them. One into another base must not suppress the
# create, but must not go unmentioned either.
open_prs_on_branch() {
    gh pr list --repo "$(repo_slug)" --head "$1" --state open \
        --json number,baseRefName,url \
        --jq '.[] | "#\(.number) into \(.baseRefName) (\(.url))"' \
        2>/dev/null || true
}

# A stale base would make the generated changelog and the tagged commit
# disagree with what everyone else sees.
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

# Stated, not guessed: gh has no default repository with two GitHub
# remotes, and deriving it from one would let a misnamed origin aim a
# release at a fork. Override for testing against a fork.
: "${RELEASE_REPO:=timescale/pg_vectorsearch}"

repo_slug() {
    printf '%s\n' "$RELEASE_REPO"
}

# owner/repo from any remote URL spelling, .git suffix or not.
remote_slug() {
    sed -E -e 's#\.git$##' -e 's#^.*github\.com[:/]##' <<<"$1"
}

# The remote pointing at RELEASE_REPO, found by URL rather than assumed
# to be "origin" -- in a fork workflow origin is the fork, and pushing
# there while opening the PR upstream would fail with the branch missing.
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

# Refresh an open PR's body. It is generated content, so a stale one just
# means the PR disagrees with its branch; the release notes live in
# CHANGELOG.md and are not at risk. The one edit that can be lost is a
# reviewer's Next-Version, so the old body is saved and a differing
# trailer warned about.
#
# $1 PR url, $2 new body file, $3 this run's next version, $4 backup path.
update_pr_body() {
    local pr="$1" body="$2" want="$3" backup="$4" prev was
    prev="$(gh pr view "$pr" --repo "$(repo_slug)" --json body \
        --jq .body 2>/dev/null)" || prev=""

    if [[ -n "$prev" ]]; then
        printf '%s\n' "$prev" >"$backup"
        was="$(printf '%s\n' "$prev" | git interpret-trailers --parse |
            sed -n 's/^Next-Version: *//p' | head -1)"
        if [[ -n "$was" && "$was" != "$want" ]]; then
            warn "the PR body declares Next-Version: $was, but this run" \
                "used $want -- replacing it; the old body is at $backup"
        fi
    fi

    gh pr edit "$pr" --repo "$(repo_slug)" --body-file "$body"
}

# Idempotent, and never fatal -- a missing label must not stop a release.
ensure_label() {
    local name="$1" color="$2" description="$3" slug labels
    slug="$(repo_slug)"
    if gh label create "$name" --repo "$slug" --color "$color" \
        --description "$description" >/dev/null 2>&1; then
        return 0
    fi
    # Captured rather than piped into `grep -q`, which would SIGPIPE gh.
    labels="$(gh label list --repo "$slug" --json name \
        --jq '.[].name' 2>/dev/null)" || labels=""
    grep -qxF "$name" <<<"$labels" ||
        warn "could not create or find the '$name' label"
    return 0
}

# The "Release X.Y.Z" milestone for a version, falling back to the
# release part (0.2.0-rc1 -> "Release 0.2.0") and then to nothing --
# metadata must never block a release.
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

# The index format version -- the low byte of PRISM_META_MAGIC -- at git
# ref $1, or the working tree. Prints nothing when unreadable, which
# callers must treat as "unknown", never as "changed".
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

# The newest release tag reachable from HEAD, or nothing. Captured before
# slicing: piping many tags into `head -1` SIGPIPEs git, which pipefail
# turns into a failure of this function.
previous_release_tag() {
    local tags
    tags="$(git tag --list 'v[0-9]*' --merged HEAD --sort=-v:refname)"
    [[ -z "$tags" ]] || head -1 <<<"$tags"
}

# True when $2 is only a patch bump away from $1 (same major.minor).
is_patch_bump() {
    local from to
    from="$(strip_prerelease "$1")"
    to="$(strip_prerelease "$2")"
    [[ "${from%.*}" == "${to%.*}" && "$from" != "$to" ]]
}

# A format change makes every existing index unreadable, so it cannot
# ship in a patch release -- a patch upgrade must not require reindexing.
# Skipped without a previous release, or when either end is unreadable.
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
