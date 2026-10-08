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

    # Without this a malformed argument produces a malformed answer
    # rather than an error: 1.2 became 1.3.0-dev, inventing a patch
    # component, and 1.2.3.4 became 1.3.0-dev, dropping one.
    validate_version "$released"

    base="$(strip_prerelease "$released")"

    # A release candidate is a step toward its own version rather than
    # past it, so the cycle reopens at that version: 0.2.0-rc1 is
    # followed by 0.2.0-dev, from which the next release can be
    # 0.2.0-rc2 or 0.2.0. Bumping instead would skip the version the
    # candidate was a candidate for. -dev is excluded because it is the
    # development state, never something that was released.
    if is_prerelease "$released" && ! is_dev_version "$released"; then
        echo "$base-dev"
        return 0
    fi

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

# Does v$1 exist in this repository?
#
# What CI should ask: a checkout that fetched tags has the answer, and
# asking a remote instead would tie every check to one repository slug
# and fail on a fork.
tag_exists_local() {
    git rev-parse -q --verify "refs/tags/v$1" >/dev/null
}

# Does v$1 exist here or on the release remote? For the operator path,
# where a tag pushed by someone else is not local yet.
#
# Fails closed: `git ls-remote --exit-code` returns 2 for "no such ref",
# but any other non-zero is a transport or auth failure. Reading that as
# "absent" would let a release proceed without ever establishing it.
#
# release_remote dies, and a die inside $( ) only kills the subshell --
# which `set -e` will not catch here, because callers test this in an
# `if`. Hence the explicit exit.
tag_exists() {
    local remote rc=0
    tag_exists_local "$1" && return 0
    remote="$(release_remote)" || exit 1
    git ls-remote --exit-code --tags "$remote" "refs/tags/v$1" \
        >/dev/null 2>&1 || rc=$?
    case "$rc" in
        0) return 0 ;;
        2) return 1 ;;
        *) die "cannot check for tag v$1 on $remote (git ls-remote" \
            "exited $rc) -- refusing to assume it is absent" ;;
    esac
}

require_tag_absent() {
    if tag_exists "$1"; then
        die "tag v$1 already exists"
    fi
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

# The repository to release to.
#
# In CI it is the repository the workflow runs in, which is where the
# workflow file came from. Locally it is stated rather than derived from
# a remote, so a misnamed origin cannot aim a release at a fork; --repo
# overrides it for a rehearsal.
: "${RELEASE_REPO:=${GITHUB_REPOSITORY:-timescale/pg_vectorsearch}}"

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
    # Naming the prerelease suffixes is what puts a release above its own
    # candidates: git's version sort otherwise ranks v0.1.0-rc1 over
    # v0.1.0, and the upgrade script and format check anchor on whatever
    # this returns. Listed from least mature, so -dev sorts lowest; the
    # pipeline never tags one, since the gate refuses to release a -dev
    # version at all.
    tags="$(git \
        -c versionsort.suffix=-dev \
        -c versionsort.suffix=-alpha \
        -c versionsort.suffix=-beta \
        -c versionsort.suffix=-rc \
        tag --list 'v[0-9]*' --merged HEAD --sort=-v:refname)"
    [[ -z "$tags" ]] || head -1 <<<"$tags"
}

# True when $2 is only a patch bump away from $1 (same major.minor).
is_patch_bump() {
    local from to
    from="$(strip_prerelease "$1")"
    to="$(strip_prerelease "$2")"
    [[ "${from%.*}" == "${to%.*}" && "$from" != "$to" ]]
}

# Where the reopened cycle has to sit relative to what shipped. Both the
# trailer check on the release PR and the bump after it is published ask
# this, so it lives here rather than once per caller.
#
# A candidate is a step toward its own version rather than past it, so it
# reopens at that version: 0.2.0-rc1 is followed by 0.2.0-dev, from which
# 0.2.0-rc2 or 0.2.0 can still be cut. An equal base is wrong anywhere
# else, where it would leave the cycle on a version already tagged. -dev
# is excluded from the exception on the same grounds next_dev_version
# excludes it: it is the state between releases, never one of them.
check_next_version_order() {
    local released="$1" next="$2" released_base next_base highest

    released_base="$(strip_prerelease "$released")"
    next_base="$(strip_prerelease "$next")"

    if [[ "$next_base" == "$released_base" ]]; then
        if is_prerelease "$released" && ! is_dev_version "$released"; then
            return 0
        fi
        die "releasing $released and reopening at $next, which is the" \
            "same version -- the cycle has to move on"
    fi

    highest="$(printf '%s\n%s\n' "$released_base" "$next_base" |
        sort -V | tail -1)"
    [[ "$highest" == "$next_base" ]] ||
        die "releasing $released but reopening at $next, which is" \
            "behind it -- the next release would collide with an" \
            "existing tag"
}

# The next development version travels as a Next-Version trailer on the
# release commit, and everything downstream trusts it: the gate reads it
# to confirm the commit is a release, and the dev-cycle job bumps VERSION
# to it. A reviewer may edit it to ask for a major bump, so it is checked
# on the pull request, where a mistake costs nothing. After publishing,
# the tag exists and a wrong trailer has already stranded main.
#
# $2 is the branch being released from, which decides the default bump.
check_next_version_trailer() {
    local version="$1" branch="$2" next want
    next="$(git log -1 --format='%(trailers:key=Next-Version,valueonly)' |
        tr -d '[:space:]')"

    [[ -n "$next" ]] ||
        die "the release commit has no Next-Version trailer --" \
            "prepare-release.sh writes one, and the gate refuses to" \
            "release a commit without it"

    validate_version "$next"

    is_dev_version "$next" ||
        die "Next-Version is $next, which has no -dev suffix -- the" \
            "gate would read the bump that lands it as another release"

    check_next_version_order "$version" "$next"

    # Anything other than the default is a deliberate choice, usually a
    # major bump, so say so rather than refuse it.
    want="$(next_dev_version "$version" "$branch")"
    if [[ "$next" != "$want" ]]; then
        warn "Next-Version is $next where the default for $branch is" \
            "$want -- intended for a major bump, wrong otherwise"
    fi

    log "next cycle: $next"
}

# An existing installation moves between versions with an upgrade script:
# without sql/<name>--<prev>--<new>.sql, ALTER EXTENSION UPDATE fails and
# the only way to the new version is dropping the extension, which takes
# the indexes with it. The install script itself needs nothing -- it is
# sql/<name>.sql copied under the versioned name at build time.
#
# Skipped for a first release, which has nothing to upgrade from.
check_upgrade_path() {
    local version="$1" prev from name script
    prev="$(previous_release_tag)"
    if [[ -z "$prev" ]]; then
        log "upgrade path: no previous release to upgrade from"
        return 0
    fi

    from="${prev#v}"
    name="$(meson_project_name)"
    script="sql/$name--$from--$version.sql"

    [[ -f "$script" ]] ||
        die "releasing $version after $from, but $script does not" \
            "exist -- an installation on $from could only reach" \
            "$version by dropping the extension"

    # Listed or not, the file has to be installed; an unlisted one fails
    # at ALTER EXTENSION UPDATE exactly as a missing one does, only
    # later and less obviously.
    grep -q "$name--$from--$version.sql" src/pg/meson.build ||
        die "$script exists but src/pg/meson.build does not list it in" \
            "ext_update_scripts, so it would never be installed"

    log "upgrade path: $script"
}

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
