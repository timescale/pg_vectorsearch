#!/bin/bash
# Shared helpers for the release scripts (scripts/release.sh and
# scripts/ci/release.sh). Sourced, not executed. Everything derives the
# extension name and version from meson.build — the single source of
# truth — rather than hardcoding them.

# shellcheck shell=bash

log()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }

require_repo_root() {
    [[ -f meson.build && -d .git ]] ||
        die "run from the repository root"
}

# Project name from the project() call (first quoted string).
meson_project_name() {
    sed -n "s/^project('\([^']*\)'.*/\1/p" meson.build | head -1
}

# Project version from the project() call.
meson_version() {
    sed -n "s/^  version: '\([^']*\)',\$/\1/p" meson.build | head -1
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

# The GitHub repo slug (owner/name), from gh when available, else from
# the origin remote URL.
repo_slug() {
    if command -v gh >/dev/null 2>&1; then
        gh repo view --json nameWithOwner --jq .nameWithOwner \
            2>/dev/null && return
    fi
    git remote get-url origin |
        sed -e 's#^git@github.com:##' -e 's#^https://github.com/##' \
            -e 's#\.git$##'
}
