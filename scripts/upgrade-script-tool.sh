#!/bin/bash
# Create, check and rename the pending upgrade script.
#
# Usage: ./scripts/upgrade-script-tool.sh <command> [options]
#
# The pending upgrade script is sql/<name>--<from>--<to>.sql whose <to>
# side is a development version. A cycle collects its catalog changes in
# it, and the release renames it for the version it ships as. See
# sql/README.md.
#
# Commands:
#
#   create [--from <version>] [--to <version>] [--dry-run]
#       Create the pending script with a header and no statements, and
#       add it to ext_update_scripts in src/pg/meson.build. Does nothing
#       if it is already there. --from defaults to the last final release
#       tag and --to to the version in VERSION.
#
#   check
#       Report every C symbol that a script in sql/ names and that no
#       PG_FUNCTION_INFO_V1 in src/ defines. One library serves every
#       installed version, so it has to keep the symbols of every script
#       that can still run.
#
#   rename <version> [--dry-run]
#       Rename the pending script, and its ext_update_scripts entry, for
#       the final release <version>. Does nothing if there is no pending
#       script.
#
# --dry-run prints what would change and writes nothing.
#
# It reads the working tree and local tags only: no network, no clean
# tree and no particular branch. The release scripts that call it make
# those checks themselves.

set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck source=scripts/release-lib.sh
source scripts/release-lib.sh

usage() {
    sed -n '2,/^$/s/^# \{0,1\}//p' "$0"
}

# Write the pending script for the step <from> -> <to> and list it in the
# build. Leaves an existing one alone.
cmd_create() {
    local from="" to="" dry_run=0 tag script pending

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --from) from="${2:?--from needs a version}"; shift 2 ;;
            --to) to="${2:?--to needs a version}"; shift 2 ;;
            --dry-run) dry_run=1; shift ;;
            *) die "create: unknown argument '$1'" ;;
        esac
    done

    if [[ -z "$from" ]]; then
        tag="$(previous_final_release_tag)"
        [[ -n "$tag" ]] ||
            die "no final release tag to upgrade from; pass --from"
        from="${tag#v}"
    fi
    [[ -n "$to" ]] || to="$(project_version)"

    validate_version "$from"
    validate_version "$to"
    ! is_prerelease "$from" ||
        die "--from has to be a final release, not $from"
    [[ "$from" != "$to" ]] || die "cannot upgrade from $from to itself"

    script="$(upgrade_script_path "$from" "$to")"
    pending="$(pending_upgrade_script)"
    [[ -z "$pending" || "$pending" == "$script" ]] ||
        die "a pending upgrade script already exists: $pending"

    if [[ -f "$script" ]] && upgrade_script_listed "$script"; then
        log "$script already exists"
        return 0
    fi

    if [[ $dry_run -eq 1 ]]; then
        if [[ ! -f "$script" ]]; then
            log "would create $script:"
            upgrade_script_header "$from" "$to"
        fi
        upgrade_script_listed "$script" ||
            log "would add $script to ext_update_scripts in" \
                "$UPGRADE_SCRIPT_LIST"
        return 0
    fi

    if [[ ! -f "$script" ]]; then
        upgrade_script_header "$from" "$to" >"$script"
        log "created $script"
    fi
    upgrade_script_list_add "$script"
    log "listed $script in $UPGRADE_SCRIPT_LIST"
}

# Fail if a script in sql/ names a C symbol the library does not define.
cmd_check() {
    local file uses named defined missing symbol status=0

    [[ $# -eq 0 ]] || die "check takes no arguments"

    defined="$({ grep -rhoE 'PG_FUNCTION_INFO_V1\([A-Za-z0-9_]+\)' src ||
        true; } | sed 's/.*(\(.*\))/\1/' | sort -u)"

    for file in sql/*.sql; do
        [[ -f "$file" ]] || continue

        # A use this cannot parse would be a symbol it never checks.
        uses="$(grep -c "AS 'MODULE_PATHNAME'" "$file" || true)"
        named="$(sed -n \
            "s/.*AS 'MODULE_PATHNAME', *'\([A-Za-z0-9_]*\)'.*/\1/p" "$file")"
        if [[ "$uses" -ne "$(grep -c . <<<"$named" || true)" ]]; then
            echo "$file: uses MODULE_PATHNAME $uses time(s), but not" \
                "always as AS 'MODULE_PATHNAME', '<symbol>' on one line" >&2
            status=1
        fi

        missing="$(comm -23 <(sort -u <<<"$named") <(echo "$defined"))"
        for symbol in $missing; do
            echo "$file names the C symbol $symbol, which no" \
                "PG_FUNCTION_INFO_V1 in src/ defines" >&2
            status=1
        done
    done

    [[ $status -eq 0 ]] ||
        die "the library has to keep every symbol a script in sql/" \
            "names; restore it, as a stub that raises an error if the" \
            "function is gone"
    log "every C symbol named in sql/ is defined in src/"
}

# Rename the pending script and its list entry for a final release.
cmd_rename() {
    local version="" dry_run=0 pending from script

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) dry_run=1; shift ;;
            -*) die "rename: unknown option '$1'" ;;
            *)
                [[ -z "$version" ]] || die "rename takes one version"
                version="$1"
                shift
                ;;
        esac
    done
    [[ -n "$version" ]] || die "rename needs the version being released"

    validate_version "$version"
    ! is_prerelease "$version" ||
        die "only a final release renames the pending script, and" \
            "$version is not one"

    pending="$(pending_upgrade_script)"
    if [[ -z "$pending" ]]; then
        log "no pending upgrade script to rename"
        return 0
    fi

    from="$(upgrade_script_from "$pending")"
    [[ "$from" != "$version" ]] ||
        die "$pending already starts at $version"
    script="$(upgrade_script_path "$from" "$version")"
    [[ ! -e "$script" ]] || die "$script already exists"

    if [[ $dry_run -eq 1 ]]; then
        log "would rename $pending to $script, and its entry in" \
            "$UPGRADE_SCRIPT_LIST"
        return 0
    fi

    mv "$pending" "$script"
    if upgrade_script_listed "$pending"; then
        upgrade_script_list_rename "$pending" "$script"
    else
        upgrade_script_list_add "$script"
    fi
    log "renamed $pending to $script"
}

command="${1:-}"
[[ $# -eq 0 ]] || shift

case "$command" in
    create) cmd_create "$@" ;;
    check) cmd_check "$@" ;;
    rename) cmd_rename "$@" ;;
    -h | --help | help) usage ;;
    "") usage >&2; exit 2 ;;
    *) die "unknown command '$command' (create, check or rename)" ;;
esac
