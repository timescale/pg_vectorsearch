#!/bin/bash
# Static security lint of the extension's install-time SQL with pgspot.
#
# pgspot flags the search_path / privilege-escalation classes that the
# install script (and any upgrade scripts) must avoid: functions without a
# pinned search_path (PS005) and unqualified builtins in dynamic SQL
# (PS016), plus outright errors. See .claude/skills/security-review for the
# full threat model.
#
# Usage:
#   ./scripts/ci/pgspot.sh [sql-file ...]
#
# With no arguments, checks every SQL script under sql/ (the install script
# and any upgrade scripts), so new scripts are covered without editing this.
#
# pgspot must be on PATH: pip install pgspot

set -euo pipefail

# PS017 (unqualified reference to the extension's OWN objects) is the
# standard PostgreSQL idiom: meerkat references its own types and operators
# unqualified on purpose, relying on the forced install search_path. It is
# not an escalation vector and cannot be silenced without @extschema@, which
# a possibly-relocatable extension must avoid. pgvector trips ~50 of the same
# and cannot remove them either. Everything else pgspot reports stays
# blocking -- Errors, PS005, PS016, and anything new.
IGNORE_CODES=(PS017)

if ! command -v pgspot >/dev/null 2>&1; then
    echo "ERROR: pgspot not found on PATH. Install with: pip install pgspot" >&2
    exit 1
fi

# Files to check: explicit args, else every SQL script under sql/.
if [[ $# -gt 0 ]]; then
    files=("$@")
else
    shopt -s nullglob
    files=(sql/*.sql)
    shopt -u nullglob
fi

if [[ ${#files[@]} -eq 0 ]]; then
    echo "ERROR: no SQL files to check" >&2
    exit 1
fi

ignore_args=()
for code in "${IGNORE_CODES[@]}"; do
    ignore_args+=(--ignore "$code")
done

status=0
for f in "${files[@]}"; do
    echo "==> pgspot $f"
    if ! pgspot "${ignore_args[@]}" "$f"; then
        status=1
    fi
done

if [[ $status -ne 0 ]]; then
    echo "pgspot found issues (see above). PS017 own-object references are" \
         "ignored by design; everything else is blocking. See" \
         ".claude/skills/security-review for how to interpret and fix these." >&2
fi
exit "$status"
