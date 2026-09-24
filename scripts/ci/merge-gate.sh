#!/bin/bash
# Fail unless every check that ran on this commit, passed.
#
# Usage: ./scripts/ci/merge-gate.sh <sha> [--no-wait]
#
# One required check standing in for all the others, so a ruleset need
# not name every job in every workflow -- and so a job skipped by path
# filtering does not block a pull request it never applied to.
#
# Waits for the checks it can see, then waits once more, because a check
# that has not *started* is indistinguishable from one that will never
# run. Checks appear within seconds of the triggering event, so the
# settle window covers scheduling rather than execution.
#
# Excludes itself: this runs as a check too, and would otherwise wait
# for its own completion.
#
# Checks named in IGNORE are reported but never fail the gate. An
# aggregate gate makes every check mandatory by default, the opposite of
# naming them in a ruleset, so advisory ones are listed here or a flaky
# one blocks every merge.
#
# --no-wait judges the current state, for testing against a commit whose
# checks have long finished.

set -euo pipefail

SELF="${MERGE_GATE_NAME:-Merge gate}"
REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"
SHA="${1:?usage: $0 <sha> [--no-wait]}"
WAIT=1
[[ "${2:-}" != "--no-wait" ]] || WAIT=0

TIMEOUT_S="${MERGE_GATE_TIMEOUT_S:-1800}"
SETTLE_S="${MERGE_GATE_SETTLE_S:-30}"
POLL_S="${MERGE_GATE_POLL_S:-20}"

IGNORE=(
    "Claude Code Review"
    "Claude Code Assistant"
)

is_ignored() {
    local name
    for name in "${IGNORE[@]}"; do
        [[ "$1" != "$name" ]] || return 0
    done
    return 1
}

checks() {
    gh api --paginate "repos/$REPO/commits/$SHA/check-runs" \
        --jq ".check_runs[] | select(.name != \"$SELF\") |
              \"\(.name)\t\(.status)\t\(.conclusion // \"\")\""
}

deadline=$((SECONDS + TIMEOUT_S))
settled=0

while :; do
    all="$(checks)"
    pending="$(awk -F'\t' '$2 != "completed"' <<<"$all" | grep -c . || true)"

    if [[ "$pending" == 0 ]]; then
        [[ "$WAIT" == 1 && "$settled" == 0 ]] || break
        echo "==> nothing pending; settling for ${SETTLE_S}s in case a" \
            "workflow has yet to register"
        sleep "$SETTLE_S"
        settled=1
        continue
    fi

    settled=0
    echo "==> waiting ${POLL_S}s for $pending check(s):"
    awk -F'\t' '$2 != "completed" { printf "      %s (%s)\n", $1, $2 }' \
        <<<"$all"
    [[ $SECONDS -lt $deadline ]] ||
        { echo "ERROR: timed out after ${TIMEOUT_S}s with $pending" \
            "check(s) still running" >&2; exit 1; }
    sleep "$POLL_S"
done

echo "==> merge gate: judging $(grep -c . <<<"$all") check(s) on ${SHA:0:8}"

passed=0 did_not_run=0 ignored=0
failed=()
while IFS=$'\t' read -r name _status conclusion; do
    [[ -n "$name" ]] || continue
    if is_ignored "$name"; then
        printf '      %-42s %s\n' "$name" "ignored ($conclusion)"
        ignored=$((ignored + 1))
        continue
    fi
    case "$conclusion" in
        success)
            printf '      %-42s %s\n' "$name" "passed"
            passed=$((passed + 1)) ;;
        skipped|neutral)
            printf '      %-42s %s\n' "$name" "did not run ($conclusion)"
            did_not_run=$((did_not_run + 1)) ;;
        *)
            printf '      %-42s %s\n' "$name" "DID NOT PASS ($conclusion)"
            failed+=("$name=$conclusion") ;;
    esac
done <<<"$all"

summary="$passed passed, $did_not_run did not run, $ignored ignored,"
summary="$summary ${#failed[@]} did not pass"

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "==> verdict: FAIL -- $summary"
    echo "ERROR: these checks did not pass: ${failed[*]}" >&2
    exit 1
fi
echo "==> verdict: PASS -- $summary"
