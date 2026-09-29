#!/bin/bash
# Reopen the development cycle after a release: bump VERSION to the next
# -dev value on chore/dev-<version>, open its pull request, and arm
# auto-merge so it lands once the checks and review are in.
#
# Usage: ./scripts/ci/release-open-dev-cycle.sh <released> <next-dev>
#            [base-branch]
#
# base-branch is where the release landed and where the bump goes back.
# It has to be passed in CI: the release job checks out a sha, so the
# checkout is detached and the branch name is not recoverable from it.
#
# Runs after the release is published, so a failure here leaves a
# complete release and a main branch still carrying the released
# version -- recoverable by the manual bump in docs/release.md.
#
# Pushing and opening the pull request need a credential that is not
# GITHUB_TOKEN. GitHub starts no workflow runs from events that token
# creates, so main's required checks would never report and nothing
# could merge the pull request -- not even by hand.
#
# Idempotent: a branch that already carries an open pull request is
# left alone, so re-running the release workflow does not duplicate it.

set -euo pipefail

# shellcheck source=scripts/release-lib.sh
source "$(dirname "$0")/../release-lib.sh"

require_repo_root

[[ $# -ge 2 && $# -le 3 ]] ||
    die "usage: $0 <released-version> <next-dev-version> [base-branch]"

RELEASED="$1"
NEXT="$2"
BASE="${3:-$(git rev-parse --abbrev-ref HEAD)}"
BRANCH="chore/dev-$NEXT"

[[ "$BASE" != HEAD ]] ||
    die "the checkout is detached, so the base branch cannot be" \
        "inferred -- pass it as the third argument"

validate_version "$RELEASED"
validate_version "$NEXT"

is_dev_version "$NEXT" ||
    die "next version $NEXT carries no -dev suffix, so the gate would" \
        "read the bump as another release"

# Strictly greater than what shipped, or the cycle reopens behind
# itself and the next release would collide with an existing tag.
released_base="$(strip_prerelease "$RELEASED")"
next_base="$(strip_prerelease "$NEXT")"
highest="$(printf '%s\n%s\n' "$released_base" "$next_base" |
    sort -V | tail -1)"
[[ "$highest" == "$next_base" && "$next_base" != "$released_base" ]] ||
    die "next version $NEXT is not greater than the released" \
        "$RELEASED -- the Next-Version trailer names the wrong cycle"

[[ "$(project_version)" == "$RELEASED" ]] ||
    die "VERSION says $(project_version) but this run released" \
        "$RELEASED -- refusing to bump a commit that is not the release"

# An open pull request means a previous run got this far. Two of them
# would both try to bump the same version.
if branch_exists_remote "$BRANCH"; then
    existing="$(open_pr_url "$BRANCH" "$BASE")"
    if [[ -n "$existing" ]]; then
        log "$BRANCH already has an open pull request: $existing"
        exit 0
    fi
    die "$BRANCH is already on the remote with no open pull request" \
        "into $BASE -- delete the branch, or open the pull request by" \
        "hand, then re-run"
fi

log "opening the $NEXT development cycle on $BRANCH"

git switch -c "$BRANCH"
set_version "$NEXT"
git add VERSION
git commit -m "chore: open the $NEXT development cycle" \
    -m "$RELEASED is released and tagged, so main carries a version
that has already shipped. Until this lands, anything merged is built
and tested as $RELEASED.

Only VERSION changes; release-dev-bump-check asserts that."

git push "$(release_remote)" "HEAD:refs/heads/$BRANCH"

url="$(gh pr create \
    --repo "$(repo_slug)" \
    --base "$BASE" \
    --head "$BRANCH" \
    --title "chore: open the $NEXT development cycle" \
    --body "$RELEASED is released and tagged, so \`$BASE\` currently
carries a version that has already shipped. This bumps \`VERSION\` to
\`$NEXT\` so anything merged from here is built and tested as the next
development version.

Only \`VERSION\` changes. The \`release-dev-bump-check\` job asserts
that, that the new value carries a \`-dev\` suffix, and that it is the
correct bump of \`$RELEASED\`.

Opened by the release workflow after publishing \`v$RELEASED\`.")"

log "opened $url"

# Best effort: auto-merge needs the repository setting enabled, and a
# release must not be reported as failed because the bump waits for a
# human instead of landing on green.
if gh pr merge --repo "$(repo_slug)" --auto --squash "$url"; then
    log "auto-merge armed; it lands once checks and review pass"
else
    warn "could not arm auto-merge on $url -- merge it by hand"
fi
