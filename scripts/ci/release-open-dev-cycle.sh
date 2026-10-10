#!/bin/bash
# Reopen the development cycle after a release: bump VERSION to the next
# -dev value on chore/dev-<version>, open its pull request, and arm
# auto-merge so it lands once the checks and review are in.
#
# Usage: ./scripts/ci/release-open-dev-cycle.sh <released> [next-dev]
#            [base-branch]
#
# next-dev defaults to the Next-Version trailer on the checked-out
# commit, which is where it comes from anyway: the gate reads the same
# trailer, and prepare-release.sh is what wrote it. Pass it only to
# override, as a standalone re-run might.
#
# base-branch is where the release landed and where the bump goes back.
# It has to be passed in CI: the release job checks out a sha, so the
# checkout is detached and the branch name is not recoverable from it.
#
# With APP_SLUG set, the commit is attributed to that App's bot account
# and the remote is pointed at an authenticated URL built from
# GH_TOKEN. Both are skipped when it is unset, so running this by hand
# neither rewrites the operator's git identity nor puts a token in
# their remote.
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

[[ $# -ge 1 && $# -le 3 ]] ||
    die "usage: $0 <released-version> [next-dev-version] [base-branch]"

RELEASED="$1"
NEXT="${2:-}"
BASE="${3:-$(git rev-parse --abbrev-ref HEAD)}"

# The commit the release was cut from declares the cycle to reopen.
if [[ -z "$NEXT" ]]; then
    NEXT="$(read_next_version_trailer HEAD)"
    [[ -n "$NEXT" ]] ||
        die "$(git rev-parse --short HEAD) has no Next-Version trailer," \
            "so it is not a release commit -- pass the next version" \
            "explicitly to bump anyway"
    log "the release commit declares $NEXT"
fi

BRANCH="chore/dev-$NEXT"

[[ "$BASE" != HEAD ]] ||
    die "the checkout is detached, so the base branch cannot be" \
        "inferred -- pass it as the third argument"

validate_version "$RELEASED"
validate_version "$NEXT"

is_dev_version "$NEXT" ||
    die "next version $NEXT carries no -dev suffix, so the gate would" \
        "read the bump as another release"

# Where the cycle may reopen, by the same rule the release PR's trailer
# check applied before this ran.
check_next_version_order "$RELEASED" "$NEXT"

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

# Write as the App, when running as one. An unattributed commit can
# draw the ruleset's extra-approval requirement, which would stall the
# bump; and the push needs a credential in the remote, because the
# checkout deliberately keeps none.
if [[ -n "${APP_SLUG:-}" ]]; then
    log "committing as ${APP_SLUG}[bot]"
    bot_id="$(gh api "/users/$APP_SLUG%5Bbot%5D" --jq .id)"
    git config user.name "${APP_SLUG}[bot]"
    git config user.email \
        "$bot_id+${APP_SLUG}[bot]@users.noreply.github.com"
    : "${GH_TOKEN:?APP_SLUG is set but GH_TOKEN is not}"
    # Only a GitHub remote, and only to add the credential the checkout
    # deliberately did not keep. Rewriting any other remote would aim
    # the push at github.com rather than where it actually points.
    remote="$(release_remote)"
    if [[ "$(git remote get-url "$remote")" == *github.com* ]]; then
        git remote set-url "$remote" \
            "https://x-access-token:$GH_TOKEN@github.com/$(repo_slug)"
    else
        warn "remote $remote is not on github.com: pushing to it as it" \
            "stands"
    fi
fi

log "opening the $NEXT development cycle on $BRANCH"

git switch -c "$BRANCH"
set_version "$NEXT"
git add VERSION
git commit -m "chore: open the $NEXT development cycle" \
    -m "$RELEASED is tagged, so $BASE carries a version that has
already been cut. Until this lands, anything merged is built and
tested as $RELEASED.

Only VERSION changes; release-dev-bump-check asserts that."

git push "$(release_remote)" "HEAD:refs/heads/$BRANCH"

url="$(gh pr create \
    --repo "$(repo_slug)" \
    --base "$BASE" \
    --head "$BRANCH" \
    --title "chore: open the $NEXT development cycle" \
    --body "$RELEASED is tagged, so \`$BASE\` currently carries a
version that has already been cut. This bumps \`VERSION\` to
\`$NEXT\` so anything merged from here is built and tested as the next
development version.

Only \`VERSION\` changes. The \`release-dev-bump-check\` job asserts
that, that the new value carries a \`-dev\` suffix, and that it is the
correct bump of \`$RELEASED\`.

Opened by the release workflow after tagging \`v$RELEASED\`. The
Releases entry is published once this and every other step has
succeeded.")"

log "opened $url"

# Best effort: auto-merge needs the repository setting enabled, and a
# release must not be reported as failed because the bump waits for a
# human instead of landing on green.
if gh pr merge --repo "$(repo_slug)" --auto --squash "$url"; then
    log "auto-merge armed; it lands once checks and review pass"
else
    warn "could not arm auto-merge on $url -- merge it by hand"
fi
