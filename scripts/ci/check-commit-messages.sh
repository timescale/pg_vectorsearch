#!/bin/bash
# Reject commit messages that credit an AI coding agent.
#
# Usage: ./scripts/ci/check-commit-messages.sh [<msg-file> | <rev-range>]
#   <msg-file>   Check one commit message read from a file. This is the
#                form pre-commit uses at the commit-msg stage.
#   <rev-range>  Check every commit in a git revision range, e.g.
#                origin/main..HEAD (the default when no argument is given)
#                or "$BASE_SHA..$HEAD_SHA" in CI.
#
# A message fails if it carries a Co-Authored-By trailer naming an AI
# agent or its vendor, or a "Generated with <agent>" footer. Trailers
# crediting human contributors are fine and pass unchanged. This repo's
# commit-message conventions exclude tool attribution: the message should
# describe the change, not the tool that typed it.

set -euo pipefail

# Signals that a Co-Authored-By trailer names an agent rather than a
# person. Model names are deliberately not listed: they change too often
# to track, and the sender address and bot markers identify the agent
# whichever model is named. Bare first names are not matched either, so
# a contributor called Claude or Devin is not flagged.
#
#   - a no-reply sender, which is how agents sign (noreply@anthropic.com,
#     noreply@openai.com, ...). GitHub's private addresses look different
#     (<id>+<login>@users.noreply.github.com) and pass.
#   - a GitHub app account ([bot]) or a self-described bot, agent or
#     assistant
#   - a vendor domain or product name of a coding agent
agent_address='<(no-?reply|noreply)@'
agent_marker='\[bot\]|\b(bot|agent|assistant)\b'
agent_vendor='anthropic\.com|openai\.com|cursor\.com|aider\.chat|codeium\.com|windsurf\.com'
agent_product='claude code|copilot|codex|gemini|\baider\b|windsurf|codewhisperer|amazon q|devin[- ]ai'
agent="${agent_address}|${agent_marker}|${agent_vendor}|${agent_product}"

# Extended regexes, matched case-insensitively against each message line.
# The footer form is the "Generated with <tool>" line some agents append.
forbidden=(
    "^Co-Authored-By:.*(${agent})"
    "^(🤖 )?Generated (with|by) .*(${agent}|https?://)"
)

pattern=$(IFS='|'; echo "${forbidden[*]}")

# Print the offending lines of one message, prefixed with a label.
# Returns 0 if the message is clean, 1 otherwise.
check_message() {
    local label=$1 message=$2 hits
    hits=$(grep -inE "$pattern" <<<"$message" | sed 's/^/    /' || true)
    if [[ -n $hits ]]; then
        echo "==> $label" >&2
        echo "$hits" >&2
        return 1
    fi
}

if [[ $# -gt 0 && -f $1 ]]; then
    # Message file: strip the comment lines git adds to the editor template.
    message=$(grep -v '^#' "$1" || true)
    if ! check_message "commit message" "$message"; then
        echo "Remove the line(s) above and commit again." >&2
        exit 1
    fi
    echo "==> Commit message credits no AI agent"
    exit 0
fi

range=${1:-origin/main..HEAD}
failed=0
checked=0
while read -r sha; do
    checked=$((checked + 1))
    subject=$(git log -1 --format=%s "$sha")
    body=$(git log -1 --format=%B "$sha")
    if ! check_message "$(git rev-parse --short "$sha") $subject" "$body"; then
        failed=1
    fi
done < <(git rev-list --reverse "$range")

if [[ $failed -ne 0 ]]; then
    echo "Rewrite the commit(s) above to drop the line(s)." >&2
    exit 1
fi

echo "==> All $checked commit message(s) in $range credit no AI agent"
