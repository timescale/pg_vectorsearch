#!/bin/bash
# Submit a cov-build capture to Coverity Scan and fail unless Scan accepts it.
#
# Usage: ./scripts/ci/coverity-upload.sh <cov-int.tgz>
#
# Environment:
#   COVERITY_TOKEN      Project token from Scan's project settings.
#   COVERITY_EMAIL      A Scan project member's address.
#   COVERITY_PROJECT    Project name as registered, e.g. owner/repo.
#   COVERITY_VERSION    Version string recorded with the build.
#   COVERITY_DESC       Free-text description recorded with the build.
#
# Scan answers every upload with HTTP 200 and a sentence. Only "Build
# successfully submitted." means the build entered the queue; anything else
# (a build already queued, a quota exhausted, a rejected token) is a
# failure, so the sentence is checked rather than the status code.

set -euo pipefail

archive=${1:?usage: coverity-upload.sh <cov-int.tgz>}
: "${COVERITY_TOKEN:?COVERITY_TOKEN is not set}"
: "${COVERITY_EMAIL:?COVERITY_EMAIL is not set}"
: "${COVERITY_PROJECT:?COVERITY_PROJECT is not set}"
: "${COVERITY_VERSION:?COVERITY_VERSION is not set}"
COVERITY_DESC=${COVERITY_DESC:-}

# The project name goes in the URL; slashes must be escaped.
project_url=$(printf '%s' "$COVERITY_PROJECT" | sed 's#/#%2F#g')

echo "==> Submitting $(du -h "$archive" | cut -f1) capture of $COVERITY_PROJECT ($COVERITY_VERSION)"
response=$(curl --silent --show-error --fail-with-body \
    --form "token=$COVERITY_TOKEN" \
    --form "email=$COVERITY_EMAIL" \
    --form "file=@$archive" \
    --form "version=$COVERITY_VERSION" \
    --form "description=$COVERITY_DESC" \
    "https://scan.coverity.com/builds?project=$project_url")

echo "==> Scan replied: $response"
if [[ $response != *"successfully submitted"* ]]; then
    echo "Scan did not accept the build." >&2
    exit 1
fi
