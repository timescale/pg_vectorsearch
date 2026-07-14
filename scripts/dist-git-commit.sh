#!/bin/bash
# meson dist script: bake the packaged commit into the tarball.
#
# A distribution tarball has no .git, so vcs_tag's fallback would
# report 'unknown' from git_commit(). Writing the real hash into the
# template inside the dist tree makes the identity survive: vcs_tag's
# @VCS_TAG@ replacement then simply finds nothing to substitute.

set -euo pipefail

commit="$(git -C "$MESON_SOURCE_ROOT" rev-parse HEAD)"
sed -i "s/@VCS_TAG@/$commit/" "$MESON_DIST_ROOT/src/git_commit.h.in"
