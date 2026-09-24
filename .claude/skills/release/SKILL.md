---
name: release
description: >
  Cut a release PR with scripts/prepare-release.sh, then fill the
  changelog and open the PR with the script that run wrote. Also
  resume notes on an existing release branch without recutting.
  Trigger when asked to cut a release, prepare a release PR, or fill
  release notes.
---

# Cutting a release

`scripts/prepare-release.sh` cuts the branch, writes the changelog
entry, and leaves a script that opens the PR. This skill drives that
script. It does not reimplement it. The human procedure is
`docs/release.md`.

The agent's job is the notes: a short, human summary, and a changes
list trimmed to what a user would care about. Then commit that, fix
the generated PR body, and open the PR with the generated script.

The interactive path in `docs/release.md` opens the PR with `FILL-IN`
still in the notes and finishes them there. This skill fills them
before opening, so the PR does not start life failing the release
check. The check is the same either way: no `FILL-IN` at merge.

## If the release branch already exists

Someone asking to fill the notes, or to continue a cut, is not asking
for a new cut. If `chore/release-<version>` exists, or the checkout
is already on it, skip section 2. Check that branch out and continue
at section 3. Do not pass `--force` to get a fresh `open-pr.sh`.
That recreates the branch from the base and discards the notes.

If the PR is already open, amend the release commit. A second commit
fails the release check. Do not call `gh pr create`. If `open-pr.sh`
from the cut is still around, run it: it force-pushes by itself once
the branch is on the remote, and refreshes the body so it picks up the
highlights. If it is gone, push the amend with `--force-with-lease`
and stop. The changelog is the notes. Do not rebuild the `gh`
invocation to refresh the body.

If the PR was never opened and `open-pr.sh` is gone, stop and say so.
Recutting would throw the notes away.

## 1. Before running anything

Work from a clean `main`, or an `X.Y.x` branch for a patch release,
and only when cutting a new release. The base must match its remote.
The script fetches and refuses a local HEAD that differs.

Tracked changes are refused; they would ride onto the release branch.
Untracked files are not, so look at them first. If one belongs in the
release, stop. It has to reach the base branch through the normal
pull-request workflow, and that branch has to be in sync with its
remote, before the release is cut. A local commit on `main` is not
that. This repo does not commit directly to `main`, and the script
would refuse the cut until the remote had the same commit. If none
of the untracked files belong, pass `--ignore-untracked-files`.
Closing stdin already skips that question; the flag records the
decision instead of relying on it.

`VERSION` must end in `-dev`. The release version is that line with
the suffix removed. The next cycle is a minor bump on `main`, or a
patch bump on an `X.Y.x` branch. Use versions the user named.
Otherwise use those defaults and say which pair you are cutting
before you run the script.

`git-cliff` must be on `PATH`. `cliff.toml` pins the version it
targets.

Do not pass `--force` unless the user asked to recut that version.
An existing `chore/release-<version>` is someone else's work until
they say otherwise. A version that already has a `v<version>` tag
cannot be recut.

## 2. Prepare the branch, and stop

Do not pass `--create-pr`. Closing stdin keeps the run from prompting
on a terminal and from opening the PR:

```bash
./scripts/prepare-release.sh \
  --version "$VERSION" \
  --next-version "$NEXT" \
  --ignore-untracked-files \
  </dev/null
```

That creates `chore/release-<version>`, sets `VERSION`, prepends a
changelog entry, commits, and writes `open-pr.sh` plus `body.md`
under a temp directory. The path is in the script's output. Keep it.

Do not run `prepare-release.sh` again after this. From the release
branch it refuses: `VERSION` is no longer a `-dev` version, so there
is nothing to release. `--force` from the base branch is the real
risk. It recreates the branch from that base and discards the notes
just written. Later edits amend the release commit. They are not new
commits.

## 3. Write the notes

Edit the new `CHANGELOG.md` entry. `git-cliff` already wrote
`### Changes` from the commit history. The sections above it still
contain `FILL-IN`. The release check fails while any remain.

Write for users. Short. Say what they can do now, or what they must
do to upgrade. Not a commit recap, and not a tour of the branch.

- **Highlights.** A few sentences, or a short bullet list if there
  are several things. One release usually has one point. Skip
  refactors, CI, and naming.
- **Breaking changes.** `None.` if there are none. If the script
  inserted an "Indexes must be rebuilt" bullet, keep it. Do not
  soften it.
- **Deprecations, known issues, thanks.** Delete the whole section
  when it is empty. Do not leave a placeholder, and do not write
  "None." in a section the template says to delete.
- **Changes.** Raw material. Cut it down to the notable items. The
  first release has no previous tag, so the list is the whole
  history and must be cut hard. Leave the compare link if there is
  one. Dropping a bullet does not drop the change.

No agent voice. No "this release introduces", no "we are excited",
no "delivers", no semicolon-heavy sentences. Plain words. If a
sentence could be shorter, shorten it. Do not narrate how the
release was cut.

## 4. Amend the release commit

The release PR must be one commit. The release check fails if it is
not. Do not add a notes commit.

Amend the commit `prepare-release.sh` just made. Keep its subject,
`chore: release 0.2.0`, and keep `Next-Version` as the last
paragraph. The trailer is the next cycle, not the version just
released. Releasing `0.2.0` from `main` records `0.3.0-dev`. A patch
release from an `X.Y.x` branch records the next patch, such as
`0.2.1-dev`.

```
chore: release 0.2.0

Next-Version: 0.3.0-dev
```

```bash
git add CHANGELOG.md
git commit --amend --no-edit
```

`--no-edit` keeps the trailer the script already wrote. If the
trailer itself must change, amend with a message that still ends
with it. A squash merge reads the trailer from the PR body instead,
so the copy in step 5 still has to match.

If that commit is already on the remote, force-push it. A normal
push is rejected, and a second commit is what the check rejects.

```bash
git push --force-with-lease
```

Never a bare `--force`. Stage `CHANGELOG.md` by name. Run the release
check against the base branch before going on:

```bash
./scripts/ci/release-check.sh main     # or the X.Y.x branch
```

It must pass. A remaining `FILL-IN`, or a second commit, means the
release is not done. Pass the base: counting commits needs something to
count from, so without it the single-commit assertion does not run and
the check reports success on a two-commit branch.

## 5. Fix the PR body, then open the PR

`body.md` was written before the notes existed, so it has no
highlights. Rewrite that file. Do not write a second summary. Copy
the highlights section, without its heading, and keep the trailer as
the last paragraph:

```markdown
Releases `0.2.0`.

## Highlights

<highlights, as written in CHANGELOG.md>

Next-Version: 0.3.0-dev
```

`Next-Version` must be the value passed to the script, not the
version being released, and it must end the file. A squash merge
uses this body as the commit message,
and git only reads a trailer in the last paragraph. No `FILL-IN` in
the body. That string would land on `main`.

Then run the generated script, not `gh pr create` and not
`prepare-release.sh --create-pr`:

```bash
"$PR_DIR/open-pr.sh"
```

It pushes `chore/release-<version>` and opens the PR with the title,
label, milestone, and the body file you just edited. Do not edit the
script: it asks the remote whether the branch is already there and
force-pushes with a lease only then, so a first push and a re-push
after an amend both work. Re-running it refreshes the body instead of
creating a second PR. It does not add a second commit.

If the notes are uncertain, stop before this step and ask.

Opening the PR is the last thing this skill does. Merging it is what
publishes the release. Reopening the development cycle — bumping
`VERSION` back to a `-dev` — is a separate step, in `docs/release.md`.
Do not do it from here, and do not treat an open PR as a finished
release.
