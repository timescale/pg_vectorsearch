---
name: release
description: >
  Drive a release from cut to finished: cut the PR with
  scripts/prepare-release.sh, fill the changelog, open the PR with the
  script that run wrote, then after a human merges it, watch
  release.yml publish, check the tag and assets, and see the
  development-cycle PR through to a -dev VERSION. Also resumes notes
  on an existing release branch without recutting. Trigger when asked
  to cut a release, prepare a release PR, fill release notes, check on
  or monitor a release in progress, or find out whether a release
  finished.
---

# Releasing

Sections 1 to 5 cut the release PR, whether run from a checkout or by
`.github/workflows/release-prepare.yml` ("Cut a release PR"), which
invokes this same procedure as the release App so the pull request it
opens carries the required checks. That workflow installs `git-cliff`,
configures git as the App and passes the versions in; everything else
below applies unchanged.

The user merges it — that is the irreversible step, and it is theirs.
Sections 6 to 8 follow what the merge sets off: the publish, then the
development-cycle PR that puts `VERSION` back on a `-dev` value. A
release is not finished until that lands.

Two files under `reference/` carry the parts that need judgement
rather than sequence: `release-notes.md` and `thanks.md`. Section 3
says when to read them.

`scripts/prepare-release.sh` cuts the branch, writes the changelog
entry, and leaves a script that opens the PR. This skill drives that
script. It does not reimplement it. The human procedure is
`docs/release.md`.

The agent's job in the first half is the notes: a short, human
summary, and a changes list trimmed to what a user would care about.
Then commit that, fix the generated PR body, and open the PR with the
generated script. In the second half it is reporting: which job ran,
what it produced, and what is still outstanding.

The interactive path in `docs/release.md` opens the PR with `FILL-IN`
still in the notes and finishes them there. This skill fills them
before opening, so the PR does not start life failing the release
check. The check is the same either way: no `FILL-IN` at merge.

## If the release branch already exists

Someone asking to fill the notes, or to continue a cut, is not asking
for a new cut. If `chore/release-<version>` exists, or the checkout
is already on it, skip section 2. Check that branch out and continue
at section 3. Do not pass `--force` to `prepare-release.sh` to get
a fresh `open-pr.sh`.
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
that. This repo does not commit directly to `main`, and
`prepare-release.sh` would refuse the cut until the remote had the
same commit. If none of the untracked files belong, pass
`--ignore-untracked-files` to `prepare-release.sh`. Closing its stdin
already skips that question. The flag records the decision instead of
relying on that.

The version in the file `VERSION` must end in `-dev` in the branch you
are cutting a release from. The release version is that line with the
suffix removed. The next cycle is a minor bump on `main`, or a patch
bump on an `X.Y.x` branch. Use versions the user named. Otherwise use
those defaults and say which pair you are cutting before you run
`prepare-release.sh`.

`git-cliff` must be on `PATH`. `cliff.toml` pins the version it
targets.

Do not pass `--force` to `prepare-release.sh` unless the user asked to
recut that version. An existing `chore/release-<version>` branch is
someone else's work until they say otherwise. A version that already
has a `v<version>` tag cannot be recut.

## 2. Prepare the branch, and stop

Do not pass `--create-pr` to `prepare-release.sh`. Closing its stdin
keeps the run from prompting on a terminal and from opening the PR:

```bash
./scripts/prepare-release.sh \
  --version "$VERSION" \
  --next-version "$NEXT" \
  --ignore-untracked-files \
  </dev/null
```

That creates `chore/release-<version>`, sets `VERSION`, prepends a
changelog entry, commits, and writes `open-pr.sh` plus `body.md`
under a temp directory. The path is in `prepare-release.sh`'s output.
Keep it.

Do not run `prepare-release.sh` again after this. From the release
branch it refuses: `VERSION` is no longer a `-dev` version, so there
is nothing to release. `--force` from the base branch is the real
risk. It recreates the branch from that base and discards the notes
just written. Later edits amend the release commit. They are not new
commits.

## 3. Write the notes

Edit the new `CHANGELOG.md` entry. `git-cliff` already wrote
`### Changes` from the commit history. The sections above it still
contain `FILL-IN`, and `release-check.sh` fails while any remain.

This is the part that takes judgement, so the brief is its own file:

- **`reference/release-notes.md`** — who reads a release note and what
  they want from it, where to find what a user can actually do
  differently, how to cut a long commit list down rather than
  reproduce it, the voice to write in and the tells to avoid, and a
  worked example of a good entry beside a bad one.
- **`reference/thanks.md`** — who to credit, which is mostly people
  who reported bugs and who appear nowhere in the commit log, and how
  to tell an outside contributor from the team.

Read the first before writing. Read the second if anyone outside the
team was involved, and delete the Thanks section otherwise.

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

`--no-edit` keeps the trailer `prepare-release.sh` already wrote. If
the
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

`Next-Version` must be the value passed to `prepare-release.sh`, not
the
version being released, and it must end the file. A squash merge
uses this body as the commit message,
and git only reads a trailer in the last paragraph. No `FILL-IN` in
the body. That string would land on `main`.

Then run the generated `open-pr.sh`, not `gh pr create` and not
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

If the notes are uncertain, stop before this step and ask — unless
this is running from `.github/workflows/release-prepare.yml`, where
there is nobody to ask. There, write the best short version and name what you
were unsure about in the pull request body, so the reviewer looks
there first. Never leave `FILL-IN`: a squash merge would put it on
`main`.

Merging the PR is what publishes the release, and it is the user's to
do. Do not merge it, do not approve it, and do not treat an open PR as
a finished release. Once it is merged, continue at section 6.

## What this skill never does

Four things belong to the user, whatever they seem to have asked for:

- **merging** the release PR, which is the irreversible step;
- **approving** any PR, including the development-cycle one this
  process opens — approving your own automation is not review;
- **dispatching** `release.yml`, **creating or deleting a tag**, and
  **editing or deleting a Releases entry**;
- **changing rulesets, secrets or variables.**

Reporting what a step did, and what it would take to fix a failure, is
the job. Doing the irreversible part is not.

## 6. Watch the release publish

Merging pushes `VERSION` to `main`, which triggers `release.yml`. Four
jobs run in order: `gate`, `build`, `release`, `dev-cycle`.
`docs/release.md`, under "What merging a release PR creates", says what
each one does and why they are ordered that way. What matters here is
where a run stopped.

```bash
gh run list --workflow=release.yml -L 1
gh run watch <run-id>
```

Report each job's outcome rather than only the final one, because
where it stopped determines what state the repository is in:

- **gate** refuses: nothing happened. `main` carries the released
  `VERSION` with no tag, which is the broken state the `-dev`
  invariant watches for. The fix goes through a PR like anything else.
- **build** fails: still nothing published, which is the point of
  packaging sitting upstream of tagging.
- **release** fails: check whether the tag exists. `release-create.sh`
  makes the tag and the entry together, and uploads nothing; if the
  tag is there and the assets are not, re-running the job attaches
  them. Do not create or move a tag by hand.
- **dev-cycle** fails: the release is complete and correct. Only the
  bump is missing, and `docs/release.md` has the manual path.

Then confirm what shipped, rather than assuming the green run means it:

```bash
gh release view "v$VERSION"
gh release view "v$VERSION" --json assets \
  --jq '.assets[].name'
```

There should be a tarball and its `.sha256sum`. `scripts/verify-published.sh`,
when it exists, downloads and rebuilds from them.

## 7. The development-cycle PR

The `dev-cycle` job opens it: `chore/dev-<next>`, one commit, `VERSION`
and nothing else. It is not optional — until it lands, `main` carries a
version that has already shipped, and anything merged meanwhile is
built and tested as that release.

```bash
gh pr list --head "chore/dev-$NEXT" --state open
gh pr checks <number>
```

`Dev bump check` is the one to read: it asserts the diff is `VERSION`
alone, that the new value carries `-dev`, and that the branch name
agrees with the file. Auto-merge is already armed, so the PR lands on
its own once the checks pass **and** someone approves — the ruleset
requires one approval, and an App cannot approve its own pull request.

Review it if asked: the diff should be one line. Then say it is waiting
for an approval. Do not approve it.

If the job did not run at all, `RELEASE_APP_CLIENT_ID` is unset, and
the bump is the manual path in `docs/release.md`. Say so rather than
doing it silently.

## 8. Confirm the release is finished

Two things, both after the development-cycle PR merges:

```bash
git fetch origin && git show origin/main:VERSION   # a -dev version
git ls-remote --tags origin | grep "v$VERSION"     # the release tag
```

`VERSION` back on a `-dev` value and the tag present is the finished
state. Until both hold, say which one is outstanding rather than
calling the release done.
