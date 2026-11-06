---
name: release
description: >
  Cut a release PR: draft it with scripts/prepare-release.sh, write
  the release notes, and open the PR with scripts/open-release-pr.sh.
  Also finishes notes on an existing release branch without recutting.
  Merging the PR and what follows it are not this skill's job. Trigger
  when asked to cut a release, prepare a release PR, or fill release
  notes.
---

# Cutting a release PR

Sections 1 to 5 cut the release PR, whether run from a checkout or by
`.github/workflows/release-prepare.yml` ("Cut a release PR"). The
procedure is the same in both: draft, write the notes, open. The
workflow installs `git-cliff`, passes the versions in, and runs the
scripts itself: section 2 as a step before the agent, section 5 in a
second job as the release App, on the two files the agent finished.
The agent there does sections 3 and 4 and stops, so it never holds a
credential that can write.

The PR is where this skill ends. Merging it is the user's step, and
CI does the rest: `release.yml` tags, builds and publishes, then opens
the development-cycle PR. `docs/release.md` describes that under
"What merging a release PR creates".

Two files under `reference/` carry the parts that need judgement
rather than sequence: `release-notes.md` and `thanks.md`. Section 3
says when to read them.

Two scripts do the mechanics. `scripts/prepare-release.sh` checks the
checkout and writes a draft: the changelog with the generated entry
and the pull request body, as two files under a temporary directory.
`scripts/open-release-pr.sh` takes those two files, cuts the branch,
commits, pushes and opens the PR. This skill drives them. It does not
reimplement them. The human procedure is `docs/release.md`.

The agent's job is the notes: a short, human summary, and a changes
list trimmed to what a user would care about.

## If the release branch already exists

Someone asking to fill the notes, or to continue a cut, is not asking
for a new cut. If `chore/release-<version>` exists, or the checkout
is already on it, the PR is open or about to be, and the notes live
in that branch's `CHANGELOG.md`. Edit them there, amend the release
commit, and push with `--force-with-lease`. A second commit fails the
release check. A bare `--force` passes it, and overwrites whatever
someone else pushed meanwhile, so never use one. Do not call
`gh pr create`, and do not run `open-release-pr.sh --force` to get
back to a draft: that recreates the branch from the base and discards
the notes on it.

If the draft directory from a cut that never got to section 5 is
still around, continue at section 3 with it.

## 1. Before running anything

Work from a clean `main`, or an `X.Y.x` branch for a patch release,
and only when cutting a new release. The base must match its remote.
`prepare-release.sh` fetches and refuses a local HEAD that differs.

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

An existing `chore/release-<version>` branch is someone else's work
until they say otherwise. `prepare-release.sh` warns about one, and
`open-release-pr.sh` refuses it unless passed `--force`, which only
the user can ask for. A version that already has a `v<version>` tag
cannot be recut.

## 2. Write the draft

Closing stdin keeps the run from prompting on a terminal:

```bash
./scripts/prepare-release.sh \
  --version "$VERSION" \
  --next-version "$NEXT" \
  --ignore-untracked-files \
  </dev/null
```

That changes nothing in the checkout. It writes `CHANGELOG.md` and
`body.md` under a temp directory and prints the command that opens
the PR from them. The path is in its output. Keep it.

Do not run `prepare-release.sh` again after this: a second run writes
a second draft, and the one being edited is the one that counts.

From `.github/workflows/release-prepare.yml` the draft is already
written, by a step before the agent, and its path is in the prompt.
Skip this section there.

## 3. Write the notes

Edit the draft's `CHANGELOG.md`, not the checkout's. `git-cliff`
already wrote `### Changes` from the commit history. The sections
above it still contain `FILL-IN`, and `open-release-pr.sh` refuses
while any remain.

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

Read the diff of the range `git-cliff` summarised rather than only
its bullet list: the checkout has the history, and `git log` and
`git diff` are what to read it with.

## 4. Write the PR body

The draft's `body.md` was written before the notes existed, so it has
no highlights. Rewrite that file. Do not write a second summary. Copy
the highlights section, without its heading, and keep the trailer as
the last paragraph:

```markdown
Releases `0.2.0`.

## Highlights

<highlights, as written in CHANGELOG.md>

Next-Version: 0.3.0-dev
```

`Next-Version` must be the value passed to `prepare-release.sh`, not
the version being released, and it must end the file. A squash merge
uses this body as the commit message, and git only reads a trailer in
the last paragraph. No `FILL-IN` in the body. That string would land
on `main`.

If the notes are uncertain, stop here and ask — unless this is
running from `.github/workflows/release-prepare.yml`, where there is
nobody to ask. There, write the best short version and name what you
were unsure about in `body.md`, so the reviewer looks there first.
Never leave `FILL-IN`: a squash merge would put it on `main`.

From the workflow, this is where the agent stops. Do not run
`open-release-pr.sh`, push, or call `gh pr create` there: the
workflow's second job runs the script on the two files, as the
release App, on a fresh checkout. Finish by reporting what you left
for the reviewer.

## 5. Open the PR

From a checkout, run the command `prepare-release.sh` printed, which
is `open-release-pr.sh` with the two draft files. Not `gh pr create`:

```bash
./scripts/open-release-pr.sh "$VERSION" "$NEXT" main \
  "$DRAFT/CHANGELOG.md" "$DRAFT/body.md"
```

It cuts `chore/release-<version>` from the base, commits
`CHANGELOG.md` and `VERSION` with the trailer, runs the release check,
pushes, and opens the PR with the title, label, milestone and the
body file. A `FILL-IN` left anywhere fails the check and nothing is
pushed; a refused run puts the checkout back where it started. The
checkout must be clean and on the base; the commit is built from the
two files alone.

Turning a release check off, such as `--disable-check upgrade-path`
for a release that ships without an upgrade path, is the user's
decision. Ask; never write a `Disable-check` line into `body.md`,
which the script refuses anyway.

Merging the PR is what publishes the release, and it is the user's to
do. Report the PR and stop.

## What this skill never does

Four things belong to the user, whatever they seem to have asked for:

- **merging** the release PR, which is the irreversible step;
- **approving** any PR, including the development-cycle one the merge
  opens — approving your own automation is not review;
- **dispatching** `release.yml`, **creating or deleting a tag**, and
  **editing or deleting a Releases entry**;
- **changing rulesets, secrets or variables.**
