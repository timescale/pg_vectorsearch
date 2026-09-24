# Releasing pg_vectorsearch

How a release is versioned, how a release PR is cut, and what merging
one creates.

> Merging a release PR creates the `v<version>` tag and a GitHub
> Releases entry with the notes. It attaches no files: the only
> downloads are GitHub's own auto-generated source archives. Building a
> named source tarball, announcing, and reopening the development cycle
> are all still to come — see [After a release](#after-a-release).

## Versioning

Semantic versioning, `MAJOR.MINOR.PATCH`, with an optional prerelease
suffix (`-alpha1`, `-rc1`, `-dev`). A version must also be legal as a
PostgreSQL extension version: no `--`, and no leading or trailing `-`.

**The version lives in one file: `VERSION`**, a single line read by
`meson.build`'s `project(version: files('VERSION'))`. A release generates
these from it:

| artifact | carries the version as |
|---|---|
| `pg_vectorsearch.control` | `default_version` |
| `pg_vectorsearch--<version>.control` | `module_pathname` |
| `pg_vectorsearch--<version>.sql` | its filename |
| `pg_vectorsearch-<version>.so` | its filename |
| `pg_vectorsearch_version()`, the CLI banner | `VS_VERSION` |

**Between releases, `VERSION` carries a `-dev` suffix.** `main` sits at
`X.Y.Z-dev`. A release PR flips it to `X.Y.Z` and the next cycle reopens
at the following `-dev`.

`-dev` is the only suffix that means "not a release". `-rc1` and
`-alpha1` are versions that *get* released — tagged, entered in Releases and gated
like any other — so the release check keys on `-dev` specifically rather
than on any prerelease suffix.

**Multiple versions can be installed side by side.** The library is
version-named and each version installs its own secondary control file,
whose `module_pathname` PostgreSQL uses to resolve `MODULE_PATHNAME` in
that version's scripts. Several versions can be installed in one cluster
and databases upgrade independently. C symbols may therefore change
freely between versions; the obligation is that an upgrade script
repoints *every* C function at the new version's library.

A prerelease version also turns on an install-time notice: `CREATE
EXTENSION` warns that upgrading from a prerelease may not be possible.
It is a runtime check against `pg_vectorsearch_version()`, so a final
release carries nothing to strip.

## Releases with a new on-disk format (breaking change)

The low byte of `PRISM_META_MAGIC` (`src/pg/meta.h`) is the index format
version. Bumping it makes an index built by an earlier version get
rejected when opened, so every existing index has to be recreated.

This is checked rather than remembered. Both `prepare-release.sh` and the
release check compare the byte against the previous release tag, and:

- **a patch release is refused.** Patch upgrades must not require
  reindexing, so a format change needs a minor or major version;
- **the release notes get the upgrade note automatically** —
  `prepare-release.sh` writes an "Indexes must be rebuilt" bullet into
  the breaking-changes section, naming the old and new format bytes.

The comparison is skipped when there is no previous release, or when the
byte cannot be read at either end — an unreadable end means *unknown*,
never *changed*.

## Cutting a release PR

```bash
./scripts/prepare-release.sh
```

An agent cutting a release follows the `release` skill, which drives
this script. The skill fills the notes before the PR is opened. The
interactive path below still opens with placeholders and finishes the
notes in the PR. Either way the release check refuses a merge while
a `FILL-IN` remains.

It checks the repository first, before asking anything. Uncommitted
changes to tracked files are refused — they would ride onto the release
branch, so anything you built or tested there would not be what the PR
contains. Untracked files do not block it: having them is normal, and
they can reach neither the release commit nor the tag. But one of them
might be a file that *should* have been committed, so they are listed
and confirmed rather than passed over:

```
WARNING: untracked files present -- check none belong in the release:
    massif.out.36
    results/
Release anyway? [y/N]
```

With no terminal there is nobody to ask, and stray files in a checkout
must not fail a release, so it warns and continues.
`--ignore-untracked-files` skips the question outright; they are still
listed, since knowing what was passed over costs one line.

Then it proposes both versions from `VERSION` — the release is that
version with its suffix removed, the next cycle a minor bump (a patch
bump on an `X.Y.x` maintenance branch) — and asks you to confirm or edit
them. With no terminal it takes the defaults, so automation needs no
flags, and `--version` and `--next-version` override either.

For automation, pass all four flags:

```bash
./scripts/prepare-release.sh --version 0.2.0 --next-version 0.3.0-dev \
    --ignore-untracked-files --create-pr
```

That run never consults a terminal, so it cannot stall on a runner that
happens to allocate one — relying on terminal detection alone would.

It then creates `chore/release-<version>`, sets `VERSION`, writes the
changelog entry and commits. None of that is asked about, because all of
it happens *on the branch* — deleting the branch reverses every bit of
it, and nothing has left the machine. Opening the PR is the step that
cannot be taken back quietly: it makes the release public. So that is the
one question:

```
==> chore/release-0.2.0 is ready: VERSION is 0.2.0 and CHANGELOG.md has its
==> entry (notes still unfinished).
Push chore/release-0.2.0 and open the release PR? [y/N]
```

Answer it up front with `--create-pr`, which is what automation wants.

Either way the `gh` invocation is written to a temporary script first,
and *that* is what runs — so declining leaves something runnable rather
than instructions to retype:

```
==> PR command written to /tmp/pg_vectorsearch-release-0.2.0.XXXXXX/open-pr.sh
==> Branch prepared, PR not opened. To open it, run:

      /tmp/pg_vectorsearch-release-0.2.0.XXXXXX/open-pr.sh
```

It pushes the branch and opens the PR with the title, label, milestone
and body that run worked out, and it is safe to re-run. Because it is
the same file the script executes itself, it cannot drift from what
would have happened.

Re-running it after amending the notes in works: the push uses
`--force-with-lease` once the branch is on the remote, and the lease is
git's default — the remote-tracking ref. Your own push updates that, so
an amend goes through; someone else moving the branch does not, so that
is refused with `stale info`.

To preview a release, run it and answer no. That leaves the real branch,
the real changelog entry — generated commit list included — and a PR
script you can read, which is strictly more than a rehearsal could show.
Then either run the script or drop the branch.

The script refuses to run once its branch is gone, rather than failing
inside `git push`, so a stale one left in `/tmp` cannot surprise you.

### Re-cutting a version

Cutting the same version twice is supported, because rehearsing the flow
means doing it repeatedly and a cut that went wrong is easiest to redo
from scratch. If `chore/release-<version>` already exists — locally, on
the remote, or both — the run says so and asks:

```
WARNING: chore/release-0.1.0 already exists locally and on origin
WARNING: pushing later would force over the commit these open PRs are reviewing:
    #278 into main (https://github.com/timescale/pg_vectorsearch/pull/278)
Recreate chore/release-0.1.0 from main? [y/N]
```

That question is only about the local branch — the push is asked
separately, further down. `--force` answers it up front, for automation.
With no terminal and no flag the answer is **no**: unlike untracked
files, carrying on here would destroy work.

The branch is then recreated with `git checkout -B`, and the push carries
`--force-with-lease` pinned to the commit the check looked at — so it
refuses if the branch moved in between, which is exactly when somebody
else is working on it:

```
 ! [rejected]  chore/release-0.1.0 -> chore/release-0.1.0 (stale info)
```

What makes overwriting safe to offer at all is that the tag check runs
first. A version that has shipped has a `v<version>` tag, so it is
refused whatever the flags:

```
ERROR: tag v0.1.0 already exists locally
```

The push question names what it will actually do when a branch is being
overwritten:

```
Force-push chore/release-0.1.0 over 2c04b345 and update the release PR?
```

A pull request follows its head branch **by name**, so a force-push
rewrites the head of every open PR built on that branch — including one
against a different base, whose diff then becomes meaningless. GitHub
never repoints a PR's base; that only changes when someone changes it.
So every open PR on the branch is listed, and if none of them targets
the base being released from, that is called out, because the run will
open a *new* PR and leave the others pointing at rewritten history:

```
WARNING: pushing later would force over the commit these open PRs are reviewing:
    #278 into feature/release-process (https://github.com/.../pull/278)
WARNING: none of them targets main, so this run opens a new PR and leaves
         the ones above pointing at a branch whose history it rewrote --
         close them first unless you mean that
```

**No duplicate PR is possible.** Before creating one, the run looks for
an open PR from this branch into this base and, finding one, reports it
instead — a force-push has already updated it. `gh pr create` refuses a
duplicate of its own accord as well, so the guard is belt and braces.
The base is part of that lookup, not decoration: GitHub allows one open
PR per head-and-base pair, so a PR from this branch into some *other*
base must not stand in for the one being opened. Only open PRs count; a
closed one is an abandoned attempt, and a fresh PR is the right answer
there.

Re-running with a PR already open **refreshes its body**, so the PR
never disagrees with the branch it is reviewing. The body is generated
content, and the release notes are not at risk: they live in
`CHANGELOG.md` and are edited by committing to the branch.

The one edit that could be lost is a reviewer changing `Next-Version` to
ask for a different bump. That is not discarded silently — the previous
body is saved to the run's temporary directory, and a differing trailer
is named:

```
WARNING: the PR body declares Next-Version: 1.0.0-dev, but this run used
         0.2.0-dev -- replacing it; the old body is at /tmp/.../body-previous.md
```

The PR gets the `release` label and the matching `Release <version>`
milestone, warning and continuing if none matches.

**The PR body says what the PR does, not how releasing works.** It
states the version being released and summarises the highlights — taken
from the changelog entry the run just wrote, so the summary cannot
contradict the notes. The procedure lives here in this document, where
it stays current, instead of being copied into every release PR.

### The next development version

The version the cycle reopens at is declared as a **git trailer** — a
key-value line in RFC 822 header style, in the last paragraph of the
commit message:

```
Next-Version: 0.3.0-dev
```

A trailer rather than prose because the release pipeline reads it back:
`git interpret-trailers --parse`, or
`git log -1 --format='%(trailers:key=Next-Version,valueonly)'`, gets the
value off the commit that landed on `main` with no API call and no
parsing of human text.

The script writes it in two places: the release commit, and the PR
body. Which one survives depends on the merge. A **rebase** merge
keeps the branch commits, and the post-merge step reads the trailer
with `git log -1`, so the tip commit is the one that counts. A
**squash** merge builds a new commit message from the PR body
(`squash_merge_commit_message` is `PR_BODY`), so the body's copy is
the one that lands.

The release PR is one commit. The release check rejects a second.
Filling the notes amends that commit rather than adding another, and
a branch that was already pushed is force-pushed with
`--force-with-lease`. The trailer on that commit and the copy in the
PR body must be the same value. A rebase merge reads the commit. A
squash merge reads the body.

Editing it is how a different bump gets requested — a major one, say.
Change both copies, and keep each one the last paragraph: git only
parses a trailer block that ends the message.

**On the interactive path, the notes are finished in the PR, not before it.**
The `release` skill is the exception: it fills them before opening, so
the PR does not open already failing the check. The entry is
written with `.release-notes-template.md`'s `FILL-IN` placeholders still
in it, and the release check fails while any remain — so an unfinished
release cannot merge. `git-cliff` generates the entry's `### Changes`
list from the conventional commit history; that list is raw material, so
trimming it to the notable items is part of editing the PR. This applies
to the first release too: `--unreleased` means "not reachable from any
tag", which with no tags yet is the whole history rather than nothing.

## The release check

`.github/workflows/release-check.yml` runs `scripts/ci/release-check.sh`
on every PR. It asserts:

- the version string is valid — checked before the PR is classified, so
  a malformed `VERSION` cannot read as a development version and skip
  everything below;
- no `v<version>` tag exists yet, locally or on `origin`;
- the PR is a single commit;
- `CHANGELOG.md` has an entry for it with no `FILL-IN` left;
- `README.md` and `docs/` carry no stale version references;
- an incompatible on-disk format change is not shipping as a patch.

Which PRs get checked is decided by what the PR *does*, not by what its
branch is called: a release PR is one that changes `VERSION` to a value
with no `-dev` suffix. Anything else — an ordinary PR, or one
reopening a development cycle — reports success without checking
further. Keying on the branch name would let a release PR cut from a
differently-named branch skip its checks silently, which is the one PR
where that matters most.

It runs locally too, against the current checkout:

```bash
./scripts/ci/release-check.sh
```

## What merging a release PR creates

Merging a release PR pushes `VERSION` to `main`, and that is what
`.github/workflows/release.yml` triggers on. Nothing else touches that
file, and there are exactly two things that can happen to it — a release
(`0.2.0-dev` → `0.2.0`) or reopening the cycle (`0.2.0` → `0.3.0-dev`) —
so the suffix test is a precise partition rather than a guess.

`CHANGELOG.md` is deliberately not a trigger. It would work under the
current convention, where entries are generated at release time, but
that is a convention rather than a property: if PRs ever append to it
during a cycle, it becomes a noisy path firing the pipeline on every
merge. `VERSION` cannot degrade that way.

Three jobs run in order.

**Gate** (`scripts/ci/release-gate.sh`) first checks the commit is one
`prepare-release.sh` made, by requiring its `Next-Version` trailer.
Everything else it asks is a property of the tree — no `-dev` suffix, a
finished changelog entry, no tag yet — and all of those stay true for
any commit merged between a release and the dev-cycle bump. The trailer
is what identifies the commit rather than the state.

It then answers two separate questions.
*Eligibility* — is this a release at all? A `-dev` version, or one whose
tag already exists, is not, and the run ends quietly with
`release=false`. *Readiness* — is it fit to release? A missing or unfinished
changelog entry fails loudly, because that means this **is** a release
and it is broken.

Keeping them apart is what lets the gate be cheap on ordinary pushes and
strict on real ones. It also means releasing does not depend on
`release-check` having blocked the PR: an unfinished release that reached
`main` anyway stops here rather than shipping.

**Build** builds and tests at the exact commit the gate approved, not at
whatever `main` has moved to since.

**Tag and create the Releases entry** (`scripts/ci/release-create.sh`)
makes exactly two things: the git ref `refs/tags/v<version>` in this
repository, and one GitHub Releases entry against it whose body is the
`CHANGELOG.md` section for that version. It uploads no files. A
prerelease is flagged as one so it does not become the "latest release".
It refuses unless `HEAD` is the sha the build job tested — a release may
only ever tag a verified commit — and it does nothing else, so a later
failure attaching artifacts cannot damage an entry that already
exists.

It reports the release milestone's remaining open issues but never closes
it: closing a milestone that still has open work is a judgement call.

A release that failed after its push event was consumed can be re-run
from the Actions tab (`workflow_dispatch`), giving the version. The gate
asserts that the input matches `VERSION`, so this cannot release a
version the ref does not carry.

## After a release

`main` now sits at a released version, and two things are still manual.

**Reopen the development cycle.** Bump `VERSION` to the next `-dev` —
the release commit says which in its `Next-Version` trailer:

```bash
git log -1 --format='%(trailers:key=Next-Version,valueonly)' origin/main
```

Until that lands, `main` carries a version that has already shipped, so
anything merged in between is built and tested as a released version.

**Nothing is attached to the release.** No source tarball, no checksum.
Installing means building from the tag.

## Branches

Minor and major releases happen on `main`. Patch releases happen on an
`X.Y.x` maintenance branch, where `prepare-release.sh` defaults the next
development version to a patch bump instead of a minor one.
