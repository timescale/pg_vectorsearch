# Releasing pg_vectorsearch

How a release is versioned and how a release PR is cut.

> Publishing is not wired up yet. Merging a release PR currently does
> nothing automatic: no tag, no GitHub release, no artifacts. That
> pipeline is the next step; until it lands, a merged release PR just
> means `main` carries a release version.

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

Then it proposes both versions from `VERSION` — the release is that
version with its suffix removed, the next cycle a minor bump (a patch
bump on an `X.Y.x` maintenance branch) — and asks you to confirm or edit
them. With no terminal it takes the defaults, so automation needs no
flags, and `--version` and `--next-version` override either.

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

To preview a release, run it and answer no. That leaves the real branch,
the real changelog entry — generated commit list included — and a PR
script you can read, which is strictly more than a rehearsal could show.
Then either run the script or drop the branch.

The script refuses to run once its branch is gone, rather than failing
inside `git push`, so a stale one left in `/tmp` cannot surprise you.

The PR gets the `release` label and the matching `Release <version>`
milestone (warning and continuing if none matches), and its body records
the next development version, where a reviewer can change it to request
something else such as a major bump.

**The release notes are finished in the PR, not before it.** The entry is
written with `.release-notes-template.md`'s `FILL-IN` placeholders still
in it, and the release check fails while any remain — so an unfinished
release cannot merge. `git-cliff` generates the entry's `### Changes`
list from the conventional commits since the previous release tag; that
list is raw material, so trimming it to the notable items is part of
editing the PR. The compare link still covers the full delta. The first
release has no previous tag to delta against, so it gets the heading
alone and no generated list.

## The release check

`.github/workflows/release-check.yml` runs `scripts/ci/release-check.sh`
on every PR. It asserts:

- the version string is valid;
- no `v<version>` tag exists yet, locally or on `origin`;
- `CHANGELOG.md` has an entry for it with no `FILL-IN` left;
- `README.md` and `docs/` carry no stale version references;
- an incompatible on-disk format change is not shipping as a patch.

Which PRs get checked is decided by what the PR *does*, not by what its
branch is called: a release PR is one that changes `VERSION` to a value
with no prerelease suffix. Anything else — an ordinary PR, or one
reopening a development cycle — reports success without checking
further. Keying on the branch name would let a release PR cut from a
differently-named branch skip its checks silently, which is the one PR
where that matters most.

It runs locally too, against the current checkout:

```bash
./scripts/ci/release-check.sh
```

## Branches

Minor and major releases happen on `main`. Patch releases happen on an
`X.Y.x` maintenance branch, where `prepare-release.sh` defaults the next
development version to a patch bump instead of a minor one.
