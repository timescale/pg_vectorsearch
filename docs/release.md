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
`meson.build`'s `project(version: files('VERSION'))`. Everything else
derives from it at build time:

| derived | from |
|---|---|
| `pg_vectorsearch.control` (`default_version`) | generated |
| `pg_vectorsearch--<version>.control` (`module_pathname`) | generated |
| `pg_vectorsearch--<version>.sql` | generated |
| `pg_vectorsearch-<version>.so` | version-named library |
| `pg_vectorsearch_version()`, the CLI banner | `VS_VERSION` |

Nothing else should ever hardcode a version. `scripts/release-lib.sh`
exposes `project_version()` for scripts that need it.

**Between releases, `VERSION` carries a `-dev` suffix.** `main` sits at
`X.Y.Z-dev`; a release flips it to `X.Y.Z`; the next cycle reopens at the
following `-dev`. That convention is what makes a release visible as a
transition rather than an ordinary edit, and it is what the publish
pipeline will key on.

A prerelease version also turns on an install-time notice: `CREATE
EXTENSION` warns that upgrading from a prerelease may not be possible.
It is a runtime check against `pg_vectorsearch_version()`, so a final
release carries nothing to strip.

## Upgradability policy

- **Side-by-side versions.** The library is version-named and each
  version installs its own secondary control file, whose
  `module_pathname` PostgreSQL uses to resolve `MODULE_PATHNAME` in that
  version's scripts. Several versions can be installed in one cluster and
  databases upgrade independently. C symbols may therefore change freely
  between versions; the obligation is that an upgrade script repoints
  *every* C function at the new version's library.
- **Prereleases carry no upgrade guarantee.** Whether a given pair gets
  an upgrade path is decided case by case: if the delta is worth
  shipping, ship it; otherwise users reinstall. Each release's notes say
  which, and the install script warns at `CREATE EXTENSION` time.
- **Reinstalling means losing indexes.** `DROP EXTENSION
  pg_vectorsearch CASCADE` takes the `prism` access method with it, and
  therefore every index built with it, plus the `vec32`/`vec16` types and
  any dependent table columns. Dump or recreate those columns.

## On-disk format audit

Before every release, check whether any on-disk format changed during
the cycle, and that the metapage magic's embedded version byte
(`PRISM_META_MAGIC` in `src/pg/meta.h`) was bumped to match. A format
change without a bump means an old index is read as if it were the new
format.

## Cutting a release PR

`scripts/prepare-release.sh` runs twice, with the release notes written
by hand in between.

```bash
./scripts/prepare-release.sh 0.2.0    # branch + changelog scaffold
$EDITOR CHANGELOG.md                  # fill in the FILL-IN sections
./scripts/prepare-release.sh 0.2.0    # checks, commit, open the PR
```

The first run requires a clean tree on an up-to-date base, then creates
`chore/release-0.2.0`, sets `VERSION`, and writes the `## [0.2.0]`
changelog entry with `.release-notes-template.md` spliced in beneath it.

`git-cliff` generates the entry's `### Changes` list from the
conventional commits since the previous release tag. That list is raw
material, not a contract: it is generated once, here, so **trimming it to
the notable items is part of editing the release PR**. The entry's
compare link still covers the full delta. The very first release has no
previous tag to delta against, so it gets the heading alone and no
generated list.

Fill in every `FILL-IN` placeholder. The second run refuses while any
remain, then runs `scripts/ci/release-guards.sh`, commits, and opens the
PR with the `release` label and the matching `Release <version>`
milestone (warning and continuing if no milestone matches).

The PR body records the development version the cycle should reopen at —
a minor bump by default, a patch bump on an `X.Y.x` maintenance branch.
Edit it in the PR to request something else, such as a major bump, or
pass `--next` when preparing.

`--dry-run` shows what would happen and still runs the guards, since
those only read.

### The guards

`scripts/ci/release-guards.sh` also runs as a PR check, via
`scripts/ci/release-check.sh` and
`.github/workflows/release-check.yml`, so a release PR shows green for
the things ordinary CI cannot see:

- the version string is valid, and matches `VERSION`;
- no `v<version>` tag exists yet, locally or on `origin`;
- `CHANGELOG.md` has an entry for it with no `FILL-IN` left;
- `README.md` and `docs/` carry no stale version references.

It runs locally too, against the checked-out `VERSION`:

```bash
./scripts/ci/release-guards.sh
```

Which PRs get checked is decided by what the PR *does*, not by what its
branch is called: `release-check.sh` looks for a PR that changes
`VERSION` to a value with no prerelease suffix. Anything else — an
ordinary PR, or one reopening a development cycle — reports success
without running the guards. Keying on the branch name instead would let a
release PR cut from a differently-named branch skip its guards silently,
which is the one PR where that matters most.

## Branches

Minor and major releases happen on `main`. Patch releases happen on an
`X.Y.x` maintenance branch, where `prepare-release.sh` defaults the next
development version to a patch bump instead of a minor one.
