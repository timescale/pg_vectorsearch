# Releasing meerkat

This document is the authority on the release process: the policies,
the automated path, and the full manual runbook. The mechanism is
`scripts/release.sh` (operator driver) and `scripts/ci/release.sh`
(the pipeline, identical in CI and locally); this document spells out
every step so a release can be performed — and the scripts audited —
without them.

## Version scheme

Semantic versioning: `MAJOR.MINOR.PATCH`.

- Prereleases carry a suffix: `-alpha1`, `-beta1`, `-rc1`.
- Between releases, `main` carries the next version with a `-dev`
  suffix (e.g. `0.1.0-dev`).
- Version strings must be valid PostgreSQL extension versions: no
  `--`, no leading or trailing `-`.

The version lives in exactly one place: `meson.build`'s project
version. The control files, the versioned install script, the
version-named shared library, the CLI banner, and the SQL
`mkt.extension_version()` function all derive from it at build time.
The install script itself warns at `CREATE EXTENSION` time when
`mkt.extension_version()` reports a prerelease suffix — a runtime
check, so final releases have nothing to strip.

## Upgradability policy

- **Released install scripts are immutable.** A released
  `meerkat--X.Y.Z.sql` is never edited; it is preserved by the git
  tag. Fixes go into the next version.
- **Upgrade scripts** live flat in `sql/` as adjacent-version
  `meerkat--A--B.sql` scripts (PostgreSQL chains them; inspect with
  `SELECT * FROM pg_extension_update_paths('meerkat')`). While an
  upgrade path to the next release is being maintained, every
  catalog-affecting change must land in the same PR as its addition
  to the pending upgrade script.
- **Upgrade equals fresh install:** `ALTER EXTENSION meerkat UPDATE`
  must produce catalog state identical to a fresh
  `CREATE EXTENSION`. A pg_dump-diff CI harness enforcing this is
  planned follow-up work (nothing to test before the first shipped
  upgrade path).
- **Side-by-side versions:** the shared library is version-named
  (`meerkat-<version>.so`) and each release ships a per-version
  secondary control file (`meerkat--<version>.control`) whose
  `module_pathname` PostgreSQL uses to resolve `MODULE_PATHNAME` in
  that version's install and upgrade scripts — so several extension
  versions can be installed in one cluster and databases upgrade
  independently, while all scripts stay version-blind. C symbols may
  change freely between versions; the corresponding obligation is
  that an upgrade script repoints *every* C function to the new
  version's library via version-agnostic
  `CREATE OR REPLACE ... AS 'MODULE_PATHNAME'` statements (see
  `sql/README.md`).
- **Prereleases carry no upgrade guarantee.** Whether a prerelease
  gets an upgrade path is decided case by case per version pair: if
  the delta is worth shipping, ship it; otherwise users reinstall.
  Each release's notes ("Breaking changes & upgrade notes") state
  which it is. The install script warns at CREATE EXTENSION time
  that upgrading from a prerelease might not be possible. When
  reinstalling is the answer, recovery is
  `DROP EXTENSION meerkat CASCADE` + install the new version +
  re-create indexes — and note what the cascade takes with it:
  dropping the extension drops the `mktann` access method and
  therefore every index built with it, and drops the
  `vector`/`halfvec` types and therefore dependent table columns —
  dump/restore or recreate those columns.

### On-disk format audit

Before every release, check whether any on-disk format changed during
the cycle and that the metapage magic's embedded version byte
(`MKT_META_MAGIC` in `src/pg/mktann_meta.h`) was bumped accordingly.
A format bump breaks index compatibility:

- release notes must state that indexes need rebuilding;
- once upgrade tests exist, incompatible from-versions are excluded
  there.

### Upgrade compatibility matrix

Maintained per release; states whether `ALTER EXTENSION UPDATE` works
and what user action is required (rebuild indexes, restart server).

| From | To | Compatible? | Notes |
|------|----|-------------|-------|
| —    | —  | —           | (no releases yet) |

**Bugfix releases:** after releasing a bugfix `X.Y.Z+1`, `main` must
gain an upgrade path from it to the next version so bugfix users are
not stranded (applies once upgrade support starts).

## Release notes

Each release gets an entry in `CHANGELOG.md`, structured by
`.release-notes-template.md`: hand-written prose sections
(Highlights, breaking changes & upgrade notes, deprecations, known
issues, thanks) followed by a `### Changes` list that git-cliff
generates from the conventional commit history since the previous
release. The prose can be drafted by the operator or by Claude via
the `/release` skill; either way it gets human review in the release
PR. The pipeline refuses to ship an entry that still contains a
`FILL-IN` placeholder, and the same entry becomes the GitHub
release-page notes.

The generated `### Changes` list is raw material, not a contract:
git-cliff runs once when the release branch is prepared, and nothing
regenerates or re-validates the list afterwards — trimming it down
to the notable items is part of editing the release PR. The entry's
closing compare link always covers the full delta, so cutting list
items loses nothing. The first release skips the generated list
entirely: there is no previous release to delta against, and the
full pre-release history would drown readers.

## The automated path

```text
./scripts/release.sh prepare X.Y.Z     # release branch + changelog
  (fill in the release notes)
./scripts/release.sh prepare X.Y.Z     # checks + commit
  push, open PR, review
  MERGE = APPROVAL: the release workflow fires on the version flip
    (verify -> release -> artifacts -> announce)
./scripts/release.sh post X.Y.W-dev    # open the next cycle (PR)
./scripts/release.sh verify X.Y.Z      # check published artifacts
```

Merging the release PR is the human approval that publishes: the
release workflow (`.github/workflows/release.yml`) triggers on pushes
to `main` that change `meson.build`, and its gate proceeds only when
the version has no `-dev` suffix and is untagged. Nothing is tagged
until the workflow's own test run passes; the tag and GitHub release
are then created atomically at the verified commit. Artifact upload
and announcements run as separate follow-up jobs so their failure can
never break the release itself.

`./scripts/release.sh publish X.Y.Z` exists only as a fallback: it
re-dispatches the workflow after a transient failure, or with
`--local` runs the identical pipeline (`scripts/ci/release.sh`) on
the operator's machine when Actions is unavailable.

## Manual runbook

Every step the scripts automate, written out. Commands run from the
repository root on a machine with `git`, `gh` (authenticated),
`git-cliff` (pinned in `cliff.toml`), and the meson toolchain.

### 1. Prepare the release branch

```bash
git checkout main && git pull origin main
git status                  # must be clean
git tag -l vX.Y.Z           # must not exist
git checkout -b chore/release-X.Y.Z
# meson.build: version: 'X.Y.Z-dev' -> 'X.Y.Z' (single line)
git-cliff --unreleased --tag vX.Y.Z --prepend CHANGELOG.md
# First release only (no previous vN tag): skip git-cliff and write
# the "## [X.Y.Z] - <date>" heading under "# Changelog" by hand.
# Insert .release-notes-template.md (minus the <!-- --> comments)
# directly under the new "## [X.Y.Z]" heading and fill in every
# FILL-IN section; trim the generated Changes list down to the
# notable items. Check README.md and docs/ for stale version
# references while at it.
```

### 2. Verify locally

```bash
./scripts/ci/release.sh X.Y.Z verify
# = version/tag/changelog/docs guards + full build and test run
```

### 3. Release PR

```bash
git add meson.build CHANGELOG.md
git commit -m "chore: release X.Y.Z"
git push -u origin chore/release-X.Y.Z
gh pr create
```

Review covers the release notes and any docs fixes. **Merging this PR
publishes the release** — say so in the PR description.

### 4. What the workflow does after the merge

For an out-of-band release, perform the same steps by hand:

```bash
# verify: guards + build + tests on the release commit
./scripts/ci/release.sh X.Y.Z verify
# package: meson dist tarball + sha256 + notes into dist/ (meson dist
# also rebuilds and tests the unpacked tarball; needs verify's
# build directory)
./scripts/ci/release.sh X.Y.Z package
# publish: THE RELEASE — tag + GitHub release + notes, nothing else
./scripts/ci/release.sh X.Y.Z publish
# upload: attach dist/ artifacts (idempotent, re-run on failure)
./scripts/ci/release.sh X.Y.Z upload
# announce: discussion/wiki, best-effort
./scripts/ci/release.sh X.Y.Z announce
```

The publish step treats the tag as effectively irreversible: it is
the public release. Everything after it is retryable follow-up.

### 5. Open the next development cycle

```bash
git checkout main && git pull origin main
git checkout -b chore/open-X.Y.W-dev
# meson.build: version: 'X.Y.Z' -> 'X.Y.W-dev'
git add meson.build
git commit -m "chore: open X.Y.W-dev development"
git push -u origin chore/open-X.Y.W-dev
gh pr create
```

This also disarms the release workflow's gate (the `-dev` suffix).

### 6. Verify the published release

```bash
gh release download vX.Y.Z --dir dist/verify-X.Y.Z
(cd dist/verify-X.Y.Z && sha256sum -c meerkat-X.Y.Z.tar.gz.sha256)
tar -C dist/verify-X.Y.Z -xzf dist/verify-X.Y.Z/meerkat-X.Y.Z.tar.gz
meson setup build-verify dist/verify-X.Y.Z/meerkat-X.Y.Z
meson compile -C build-verify && meson test -C build-verify
# Note: the tarball has no .git, so -Dtools defaults to disabled —
# this deliberately exercises the extension-only build packagers get.
# Smoke test in a scratch instance:
#   CREATE EXTENSION meerkat;
#   SELECT mkt.extension_version(), mkt.git_commit();
```

## Release checklist

- [ ] CI green on `main` (all suites — `paths:` filters mean the
      release PR alone may not have triggered everything)
- [ ] On-disk format audit done (`MKT_META_MAGIC` version byte)
- [ ] Compatibility matrix row added; index rebuild stated if needed
- [ ] CHANGELOG.md entry complete (no `FILL-IN`), reviewed in the PR
- [ ] README.md / docs version references current
- [ ] Release PR merged (= approval) and workflow green
- [ ] Next `-dev` version opened on `main`
- [ ] Published artifacts verified (`release.sh verify`)

## Troubleshooting

### Stale SQL files in the PostgreSQL share directory

If tests or `CREATE EXTENSION` pick up an unexpected version, list
what is actually installed:

```bash
ls "$(pg_config --sharedir)/extension/meerkat"*
```

Remove leftover dev-version files (`meerkat--X.Y.Z-dev.sql`) that
should not be installable. Old version-named libraries
(`meerkat-<version>.so` in `pg_config --pkglibdir`) are harmless but
can be cleaned the same way.

### Extension won't upgrade

If `ALTER EXTENSION meerkat UPDATE` fails:

1. The target upgrade script exists in the share directory.
2. The control file's `default_version` is the intended target.
3. An upgrade path exists:
   `SELECT * FROM pg_extension_update_paths('meerkat');`

## Future work

- Binary artifacts per PostgreSQL version / OS / architecture (alphas
  ship a source tarball only).
- pg_dump-diff upgrade test harness (upgrade equals fresh install).
- Downstream propagation (e.g. Docker images) when applicable.
