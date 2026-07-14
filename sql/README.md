# Extension SQL

This directory holds the canonical extension SQL (`meerkat.sql`,
installed verbatim under the versioned name) and, beside it,
pgvector-style, the scripts that upgrade an installed meerkat
extension from one version to the next, named

```text
meerkat--<from>--<to>.sql
```

PostgreSQL chains adjacent-version scripts automatically, so only
adjacent pairs are needed (check the available paths with
`SELECT * FROM pg_extension_update_paths('meerkat');`).

## Rules

- Scripts are added to `ext_update_scripts` in `src/pg/meson.build` as
  they are created (listed explicitly; meson discourages globs).
- Once a version is released its scripts are immutable — never edit a
  released script; fixes go into the next version's script. Only
  `meerkat.sql` is a living file; everything else here is a frozen
  release artifact.
- Scripts reference the extension library as the literal
  `'MODULE_PATHNAME'`, which PostgreSQL resolves from the *target
  version's* control file (`meerkat--<version>.control`) when the
  script runs — upgrade scripts stay version-blind, pgvector-style.
- The extension shared library is version-named
  (`meerkat-<version>.so`) and every installed version binds its own
  library, so an upgrade script must repoint **every** C function to
  the new version's library: a version-agnostic block of
  `CREATE OR REPLACE FUNCTION ... AS 'MODULE_PATHNAME', '<symbol>'`
  statements derived from the canonical `sql/meerkat.sql` (a small
  generator to build when the first supported upgrade is owed),
  followed by the hand-written migration statements.
- While an upgrade path to the next release is being maintained,
  every catalog-affecting change during the development cycle must
  land in the same PR as its addition to the pending upgrade script.
  Whether a given version pair gets a path is a case-by-case release
  decision (prereleases carry no guarantee); each release's notes
  state it.

See `docs/release.md` for the full release process and upgrade policy.
