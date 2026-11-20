# Extension SQL

This directory holds the canonical extension SQL (`pg_vectorsearch.sql`,
installed verbatim under the versioned name) and, beside it,
pgvector-style, the scripts that upgrade an installed pg_vectorsearch
extension from one version to the next, named

```text
pg_vectorsearch--<from>--<to>.sql
```

PostgreSQL chains adjacent-version scripts automatically, so only
adjacent pairs are needed (check the available paths with
`SELECT * FROM pg_extension_update_paths('pg_vectorsearch');`).

## Rules

- Scripts are added to `ext_update_scripts` in `src/pg/meson.build` as
  they are created (listed explicitly; meson discourages globs).
- Once a version is released its scripts are immutable — never edit a
  released script; fixes go into the next version's script. Only
  `pg_vectorsearch.sql` is a living file; everything else here is a
  frozen release artifact.
- Scripts reference the extension library as the literal
  `'MODULE_PATHNAME'`, which PostgreSQL resolves from
  `pg_vectorsearch.control` when the script runs, pgvector-style.
- One library, `pg_vectorsearch.so`, serves every installed version, so
  an upgrade script holds only the statements for what changed in the
  SQL. In return the library keeps every C symbol that a script in this
  directory names, and a function whose signature changes gets a new
  symbol.
- 0.1.0 bound its functions to `pg_vectorsearch-0.1.0.so`. The script
  that upgrades from 0.1.0 rebinds every C function once, with
  `CREATE OR REPLACE FUNCTION ... AS 'MODULE_PATHNAME', '<symbol>'`.
- Every catalog-affecting change during a development cycle must land
  in the same PR as its addition to the pending upgrade script (from
  the first non-alpha release on; alphas have no upgrade path).

See `docs/release.md` for the full release process and upgrade policy.
