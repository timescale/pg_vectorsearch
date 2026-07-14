-- Build-identity functions: extension_name(), extension_version(),
-- and git_commit() report what the loaded library was built as.

SELECT extension_name();

-- The binary's version must match the installed extension's version;
-- a mismatch means the library and the SQL scripts come from
-- different builds (e.g. a stale install).
SELECT extension_version() =
    (SELECT extversion FROM pg_extension WHERE extname = extension_name())
    AS version_matches_extension;

-- A commit hash, or 'unknown' when built outside a git checkout
-- (e.g. from a release tarball).
SELECT git_commit() ~ '^([0-9a-f]{40}|unknown)$' AS git_commit_shape;
