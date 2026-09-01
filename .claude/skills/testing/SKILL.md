---
name: testing
description: >
  Which of this repo's four test kinds to use for a change, where new
  tests belong, coverage expectations, and gotchas that produce
  false-positive passes. Trigger when writing tests for new or changed
  code, or judging whether coverage of a change is adequate.
---

# Testing

Tests exist to safeguard the correctness of the code — not the other way
around. There is no goal of "making the tests pass" for its own sake. If a
test fails, the default assumption is that the code is wrong; only change
the test itself when you've independently confirmed the test's expectation
was wrong, not merely inconvenient. Never edit, weaken, delete, or skip a
test in order to get it to pass without first understanding *why* it
failed — a green suite obtained that way is worse than a red one, because
it actively hides the problem instead of reporting it.

Four kinds of test exist here, each covering something the others can't:

- **Unit tests** (`test/unit/`) — link against the standalone static
  library, no PostgreSQL needed. Use the `TEST_GROUP`/`TEST`/
  `TEST_PARAMETERIZED` macros (see `docs/implementation.md` for the
  framework); `TEST_MEMCTX_FIXTURE()` catches leaks. Run one group:
  `./builddir/test/unit/run_tests --group <Name>`. Fastest feedback loop —
  prefer these for anything in code shared between standalone and PG.
- **Regression tests** (`test/pg/sql/` + `test/pg/expected/`) — SQL-level,
  single backend. For SQL-visible behavior: query results, plans, error
  messages.
- **Isolation tests** (`test/pg/specs/`) — concurrent multi-session
  scenarios (locking, MVCC, concurrent build/insert/mutate).
- **TAP tests** (`test/pg/t/`) — multi-instance scenarios, e.g. streaming
  replication.

Pick the cheapest kind that actually exercises the change — don't reach for
a TAP test where a unit test would do.

## Where new tests go

Don't proliferate small regression test files. Before adding one, look for
the existing file that already covers the concern (a broad file like
`inspect.sql` can absorb a related check rather than spawning a new file
for it) — a new file needs a distinct concern with room to grow, not a
single check. When appending to an existing file, minimize the diff rather
than reorganizing what's already there.

## Coverage

`./scripts/ci/coverage.sh` gates on a minimum line coverage percentage —
`MIN_COVERAGE` near the top of that script is the source of truth for the
number; it has moved before, so read it there rather than assuming a
figure. Its comment explains why it's below 100%: set for cross-platform
SIMD code, since foreign-architecture SIMD variants can't execute in CI
and are excluded, while same-architecture variants are exercised via
`mkt_simd_set_override()`. If coverage drops, look for genuinely untested
logic before assuming the threshold is miscalibrated.

## Gotcha: PG suites load the installed .so, not the build dir

`meson test -C builddir` for the `regress`/`isolation*` suites spins up a
temp instance that loads the extension from the **installed** PostgreSQL
prefix, not from `builddir` directly. New C changes are invisible to those
suites until `meson install -C builddir` runs first — unit tests link
directly against `builddir` and don't need this. A regress/isolation test
that "passes" against a stale `.so` is a false positive: if a test only
checks query *results*, a sequential scan can satisfy it just as well as
the code path you meant to exercise. When the point is exercising an index
or a specific plan, assert on the plan/EXPLAIN output, not just the result.

## Regenerating expected output (regress/isolation)

When a regress or isolation test's `.sql`/`.spec` changes — or a new test
is added — never hand-edit the matching `test/pg/expected/*.out` to match
what you *think* the output should be. It rarely matches byte-for-byte
(whitespace, notice ordering, plan formatting), so a hand-edited expected
file usually just produces a second, different failure.

Instead, regenerate it from what PostgreSQL actually produced:

1. Run the test so it fails: `meson test -C builddir <regress|isolation>`.
2. pg_regress/pg_isolation_regress write the actual output under the
   build dir (`--outputdir` in `test/pg/meson.build` — look under
   `results/` there) and a diff in `regression.diffs`.
3. **Read that diff before doing anything else.** Only accept the new
   output once you've confirmed the change is the one you intended — not
   just "any diff": copying over the expected file makes the test pass
   *by definition*, so this is the one place a genuinely wrong output can
   slip in and no test will ever catch it again.
4. Copy the actual output over the expected file, e.g.
   `cp builddir/test/pg/results/<test>.out test/pg/expected/<test>.out`.
5. Re-run the test to confirm it now passes, and check the resulting diff
   in `git diff` is what you'd expect from your change — nothing more.

## Writing tests that actually test something

- Cover the edge cases identified during planning, not just the happy
  path — a test that only exercises what the implementation already
  obviously does isn't adding coverage that matters.
- For anything touching both standalone and PG paths, prefer testing the
  shared logic once in standalone rather than duplicating coverage in
  regression tests.
- A test that would still pass if the logic under test were deleted or
  wrong isn't testing anything — assert on the specific behavior, not just
  "it didn't crash" or "the query returned rows."
- A feature isn't done when it compiles and passes the happy path; it's
  done when its edge cases and failure modes have tests.
