---
name: review
description: >
  This repo's review checklist — reviewing a plan before implementation,
  and adversarially reviewing a finished branch's diff before opening a
  PR. Used by `build-feature`. Trigger when asked to review a plan, or to
  adversarially review a branch/diff for correctness, security,
  performance, cleanliness, or documentation drift.
---

# Review

Two kinds of artifact get reviewed in this repo's feature workflow (see
`build-feature`): a plan, before any code is written, and a branch's diff,
once the feature is finished or nearly so. The same priorities apply to
both, aimed at different material — a plan describes an intended design,
a diff is the actual result.

Both should be reviewed by a fresh agent — not the one that produced the
artifact being reviewed — so the review isn't reasoning from the same
assumptions the author had.

## Priority order

Weigh findings in this order, whether reviewing a plan or a diff:

1. **Performance** — especially anything on the query/search path.
2. **Security** — privilege boundaries, input handling, anything
   SQL-callable.
3. **Simplicity, architecture, and clean code** — once the above two are
   satisfied. This includes avoiding duplication: check whether existing
   code already does the same thing — directly callable, or after a
   small refactor — before accepting new logic that reimplements it. This
   is a good use of a fresh reviewer, since the author may not have
   looked as hard for something to reuse as someone starting cold.

## Reviewing a plan

Check the plan against the priority order above, plus general soundness:

- **Performance implications**, especially on the query/search path —
  does the design introduce avoidable overhead (extra allocations, copies,
  locks, weakened pruning) that a different approach would not have?
- **Security implications** — privilege checks, input handling, anything
  SQL-callable that the plan introduces or touches.
- **Simplicity and architecture** — is this the simplest design that
  still satisfies the two priorities above? Check it against the actual
  code and CLAUDE.md, not just the plan's own description of the existing
  patterns. Flag a simpler alternative the plan passed over without
  justification, and specifically flag duplication — new logic proposed
  for something existing code already does, or could do after a small
  refactor.
- Missing edge cases or unstated assumptions.
- Steps underspecified enough to invite improvisation mid-implementation.

Bring findings back to the user; revise and re-review if there are
material concerns before implementing.

## Reviewing code (adversarial)

Review the diff against the plan and this codebase's conventions, actively
trying to find something wrong rather than confirming it looks fine.

### Correctness and tests

- Correctness bugs, not just style nits.
- Gaps between what the plan promised and what actually got built.
- Missing test coverage: edge cases without a test, or the wrong kind of
  test for what's being checked (see the `testing` skill).
- **Tests shaped to pass regardless of whether the code is actually
  correct.** This is the specific failure mode of an implementer (agent or
  not) writing the test *after* the code and unconsciously fitting the
  test to whatever the code already does, rather than to what it should
  do. For every test touching the new logic, check whether the expected
  value was derived independently — from the plan, the spec, or a
  known-correct calculation — or whether it was just copied from whatever
  the implementation happened to output. The latter proves nothing.
  Concretely: read the assertion and ask whether it would still pass if
  the specific bug the plan was written to avoid were reintroduced; if it
  would, the test isn't testing that bug. Also watch for the softer
  versions of this — overly loose tolerances, swallowed exceptions/errors,
  assertions weakened until they pass, or a test disabled/skipped rather
  than fixed.
- Anything that would fail the checks in `create-pr` (formatting, the
  coverage gate in `scripts/ci/coverage.sh`, lint) before those are even
  run.

### Code cleanliness and consistency

A review focused on correctness tends to skim past this, so call it out
explicitly as its own concern:

- **Duplication.** Is any of the new code a near-copy of something that
  already exists in `src/`? Prefer pointing at the existing code to reuse
  (or extract into a shared helper) over new code that happens to do the
  same thing slightly differently. This includes standalone/PG
  duplication — logic that belongs in the shared layer shouldn't be
  reimplemented separately for each.
- **Naming consistency.** Do new function, type, and variable names match
  this codebase's existing conventions — checked against neighboring code,
  not just consistency within the new diff? E.g. the `mkt_` prefix is for
  C symbols only; SQL-visible names are namespaced by the `mkt` schema
  and should not repeat that prefix.
- **Public API surface especially.** New SQL functions, views, or GUCs
  deserve more scrutiny than internal code — their names, argument order,
  and return shapes should read as if they belong next to the existing
  ones in `sql/meerkat.sql`, not as a one-off with its own conventions.
  Once shipped, renaming a public function is a breaking change for
  whoever depends on it, so an inconsistent name is far cheaper to catch
  now than after a release.

### Documentation consistency

Check whether this repo's docs still match what the branch actually
implements — not just whether the feature's own new docs were written,
but whether any *existing* doc describing the touched area is now stale.
CLAUDE.md's own "Documentation Maintenance" section names where to look
(`README.md`, `docs/architecture.md`, `docs/implementation.md`), but the
bar is accuracy, not presence: a stale description is worse than no
description, since it actively misleads the next reader instead of just
leaving a gap. Flag any doc whose text still describes the old behavior,
a changed default, a removed parameter, or a number (like a threshold or
limit) that the code no longer matches — not only missing docs for what's
new.

### Security

If the feature touches anything SQL-callable, privilege boundaries, or
parses input that ultimately comes from a client, run a dedicated
security-focused pass in addition to the general review above — a
reviewer told to look for correctness bugs tends not to also be thinking
about privilege escalation. Check specifically for:

- SQL-callable functions with no privilege check that should have one —
  should this function require a specific role, or be reachable by any
  user with `USAGE` on the schema when it shouldn't be?
- `SECURITY DEFINER` functions without a pinned `search_path` — the
  classic PostgreSQL privilege-escalation vector, since an unqualified
  identifier inside can be hijacked by a caller-controlled search_path.
- Dynamic SQL (`format()`, string concatenation) that embeds identifiers
  or values from input without `quote_ident`/`quote_literal` — SQL
  injection.
- C code reachable from a SQL-callable function that trusts a
  length/count/offset supplied via input without validating it before use
  in `memcpy`/`palloc`/pointer arithmetic — a memory-safety bug is a
  security bug when any SQL-privileged user, not just an admin tool, can
  reach it.
- New functions or views that leak information a non-privileged user
  shouldn't see (internal file paths, raw pointers/addresses, other
  tenants' data).
- Whether the new SQL surface actually matches how equivalent existing
  functions in `sql/meerkat.sql` restrict access — a good check is
  grepping for how a similar existing function handles this rather than
  inventing a new pattern.

### Performance

If the feature touches the query/search path, review with performance in
mind as well as correctness — ideally no regression, and better, an
improvement over the current baseline. Flag anything that would be
expected to add overhead on a hot path: new allocations per query or per
candidate, copies that weren't there before, removed or bypassed
SIMD/fast-path code, a new lock or syscall on a path that didn't have one,
or pruning weakened in a way that lets more candidates through to the
expensive rerank step.

Code inspection alone can miss a regression a benchmark would catch
immediately. When the plan claimed a specific performance goal (or the
change is anywhere near the hot query path), get an actual before/after
benchmark run — release build, per this repo's profiling docs
(`scripts/profile.sh`, `scripts/profile-bench.sh`) — rather than accepting
"looks fine" from a read-through alone.

### Reporting findings

Report findings to the user. Treat weak/missing/self-serving test
coverage, security, and performance findings at least as seriously as a
correctness bug — fix what's confirmed real rather than deferring it to a
follow-up, and don't paper over a finding just to get to the PR. If
something is a deliberate trade-off rather than a bug, say so instead of
silently dismissing it.
