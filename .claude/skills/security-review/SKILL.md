---
name: security-review
description: >
  Adversarial security review of the extensions: the install script, the
  pgvector compat path, the event trigger, casts, and SQL-callable
  functions, and the C-code side. Carries the search_path /
  privilege-escalation threat model, a per-object checklist, pgspot
  interpretation, and the negative-control procedure that proves the
  security tests actually catch the vulnerabilities. The `review` skill
  hands off here for anything SQL-callable or install-time. Trigger when
  asked for a security review, changing the install-time SQL, or the
  compat/event-trigger/cast logic, or when asked to security-review the
  extension.
---

# Security review

The deep, adversarial pass for the extension. The general `review` skill
treats security as one concern among several and routes here; this one makes
the threat model explicit so the review is anchored, not vibes.

**Run it as a different agent than wrote the change.** Self-review of
security-sensitive SQL has correlated blind spots — the author who "hardened"
a path is the last to notice the hole they reasoned past. This isn't
hypothetical: a hardening change here once shipped with security tests that
had silently lost their teeth, and only an adversarial pass caught it.

Scope is mainly the SQL/extension layer, but also the C-side of any API
functions.

## Why install-time SQL is dangerous

"Install-time SQL" means whatever `CREATE EXTENSION` runs — the install
script and any update scripts — plus anything an event trigger fires later.
It runs as the invoking role, under *that role's* `search_path`, with the
extension schema on the path. An attacker can pre-create schemas that the
extension installs objects in and plant objects in it *before* an admin
installs. Two consequences drive most findings:

- **Any unqualified builtin** reached during install — or later, from a
  function running under a caller-controlled path — can be shadowed. An
  exact-arity `format(text,text,text)` beats `pg_catalog.format(text,
  VARIADIC "any")` from *any* path position; schema order doesn't save you.
  The planted body then runs with the installing or triggering role's rights.
- **pgvector is a *trusted* extension**, so a non-superuser can install it,
  own `public.vector` / `public.halfvec`, and plant a `WITH FUNCTION` cast
  whose function runs as whatever role later triggers the coercion.

Reachability is wider than it looks: the compat logic runs from the install
block *and* from an event trigger that fires on *any* later `CREATE
EXTENSION`. A NOSUPERUSER can arm the trap and wait for an admin's routine
`CREATE EXTENSION vector`. Demonstrated end-to-end here: NOSUPERUSER →
SUPERUSER.

## Checklist — walk every object, don't spot-check

Read the whole install script. For each item, cite the line and say pass or
finding.

1. **Every function pins its path.** `SET search_path = pg_catalog, pg_temp`
   (pg_temp last — it's never searched for functions or operators, and last
   stops a relation lookup resolving into an attacker's temp object). Covers
   the compat helper, the event-trigger function, and any new function. A
   missing pin plus unqualified names is the classic vector — pgspot
   **PS005**.

2. **Builtins are schema-qualified.** `pg_catalog.format`,
   `pg_catalog.quote_ident`, the selectivity support functions
   (`pg_catalog.scalarltsel`/`eqsel`/…), and
   `pg_catalog.pg_event_trigger_ddl_commands()`. Qualification is
   unshadowable and load-bearing even with a pin present — either layer
   alone blocks the hijack, so confirm *both*. pgspot flags unqualified
   builtins in dynamic SQL as **PS016**.

3. **Pre-existing objects are verified, not adopted.** The cast steps create
   only when absent and *reject* any pre-existing cast that isn't the
   expected binary one (`WITHOUT FUNCTION`, `castmethod = 'b'`) — RAISE,
   don't swallow. A blind `EXCEPTION WHEN duplicate_object THEN NULL` around
   `CREATE CAST` is a finding: it adopts an attacker's planted cast.
   (Swallowing an operator-family membership dup is fine — `ALTER OPERATOR
   FAMILY` is superuser-only, so no member can be pre-planted.)

4. **External objects are discovered, not hard-coded.** pgvector's schema
   comes from `pg_extension.extnamespace` at runtime, interpolated with `%I`
   / `pg_catalog.quote_ident` — never a literal `public`. Hard-coding
   `public` is both wrong (breaks when pgvector lives elsewhere) and a
   coupled security assumption. Confirm schema/identifier positions use
   `%I`, not `%s`.

5. **Own objects stay unqualified — that's correct.** the extension's own
   types and operators (`vec32`, `<->`, …) are referenced unqualified,
   relying on the forced install path; do **not** "fix" them with
   `@extschema@` prefix — it's noise, and `@extschema@` forecloses a
   relocatable extension. pgspot **PS017** on own objects is expected
   (pgvector trips 50 and can't remove them). Don't raise a finding on
   PS017; don't gate CI on it.

6. **User-callable functions and procedures have proper privilege checks.**
   All function's that a user can call, including those reached implicitly
   from, e.g., operators, must have proper privilege checks suitable for the
   function's use case. The C-side of such functions should typically have
   an ACL-check and raise proper errors. Be *extra* vary of functions
   defined with `SECURITY DEFINER`, as those might execute with superuser
   privileges or, at least, elevated privileges from the extension
   installer. Such security-definer functions might have a legit use case,
   but warrants extra scrutiny.

## Tooling: pgspot

Run pgspot on the install script (`pgspot <install-script.sql>`).

- **Blocking:** `Errors > 0`, any **PS005** (function without a pinned
  path), any **PS016** (unqualified builtin in dynamic SQL) — the
  exploitable classes.
- **Not blocking:** **PS017** (unqualified own-object reference) — the
  standard PostgreSQL idiom, unsilenceable without `@extschema@`. Note the
  count, move on.
- Don't pass the file to `--proc-without-search-path` — that reads empty
  stdin and reports a false "clean".

pgspot never reads the control file and has no notion of `relocatable`, so it
can't tell a mandatory-and-safe own-object reference from a risky one. That
judgement is yours; PS005/PS016/errors are where it's authoritative.

## Prove the tests have teeth (the step that's easy to skip)

A security test that never fails against the vulnerability is worse than none
— it manufactures false confidence. The suite passing is necessary but not
sufficient. For every defense the change touches, prove the matching check
*fails* when the vulnerability is reintroduced.

1. Run the security suite green first.
2. Negative-control each touched defense on the **installed** SQL — the
   artifact `CREATE EXTENSION` reads (`SHAREDIR/extension/<ext>--<ver>.sql`),
   not the source tree — then restore (reinstall, or from a backup):
   - Un-pin + bare `format`: delete the `SET search_path` line and turn
     `EXECUTE pg_catalog.format(` into `EXECUTE format(`. The format-hijack,
     install-time, and escalation checks must all FAIL.
   - Disable cast-tamper detection (`IF existing_method <> 'b'` → `IF
     false`): the tampered-cast check must FAIL.
   - Remove one pin: its pin-assert must FAIL.
   Fully un-hardened → most checks fail. The positive control (compat cast
   still created) should stay green; break the setup path once to confirm it
   *can* fail, so it isn't guarding a vacuous pass.
3. Watch the teeth-loss traps this suite has actually hit:
   - **Arity coupling.** A plant matching one `format()` arity stops
     matching when a call's arg count changes (this shipped once). The
     plants cover a range; if you add or change a `format()` call, keep its
     arity in range.
   - **Subtransaction rollback.** A plant that *records* a row is undone by
     an `EXCEPTION WHEN duplicate_object` handler rolling it back. Detection
     there must RAISE and propagate, not record.
   - **NULL conditions.** `proconfig @> ARRAY[...]` is NULL when the pin is
     absent, and an assert on NULL is neither a pass nor caught by a `WHERE
     NOT passed` exit check. Pin assertions must `COALESCE(..., false)`.

A check that stays green under its matching negative control is a finding, no
matter how convincing it reads.

## Reporting

Report with `ReportFindings` when available, else to the user, most severe
first. For each: file:line, the concrete attack or failure it enables, and —
for a test finding — the negative control that should have failed but didn't.
Treat a toothless check as severity-equal to a missing pin. State per defense
touched: **teeth verified — yes / no**. Flag deliberate trade-offs as such;
don't silently dismiss them.

## Limits

This covers the *known* classes above. A novel attack shape the threat model
doesn't already model still needs a human reviewer — this skill makes the
routine checks reliable so attention is free for the rest. And independence
only holds if a *different* agent, or a later session, runs it.
