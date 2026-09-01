---
name: create-pr
description: >
  Prepare and open a pull request end to end — pre-commit checks, a
  Conventional Commits commit, rebase, push, and a `gh pr create` body
  matching this repo's style. Trigger when the user asks to commit and open
  a PR, ship a branch, or "create a PR".
---

# Creating a pull request

Companion to the `git-workflow` skill, which covers starting a branch and
cleaning up after merge. This skill is the checklist for everything from
pre-commit checks through the opened PR, plus the PR body style actually
used in this repo's merged PRs (see #223, #219 for reference).

Treat committing, pushing, and opening the PR as three separate confirmation
points — do not chain them into one silent sequence. Pause and confirm with
the user before each of the three, even if they asked to "create a PR" in one
go.

## 1. Check branch state

```bash
git status
git branch --show-current
```

If on the default branch, start a feature branch first — see the
`git-workflow` skill. Never commit directly to the default branch.

## 2. Run pre-commit checks

All of these must pass before staging anything:

```bash
meson compile -C builddir format
meson test -C builddir
./scripts/ci/coverage.sh    # minimum coverage — see MIN_COVERAGE in the script
./scripts/ci/lint.sh        # must be clean
```

If coverage or lint fails, fix it before proceeding — don't open the PR with
known-failing checks hoping CI catches it later.

## 3. Stage and commit

- Stage files explicitly by name. Never `git add -A` or `git commit -a`.
- Split unrelated changes into separate commits.
- Review with `git diff --staged` before committing.
- **Confirm with the user before running `git commit`.**

### Final commit history shape

Whatever the branch's history looked like during development (see
`git-workflow` — lots of small WIP commits, checkpoints, even a helper
branch or two, is fine and encouraged there), the history a PR is opened
from should be either a single commit, or a small number of commits that
each stand on their own.

Each commit should already be the final code it's introducing, not an
intermediate state that a later commit reworks. Concretely: avoid a commit
that adds function X followed later by a commit that refactors X — squash
those into one commit that adds the final X directly. A later commit doing
a small amount of refactoring *in order to build on top of* something
earlier (e.g. widening a function's parameters to fit new functionality)
is fine — that's forward progress, not undoing earlier work. The test is
directional: does each commit move the code toward its final state, or
does a later one take back something an earlier one did? The latter should
be squashed away with `git rebase -i` before proceeding, not left for the
reviewer to reconstruct from the diff.

Each commit in that final history should also build and pass the test
suite on its own — not just the branch's tip. A commit that introduces a
failure "temporarily," fixed by a later commit, keeps the code from being
bisectable: `git bisect` (and anyone reading history later) needs every
commit to be in a working state to be useful. If splitting the work into
several commits would leave an intermediate one broken, that's a sign
those commits should be squashed together instead. Run the checks from
step 2 against each commit, not only the last one, when in doubt — e.g.
`git rebase -i` with `exec meson test -C builddir` between commits.

Commit message format (Conventional Commits, no Co-Authored-By, no
machine-specific benchmark numbers, etc.) is covered in `git-workflow`'s
"Commit messages" section — it applies to these commits the same as any
other.

## 4. Rebase and push

```bash
git fetch origin
git rebase origin/main
```

Resolve any conflicts, then re-run the checks in step 2 if source files
changed during the rebase.

**Confirm with the user before pushing.**

```bash
git push -u origin <branch>                       # first push
git push --force-with-lease origin <branch>        # after amending/rebasing
```

## 5. Open the PR

**Confirm with the user before running `gh pr create`.**

Title mirrors the commit header style: `<type>(<scope>): <description>`.

Body style (see PR #223 and #219 for full examples) — write prose, not a
templated checklist, and keep it short: a reviewer should get the point
from the first sentence or two, not have to read to the bottom to find
out what the PR actually does.

- Lead with a snappy, high-level summary of what changed and why it
  matters — one or two sentences, before any detail. Someone skimming
  just the top of the description should already know what this PR is.
- Stay high-level throughout. This is a summary for a reviewer, not a
  transcript of the work — the diff already carries the detail; don't
  re-explain it line by line, and don't include every alternative you
  considered or every step you took to get there.
- Only add a subsection per distinct change when there genuinely are
  several unrelated changes (e.g. "The two real bugs", "One hardening") —
  organize by the reader's understanding, not by commit order. A PR with
  one coherent change doesn't need subsections at all.
- If there's a measurable before/after (a plan change, a recall number,
  pass vs fail) worth calling out, a single small table is enough — but
  never a raw QPS/latency number tied to this machine; relative or
  qualitative results only.
- End with a **Checks** section listing what was actually run and its
  result (e.g. `meson test — 33/33`, `scripts/ci/lint.sh — all hooks
  pass`), not a markdown TODO checklist.
- No "Generated with Claude Code" footer, no Co-Authored-By, no local-only
  tool names.
- If the description is getting long, that's a signal to cut detail, not
  to add more structure to organize it.

```bash
gh pr create --title "<type>(<scope>): <description>" --body "$(cat <<'EOF'
<body>
EOF
)"
```

Report the PR URL back to the user when done.
