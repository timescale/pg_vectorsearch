---
name: git-workflow
description: >
  This repo's general git conventions — starting a branch, committing as
  you go, hooks, addressing review feedback, and cleaning up after merge.
  Trigger when the user wants to start new work, begin a feature/bugfix/
  refactor, or clean up after a PR merges.
---

# Git workflow

Companion to `create-pr`, which covers the commit-and-PR mechanics specific
to opening a PR (staging, the final commit history shape, rebasing onto the
default branch, the PR body itself), and to `build-feature`, which
orchestrates a full plan/implement/review pipeline around a branch. This
skill covers the branching and committing that happens around both: before
it, during ongoing work, and after merge.

## 1. Start a branch

Never commit directly to the default branch. Always make sure it's
up-to-date first:

```bash
git checkout main
git pull origin main
git checkout -b <type>/<name>
```

Branch naming:

- `feature/<name>` — new functionality
- `bugfix/<name>` — bug fixes
- `test/<name>` — test-only changes
- `refactor/<name>` — refactoring

## 2. Work in iterations

- Commit often, at whatever granularity actually helps — before a risky
  change, once something works, before trying an alternative. This
  in-progress history doesn't need to be clean or final; that bar only
  applies to what the PR is opened from (see `create-pr`).
- It's fine to stage a complex sub-part on a throwaway helper branch and
  merge or cherry-pick it back once it works, rather than getting it right
  in one pass directly on the feature branch.
- Still keep commits logical moment-to-moment — one concern per commit,
  don't bundle unrelated changes — but expect to rewrite this history
  before opening the PR rather than needing to get it right the first
  time. Don't worry yet about whether every intermediate commit builds or
  passes tests on its own — that requirement kicks in for the *final*
  history (see `create-pr`'s "Final commit history shape", which exists
  so the branch stays `git bisect`-able).

### Never skip hooks with `--no-verify`

This repo's pre-commit hooks (`.pre-commit-config.yaml`) run clang-format,
clang-tidy, shellcheck, and commit-message linting — real checks, not
formalities. Committing with `--no-verify` because a hook is slow, in the
way, or failing for a reason that isn't obvious does not make the problem
go away; it just moves the failure to CI, or ships a formatting issue that
someone else has to notice and fix later. If a hook fails, fix what it's
flagging (or fix the hook itself if it's genuinely wrong) rather than
bypassing it — the same rule as `create-pr` step 2 for the manually-run
checks.

### Commit messages

Follow [How to write a git commit message](https://chris.beams.io/git-commit).

The commit-msg hook (`.commitlintrc.yaml`) enforces Conventional Commits on
every commit, not just the one(s) a PR is eventually opened from:

```
<type>(<scope>): <description>

<body — what changed and why, key trade-offs, caveats>
```

- Types: `feat fix docs style refactor perf test build ci chore revert`.
- Header lowercase, under 72 characters, blank line before the body.
- Body explains *why*, not a list of files touched — the diff already
  shows *what* changed.
- No `Co-Authored-By` lines, no "Generated with Claude Code" or similar
  footer.
- Keep the body clean and factual: describe what changed and why, not a
  narration of the work session.
- No benchmark numbers tied to this specific machine or environment — they
  won't reproduce elsewhere and go stale. Describe the general performance
  characteristic instead, e.g. "cuts index build time in half" or "no
  measurable regression" rather than a specific number.
- Don't name local-only tooling (e.g. local build/benchmark helpers) in
  the message.

The hook catches format violations (type, casing, header length, blank
line before the body) — fix those yourself rather than relying on it to
catch what it can't check, like whether the body actually explains *why*.

See `create-pr`'s "Final commit history shape" for how the commits on a
branch should be organized before a PR is opened from it.

## 3. Address review feedback

- Push new commits for requested changes rather than rewriting history,
  unless the reviewer specifically asks for a squash or rebase.
- Push updates to the same branch: `git push origin <branch>`.

## 4. After merge

```bash
git checkout main
git pull origin main
git branch -d <branch>
```

Only delete the local branch once the PR is actually merged and the default
branch has pulled that merge in.
