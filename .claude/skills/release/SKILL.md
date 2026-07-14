---
name: release
description: >
  Drive a meerkat release: prepare the release branch and notes, walk
  the CI-published release, open the next dev cycle, and verify the
  artifacts. Use when the user asks to release, tag, or publish a
  meerkat version.
---

# Meerkat release

`docs/release.md` is the authority on the process (read it first);
`scripts/release.sh` is the mechanism. Your job is to orchestrate the
phases and to author the normally human-written content — with the
release PR as the human review gate.

## Safety rails (non-negotiable)

- Never dispatch the release workflow, create a tag, or create a
  GitHub release without the user's explicit confirmation.
- Merging the release PR **is** the publish approval. Make sure the
  user knows: say "merging this PR publishes the release" when
  handing over the PR, and never merge it yourself.
- Stop at PR boundaries: the user reviews and merges. Expect review
  feedback and incorporate it in follow-up commits on the release
  branch. Never self-approve.
- Use `release.sh publish --local` only when the user explicitly
  asks for the no-Actions fallback path.

## Phases

1. **Preflight.** Confirm with the user: which version (e.g.
   `0.1.0-alpha1`)? Prerelease suffixes: `-alphaN`/`-betaN`/`-rcN`.
   Check `git tag -l`, CI status on `main`, and the release checklist
   in `docs/release.md` (on-disk format audit, compatibility matrix).
2. **Prepare.** Run `./scripts/release.sh prepare <version>`. This
   bumps `meson.build`, prepends the CHANGELOG entry (generated
   Changes list), and inserts the release-notes template.
3. **Author the notes.** Fill in the template's FILL-IN sections in
   `CHANGELOG.md` yourself: read the generated commit list and the
   diffs since the last tag, then write user-facing prose (what can
   users do now, what breaks, what to know) — not a commit recap.
   The breaking-changes section must state whether an upgrade path
   from the previous release exists — for prereleases that is a
   case-by-case decision (see docs/release.md). Fix any stale
   version references the docs-freshness check flags (README.md,
   docs/).
4. **Finish the branch.** Re-run
   `./scripts/release.sh prepare <version>` (runs the verify step:
   guards + full build + tests, then commits). Push and open the PR;
   its description must state that merging publishes the release.
5. **Hand over.** The user reviews, requests revisions (make them on
   the branch), and merges. The release workflow then runs
   verify → release → artifacts → announce. Watch it with
   `gh run watch` and report the outcome. If a follow-up job fails,
   re-run it (`artifacts` is idempotent); if the workflow needs a
   re-dispatch, that is `./scripts/release.sh publish <version>` —
   confirm with the user first.
6. **Post-release.** `./scripts/release.sh post <next>-dev`, push,
   PR (user merges). Then `./scripts/release.sh verify <version>`
   and report the artifact check results.
