# The Thanks section

Who to credit in a release, and how to find them. Referenced from
`reference/release-notes.md`.

A fix credits whoever made it possible, and that is often not the
person who wrote the patch. It is mostly people who reported a bug.

Look in three places, over the commit range git-cliff summarised:

1. **Commit trailers** — `Co-authored-by`, `Reported-by`,
   `Suggested-by`, `Tested-by`.
2. **Authors of the pull requests** merged in the range.
3. **Issues those pull requests closed.** The issue's author reported
   the bug, and nothing in the commit log says so:

   ```bash
   gh pr view <number> --json closingIssuesReferences \
     --jq '.closingIssuesReferences[] | "\(.number) \(.author.login)"'
   ```

Then decide who is external, because thanking the team reads as
filler. The issue or pull request records it:

```bash
gh api "repos/$REPO/issues/<number>" --jq .author_association
```

`CONTRIBUTOR` or `NONE` is an outside contributor. `OWNER`, `MEMBER`
and `COLLABORATOR` are the team, so leave them out.

Use that rather than org membership or the collaborators endpoint.
Membership needs a token scope this job does not have, and asking for
a collaborator's permission needs repository administration, which the
release App deliberately lacks. Both would fail in a way that reads as
"not a member" and would thank the whole team.

Rules that matter more than they look:

- **Never a bot.** Anything ending `[bot]`, and `Claude`,
  `dependabot`, `renovate`, `github-actions`. Every
  `Co-authored-by` trailer in this repository so far is Claude.
  Thanking a model in release notes is embarrassing.
- **Handles, never email addresses.** The git log has addresses in
  it. The notes must not.
- **One line per person**, even if they reported a bug and then fixed
  it.
- **Say what they did, and link the issue rather than the fix.** One
  line each, the problem in plain words:

  ```markdown
  **Thanks**
  * @user for reporting that a scan returned duplicate rows once a
    posting list spanned more than one page
    [#287](https://github.com/timescale/pg_vectorsearch/issues/287)
  ```

  The description is the problem in plain words, not the patch. Where
  someone did more than report, say that instead — "for reporting the
  issue and providing steps for reproduction".
- **If you cannot classify someone, leave them out of the notes and
  name them in the pull request body** so the reviewer decides. A
  wrong attribution is worse than a missing one.
