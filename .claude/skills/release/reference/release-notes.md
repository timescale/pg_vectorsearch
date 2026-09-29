# Writing the release notes

The brief for section 3 of SKILL.md. Read it before writing, not
after. `reference/thanks.md` covers the Thanks section.

Edit the new `CHANGELOG.md` entry. `git-cliff` already wrote
`### Changes` from the commit history. The sections above it still
contain `FILL-IN`. The release check fails while any remain.

Write for users. Short. Say what they can do now, or what they must
do to upgrade. Not a commit recap, and not a tour of the branch.

The reader runs PostgreSQL and is deciding whether to upgrade. What
they want, in order: can they do something they could not before,
must they do anything to upgrade, and is anything they relied on
gone. Everything else is detail.

Open with one line saying what kind of release this is (a release
introducing new features, or a bug fix release), which release it
follows, and whether to upgrade. "This release contains bug fixes
since 0.1.0. We recommend upgrading at the next opportunity." A reader
who stops after that line should still know whether it concerns them.

The commit list is not where that lives. What a user can *do* is the
SQL surface — the functions and procedures in `sql/pg_vectorsearch.sql`,
the index options in `PrismOptions`, the `prism.*` GUCs registered in
`_PG_init`, and the PostgreSQL versions `meson.build` accepts. A change
that adds or alters one of those is a highlight. A change that does
not is a `Changes` bullet, or nothing.

Budget: three to six sentences of highlights, or up to four bullets.
At most ten bullets under `Changes`. Needing more means the release is
unusually large or the list is not cut down yet.

Numbers need a source. Do not carry a performance figure out of a
commit message unless `docs/` or that commit says how it was measured.
"Faster index builds" is honest. "38% faster" is a claim this file
cannot support. Where a pull request holds the benchmark, link it and
let the reader look. That is stronger than a figure in prose and it
cannot go stale.

A highlight names the mechanism and then what changes for the user. "A
tree descent replaces the linear centroid scan, so probing cost grows
with the log of the cluster count rather than with the count" does
both. Naming only the mechanism leaves the reader to work out whether
they care.

If `CHANGELOG.md` already has an entry from a previous release, match
its shape and length rather than inventing a new format.

- **Highlights.** A few sentences, or a short bullet list if there
  are several things. One release usually has one point. Skip
  refactors, CI, and naming.
- **Breaking changes.** `None.` if there are none. If
  `prepare-release.sh` inserted an "Indexes must be rebuilt" bullet,
  keep it. Do not soften it.
- **Deprecations, known issues.** Delete the whole section when it is
  empty. Do not leave a placeholder, and do not write "None." in a
  section the template says to delete.
- **Thanks.** See below. Delete it when nobody outside the team
  contributed, which is the usual case.
- **Changes.** Raw material, not the deliverable. Cut it to the
  notable items and leave the compare link if there is one. Dropping
  a bullet does not drop the change. The history is still the
  history.

  Describe a fix by the **symptom**, not the repair, so a reader can
  recognise whether they hit it. "Wrong results from a scan when a
  posting list spans more than one page" tells them something. "Fixed
  posting page iteration" does not.

- **New settings.** A release adding an index option or a `prism.*`
  GUC gives it a line of its own: the name, what it does in one
  sentence, and its default. That is the part a reader acts on.

## A long list

A release with many commits — the first one especially, where the
list is the entire history — is a selection problem, not a
summarising one. Reproducing it is the failure mode.

Read the list, and the diff behind anything you cannot judge from its
subject. Then group by what a user would notice, because one
capability is usually many commits: a feature, the commits that fixed
it during development, and its tests are one line in the notes, not
four. Keep the commits that a highlight rests on. Drop refactors, CI,
formatting, renames, reverted work, and anything whose subject only
makes sense to someone who was reviewing at the time.

Then point at the rest instead of listing it:

```markdown
Every change is in the commit history:
<https://github.com/timescale/pg_vectorsearch/commits/v0.1.0>
```

For a later release the compare link git-cliff leaves does that job
already. Say where the full list is once, and do not apologise for
the summary being a summary.

A first release has no "what changed". Say what the extension is and
what it does, in the same few sentences. The whole history is the
material and none of it is news.

## Voice

Someone will read this who is deciding whether to upgrade a database.
Write the way a maintainer writes to a user, not the way a model
writes to be thorough.

Short declarative sentences. One idea each. A full stop where a
semicolon is tempting — if two clauses are worth joining, they are
usually worth separating. Name the thing rather than describing its
benefit: "`target_pages` sizes a posting list in pages" beats
"improved control over list sizing".

The tells to stay away from, all of which read as machine-written:

- a colon introducing a list mid-sentence, as in "brings: X, Y and Z"
- semicolons joining independent clauses
- em dashes used for rhythm rather than punctuation
- "not just X, but Y", and three-item lists where two would do
- *seamlessly*, *robust*, *powerful*, *leverage*, *delivers*,
  *enables*, *comprehensive*, *significantly*
- opening with *Additionally*, *Furthermore*, *Moreover*
- "it's worth noting that", "it's important to note that"
- a closing sentence that restates the paragraph above it
- bullets that all begin with the same gerund

No "this release introduces". No "we are excited". Do not narrate how
the release was cut. If a sentence could be shorter, shorten it.

## What that looks like

For a release that added a reloption and fixed a scan bug:

```markdown
## Highlights

Posting lists can be sized with the `target_pages` index option, so a
list rests at a predictable number of pages rather than a count
derived from the table's row estimate.

Scans no longer miss rows when a posting list spans more than one
page.
```

Not this:

```markdown
## Highlights

This release introduces exciting improvements to posting list
management, including a new `target_pages` reloption that delivers
enhanced control over list sizing, along with various bug fixes and
performance improvements.
```

The second one is longer, names nothing a user can act on, and its
last clause could belong to any release of any project.
