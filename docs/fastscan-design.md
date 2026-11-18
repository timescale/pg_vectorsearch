# Fastscan posting pages

Fastscan scores 32 RaBitQ vectors at once with in-register table lookups on
their 1-bit codes, instead of one masked float sum per vector. It is the
posting list format that searches fastest, and the default at every dimension
the index supports, up to 8192.

The fastscan approach follows the RaBitQ Library (Gao & Long) and FAISS
FastScan (André et al., "Cache locality is not enough"). For PostgreSQL, this
approach has been adapted to a paged format.

Before 0.2.0 a fastscan group had to fit whole on one page, next to the
reference vector on a list's head page. That kept fastscan off above 944
dimensions and left the gap after a page's last whole group empty, up to almost
half a page. With 0.2.0, a list's codes form one stream that continues from
page to page, so each page wastes under 64 bytes and a group may be larger than
a page. Past 1968 dimensions, where the reference vector spills onto its own
pages, the stream starts on the last of them, right after the reference.

## Terms

- **Posting list**: the vectors assigned to one cluster, stored as a chain of
  8 KB pages with 8144 bytes for data each.
- **Reference vector**: the cluster's centroid (`2 × dims` bytes as
  float16, the default, or `4 × dims` as float32). Search uses it to
  compute `query − centroid` before scoring any vector in the list.
- **Head**: the pages of a list that carry its reference vector, each
  marked as the head's **START**, **MIDDLE** or **END** page. Up to 1968
  dimensions the head is the list's first page alone. Past that the
  reference spills: the START and MIDDLE pages are filled with it, 2036
  floats each, and the END page holds the rest followed by the list's first
  entries. That is the float32 layout (`posting_reference_format =
  float32`). The default float16 reference spills only past 3820
  dimensions, 4072 halves a page.
- **Head page**: a list's first page, its head's START page.
- **AoS entry**: one vector in the AoS format. A 20-byte header (TID, flags,
  three correction factors) followed by one bit per dimension.
- **Fastscan group**: 32 vectors in the fastscan format. The 1-bit codes of
  all 32 are interleaved so one SIMD lookup scores 32 vectors at once.
- **Subquantizer**: 4 dimensions. Their 4 sign bits pick one of 16 entries
  in the query's lookup table.
- **Column**: 32 bytes of a group's codes, two subquantizers (8 dimensions)
  of all 32 vectors.
- **Chunk**: 64 bytes of a group's codes, covering 16 dimensions of all
  32 vectors.
- **Group metadata**: a group's 32 TIDs, 32 × 3 correction factors, and its
  deleted-vector mask (580 bytes).

## Scoring

### Code packing

The build packs the 1-bit RaBitQ codes of 32 vectors column by column. For
each column and vector, the 4 sign bits of the first subquantizer go into
the low nibble of a byte and those of the second into the high nibble. The
vectors are interleaved within the column (`kPerm0` in `fastscan.c`) so the
kernels can add even and odd bytes separately. A group's codes take
`ceil(dims / 8) × 32` bytes, 3072 at 768 dimensions.

```
 one 64-byte chunk = 16 dimensions of all 32 vectors
 ┌────────────────┬────────────────┬────────────────┬────────────────┐
 │ dims 0–3       │ dims 4–7       │ dims 8–11      │ dims 12–15     │
 │ 32 × 4-bit code│ 32 × 4-bit code│ 32 × 4-bit code│ 32 × 4-bit code│
 │ 16 B           │ 16 B           │ 16 B           │ 16 B           │
 └────────────────┴────────────────┴────────────────┴────────────────┘
```

### Lookup tables

Once per list, the scan builds a lookup table from the transformed query.
Each subquantizer gets 16 entries, the partial inner product of its 4 query
dimensions for each sign combination, quantized with one linear scale and
bias. After the kernel sums a vector's entries, the inner product comes back
as `accum × delta + bias`.

`prism.fastscan_bits` picks the table's precision:

- **16 bits** (the default): entries are split into low and high byte
  tables, laid out in 128-byte blocks of four subquantizers, and the kernel
  makes two lookups per code byte.
- **8 bits**: one lookup per code byte, faster but coarser.

### Kernels

Per column the kernel splits the code bytes into nibbles, looks up 32
entries for each with VPSHUFB (TBL on NEON), and adds them into 16-bit
lanes. There are AVX2, AVX-512, NEON and scalar versions, picked at
runtime. AVX2 does one column per step with 256-bit lookups, and AVX-512
does both lookups of a column in one 512-bit instruction. Every kernel steps
through the codes in 16-, 32- or 64-byte units, which is what lets a group
break at any chunk.

The 16-bit lanes overflow with worst-case input: above 1024 dimensions in
the AVX2 kernel and above 2056 in the AVX-512 one with the 16-bit table, and
above 1028 in every kernel with the 8-bit table. So the scan calls the
kernels through `vs_fastscan_accumulate_range` and
`vs_fastscan_accumulate_hacc_range`, which pass at most 128 columns (1024
dimensions) per call and add the results in 32 bits. The same wrappers score
a group a page at a time.

### Pruning

Once a group's sums are complete, the scan turns each into a distance
estimate, `f_add + g_add − 2 × f_rescale × dot`, subtracts the error bound
`f_error × g_error`, and keeps only vectors whose lower bound beats the
current top-k threshold. On AVX-512 this runs 16 vectors at a time with a
masked compare that also drops deleted lanes. Elsewhere it runs a vector at
a time.

## Posting lists

### Mixed chains

A list's chain can hold AoS and fastscan pages, and the scan
(`prism_posting_scan_cluster`) picks the kernel for each page from
`PRISM_POSTING_PAGE_FASTSCAN` in its flags.

- **Build**: the build writes fastscan pages by default. With the
  `fastscan` reloption at `auto` (the default) or `on`, the builder writes
  each list as fastscan pages directly, at every dimension. `fastscan = off`
  writes AoS pages.
- **Inserts** never write into fastscan pages. A new vector goes onto an AoS
  page after the list's fastscan pages.
- **Rewrites**: `prism_convert_posting_to_fastscan` rewrites a list as
  fastscan pages, and a split in a fastscan index writes its new lists the
  same way. Fastscan pages are only ever written for a whole list, in order.

`PRISM_META_FLAG_FASTSCAN` on the metapage records that the index was built
with fastscan, so a scan sets up its lookup table and scratch up front.

### Group format

Each group has its codes and its group metadata:

```
ItemPointerData tids[32]         192B
float f_add[32]                  128B
float f_rescale[32]              128B
float f_error[32]                128B
uint32_t tombstone_mask            4B
```

`tombstone_mask` has one bit per lane. VACUUM sets a lane's bit to delete
that vector, since its codes are interleaved with 31 others and cannot be
marked on their own (`prism_fastscan_group_tombstone_mask` in
`posting_page.h`). The kernel still sums every lane, but a masked lane is
dropped whatever its score.

### Page layout

The codes of a list's groups form one stream, in group order, that runs
across the list's fastscan pages. Each page holds a run of that stream,
after the group metadata of the groups scored on it:

```
 ┌────────┬────────┬─ ─ ─ ─┬──────────────────────────────┬────┐
 │ meta   │ meta   │       │ codes: a run of the stream,  │░░░░│
 │ 580 B  │ 580 B  │  …    │ cut at chunks                │    │
 └────────┴────────┴─ ─ ─ ─┴──────────────────────────────┴────┘
  group metadata, in group order                           unused
```

The metadata at the top of a page is for the groups whose codes **end** on
the page, wherever they started, plus any deferred from the previous page.

On the head page all of this follows the reference vector. Past 1968
dimensions the reference fills the START and MIDDLE pages, and the stream
starts on the END page, after the rest of the reference. Nothing is added
to the page for the stream: the 24-byte opaque already has the two counts a
reader needs, with one field read differently on fastscan pages.

```
 opaque (24 B)
 ┌──────────┬──────────┬────────┬───────┬─────────┬──────────────┬────────┐
 │next_blkno│cluster_id│entry_  │ flags │ page_id │ max_entries  │ head   │
 │          │          │count   │       │         │ or code_bytes│metadata│
 │ 4 B      │ 4 B      │ 2 B    │ 2 B   │ 2 B     │ 2 B          │ 8 B    │
 └──────────┴──────────┴────────┴───────┴─────────┴──────────────┴────────┘
```

| Opaque field | On a fastscan page |
|---|---|
| `entry_count` | vectors whose metadata is on the page; the page holds the metadata of `ceil(entry_count / 32)` groups |
| `code_bytes` | length of the page's code run, at most 8144. It shares storage with an AoS page's `max_entries`: fastscan pages take no inserts, so they have no capacity to track, and `PRISM_POSTING_PAGE_FASTSCAN` in `flags` says which reading applies. |
| `flags` | the page's part of the head (a 2-bit field), then `FASTSCAN`, `TOMBSTONED` and `DELETED` |

The head part says what a page holds ahead of, or instead of, entries:

| Head part | Inline list | Spilled list |
|---|---|---|
| START | whole reference, then entries | a page of reference only |
| MIDDLE | (none) | a page of reference only |
| END | (none) | rest of the reference, then entries |
| NONE | entries | entries |

START is always the list's first page, the one that carries the head
metadata (live count, chain tail) and that a leaf centroid points at.
Whether it also holds entries follows from the dimension, which every
reader knows. The values share bits, so code compares the field rather than
testing a bit (`prism_posting_head_part` in `posting_page.h`).

Where a page sits in the stream is not stored. A scan reads a list from its
head, so the position and group it reached on one page are where the next
page continues: whether the page opens with the rest of a group, a new
group, or deferred metadata follows from that state. Only the list's last
group can be partly full, so readers that only need metadata
(deleted-vector marking, split, inspect) find group `g`'s at a fixed offset.

### Where the stream may break for a new page

| Part | May break for a new page at | Why |
|---|---|---|
| Fastscan codes | any 64-byte chunk | every SIMD kernel steps in 16, 32 or 64 bytes |
| Fastscan metadata | never | it must be on the page where the group is scored |
| AoS entry | never | entries are small, so the loss stays under one entry |

Where the codes are not a whole number of chunks (dimensions not divisible
by 16), the last 32 bytes stay together as one piece.

Group metadata goes on the page where its group's codes end. When it no
longer fits there, it is deferred to the next page, and the **next group's
codes** use the space meanwhile. At most two groups' metadata is deferred at
once: with two deferred, the next group's codes may start on the page but
not finish there. So a scan has at most three groups in flight
(`PRISM_FASTSCAN_MAX_INFLIGHT_GROUPS`), two with deferred metadata and one
adding up its codes.

### Why it is laid out this way

The storage layer's contract shapes several of these rules: a reader holds
**one page at a time**. PostgreSQL buffer locks have no deadlock detection,
and holding one page never deadlocks. Reranking also reads heap pages
through the same storage slot, so no index page may be held across it.

**Codes break at 64-byte chunks.** *Reason: performance.* Every kernel steps
through codes in 16-, 32- or 64-byte units, and the 16-bit lookup table
comes in 128-byte blocks covering two columns. Any 64-byte boundary is
therefore a valid place for a kernel call to start, so a group split across
pages is scored in place by the unchanged kernels. A smaller unit would need
tail handling in the middle of a group, and a larger one would leave more of
each page empty.

**Group metadata never splits, and goes on the page where its group's codes
end.** *Reason: reading in place within the one-page contract.* Pruning
needs a group's complete sums and its metadata at the same moment, which is
when the scan reads the group's last chunk. Placed there, the metadata is on
the page the scan holds. The alternatives each cost something:

- Metadata on the page where the codes start would already be released
  when it is needed, so the scan would copy 580 bytes for every group that
  crosses a page.
- Holding two pages would change the storage contract, its lock-ordering
  rules and the rerank path.
- Splitting metadata like codes would need the same copy.

**Group metadata that does not fit is deferred to the next page, while the
next group's codes use the space.** *Reason: space.* Without this, a page
would end with up to 579 bytes empty whenever the codes leave less room than
one group's metadata, about 432 and 500 bytes per page at 96 and 128
dimensions. **At most two groups' metadata is deferred at once.** *Reason:
bounded scan state.* The scan then carries running sums for at most three
groups (384 bytes), and a third deferred group would only help below about
72 dimensions.

**Group metadata comes first on a page, codes after, rather than each
group's metadata after its own codes.** *Reasons: simplicity for metadata
readers, and reading in place.* Deleted-vector marking, split and inspect
read only metadata. With the metadata first, group `g`'s is at
`content + g × 580` and the number of groups comes from `entry_count`, so
these readers handle any page on its own without knowing where groups start
in the code stream. The codes stay one continuous run, read in place from
start to end and described by a single length, and the scan moves forward
through both regions instead of back and forth group by group.

**A page does not store its position in the stream.** *Reason: space, and
nothing needs it.* Only the scan reads codes, and it always reads a list
from its head, carrying its stream position and next group from page to
page. The one fact it cannot carry, the length of the page's code run, fits
in the opaque's `max_entries` field. Storing positions in a page header
would cost 16 bytes per page (0.2%) and serve only as a cross-check.
Corruption is still caught: metadata that arrives before its group's codes
are complete, or a stream that ends inside a group, is reported.

**A spilled list's entries start on its head's END page.** *Reason:
space.* At 2048, 4096, 6144 and 8192 dimensions the rest of the reference
after the START and MIDDLE pages takes under 200 bytes, so a page of its own
would leave nearly all of it empty, about one page per list.

**Each page says which part of the head it is.** *Reason: readers that do
not walk from the head.* An insert opens the chain's tail directly, VACUUM
and rebalance scan every block for list heads, and checks run on any page.
They must tell from the page alone where its entries start, or that it has
none, and a list split or converted after the build has its head pages
wherever the relation grew, so the block number says nothing. Four parts
(START, MIDDLE, END, none) fit a 2-bit field; the dimension separates an
inline START page, which holds entries, from a spilled one, which does
not.

**Fastscan and AoS pages stay separate, and inserts append AoS pages.**
*Reason: simplicity.* Letting inserts fill the end of the last fastscan
page would need pages with two regions in every reader and writer, to save
at most one page per list.

### Fastscan pages at 768 dimensions

```
 HEAD PAGE                                                  8144 B
 ┌────────────┬───────┬────────────────┬────────────────┬─┐
 │ reference  │ grp 0 │ group 0 codes  │ group 1 codes  │░│
 │ 3072 B     │ meta  │ 48 chunks      │ chunks 0–21   ►│░│ 12 B
 └────────────┴───────┴────────────────┴────────────────┴─┘ unused
 PAGE 2
 ┌───────┬───────┬─────────────┬────────────────┬─────────────┬─┐
 │ grp 1 │ grp 2 │►grp 1 codes │ group 2 codes  │ grp 3 codes │░│
 │ meta  │ meta  │ ch 22–47    │ 48 chunks      │ ch 0–34    ►│░│ 8 B
 └───────┴───────┴─────────────┴────────────────┴─────────────┴─┘
 PAGE 3
 ┌───────┬───────┬─────────┬────────────────┬────────────────┬─┐
 │ grp 3 │ grp 4 │►grp 3   │ group 4 codes  │ group 5 codes  │░│
 │ meta  │ meta  │ ch35–47 │ 48 chunks      │ 48 chunks      │░│ 8 B
 └───────┴───────┴─────────┴────────────────┴────────────────┴─┘
 PAGE 4
 ┌─────┬─────┬─────┬────────────────┬────────────────┬──────┬─┐
 │ g5  │ g6  │ g7  │ group 6 codes  │ group 7 codes  │grp 8►│░│
 │meta │meta │meta │ 48 chunks      │ 48 chunks      │ch 0–3│░│ 4 B
 └─────┴─────┴─────┴────────────────┴────────────────┴──────┴─┘
```

Group 5's metadata doesn't fit after its codes on page 3, so it opens page 4.
Four pages hold groups 0–7 and start group 8, leaving 32 bytes empty. With
whole groups per page, the same four pages would hold groups 0–6.

### Metadata deferred to the next page (96 dimensions)

At 96 dimensions a group is 384 bytes of codes and 580 bytes of metadata.
Two pages from the middle of a long list:

```
 PAGE n                                                     8144 B
 ┌────────────────┬─────────────────────┬────────┬────────┬──┐
 │ meta of        │ codes of groups     │grp 24  │grp 25  │░░│
 │ groups 16–23   │ 17–23               │codes   │codes   │░░│ 48 B
 └────────────────┴─────────────────────┴────────┴────────┴──┘
 PAGE n+1
 ┌────────┬────────┬────────────────┬──────────────────┬───┬──┐
 │grp 24  │grp 25  │ meta of        │ codes of groups  │►33│░░│
 │meta    │meta    │ groups 26–32   │ 26–32            │   │░░│ 44 B
 └────────┴────────┴────────────────┴──────────────────┴───┴──┘
```

Group 24's metadata doesn't fit after its codes, so group 25's codes take
the space, and the metadata of both opens page n+1. With whole groups, a
page at this dimension holds eight of them and leaves 432 bytes empty.

### Groups bigger than a page (1968 dimensions)

At 1968 dimensions a group is 8452 bytes, more than a page, and the
reference vector fills most of the head page:

```
 HEAD PAGE
 ┌──────────────────────────────────────────────────────────┬───┬┐
 │ reference 7872 B                                         │g0►││
 └──────────────────────────────────────────────────────────┴───┴┘
                                                group 0 chunks 0–3
 PAGE 2
 ┌───────────────────────────────────────────────────┬───────┬┐
 │►group 0 codes, chunks 4–122                       │grp 1 ►││
 │                                           7616 B  │ch 0–7 ││
 └───────────────────────────────────────────────────┴───────┴┘
 PAGE 3
 ┌───────┬─────────────────────────────────────────┬──────┬─┐
 │ grp 0 │►group 1 codes, chunks 8–122             │grp 2►│░│
 │ meta  │                                  7360 B │ch 0–2│░│ 12 B
 └───────┴─────────────────────────────────────────┴──────┴─┘
```

Every group crosses at least one page boundary, and each group's metadata
opens the page after its codes end.

### Spilled reference (4096 dimensions)

At 4096 dimensions the reference vector takes three pages, the head's
START, MIDDLE and END, and a group is 16384 bytes of codes, 256 chunks. The
END page holds the reference's last 24 floats and then the start of the
stream:

```
 PAGE 1 (START), PAGE 2 (MIDDLE)
 ┌──────────────────────────────────────────────────────────┐
 │ reference floats (2036 per page)                 8144 B ►│
 └──────────────────────────────────────────────────────────┘
 PAGE 3 (END)
 ┌──┬───────────────────────────────────────────────────┬──┐
 │► │ group 0 codes, chunks 0–124                      ►│░░│ 48 B
 └──┴───────────────────────────────────────────────────┴──┘
  reference floats 4072–4095 (96 B)
 PAGE 4
 ┌──────────────────────────────────────────────────────┬─┐
 │►group 0 codes, chunks 125–251                       ►│░│ 16 B
 └──────────────────────────────────────────────────────┴─┘
 PAGE 5
 ┌───────┬───────┬──────────────────────────────────────┬─┐
 │ grp 0 │►grp 0 │ group 1 codes, chunks 0–113         ►│░│ 12 B
 │ meta  │252–255│                                      │░│
 └───────┴───────┴──────────────────────────────────────┴─┘
```

The scan gathers the reference from pages 1 and 2, then reads page 3 once,
taking the rest of the reference and starting the stream there. An AoS list
fills the rest of page 3 with entries the same way, and inserts into an
empty list land there too.

### How search reads a list

```
 read HEAD PAGE ──► query − centroid, from the reference read in place
                    group 0: chunks 0–47 → 32 sums → metadata → score
                    group 1: chunks 0–21 → 32 running sums
 release HEAD PAGE
 read PAGE 2 ─────► group 1: chunks 22–47 → metadata → score
                    group 2: whole → score
                    group 3: chunks 0–34 → 32 running sums
 release PAGE 2
 read PAGE 3 ─────► group 3: chunks 35–47 → metadata → score …
```

- Pages are read once, in chain order, one at a time, as the storage layer
  requires.
- Codes and the reference are read in place. Nothing is copied.
- Between pages the scan keeps the running sums of up to three groups and
  its position in the stream, which is where the next page continues. A
  page whose metadata arrives before its group's codes are complete, or a
  list that ends inside a group, is reported as corruption, and so is a
  run that does not end on a chunk boundary, before any kernel reads it.
- A group is pruned once its last chunk and its metadata have been read.
- A page whose vectors are all deleted is skipped only when nothing else
  passes through it: its codes end on a group boundary and every group they
  touch has its metadata there.

### Unused space per page

Averaged over a long list, so the head page and the list's end don't count.

| Dimensions | Group size | Fastscan | AoS |
|---:|---:|---:|---:|
| 96 | 964 B | 46 B (0.6%) | 16 B (0.2%) |
| 128 | 1092 B | 50 B (0.6%) | 8 B (0.1%) |
| 256 | 1604 B | 60 B (0.7%) | 32 B (0.4%) |
| 384 | 2116 B | 1 B (0.0%) | 52 B (0.6%) |
| 768 | 3652 B | 7 B (0.1%) | 24 B (0.3%) |
| 1024 | 4676 B | 9 B (0.1%) | 4 B (0.0%) |
| 1536 | 6724 B | 11 B (0.1%) | 88 B (1.1%) |
| 1968 | 8452 B | 12 B (0.1%) | 104 B (1.3%) |
| 2048 | 8772 B | 12 B (0.2%) | 140 B (1.7%) |
| 4096 | 16964 B | 14 B (0.2%) | 164 B (2.0%) |
| 8192 | 33348 B | 15 B (0.2%) | 836 B (10.3%) |

A list's last group is usually only partly full, and its empty lanes still
take space. At the default list size that makes fastscan lists 3–5% larger
than AoS from about 1000 dimensions, where a fastscan vector is barely
smaller than an AoS entry; built indexes of 1536-, 2048- and 3072-dimension
embeddings come out 3–4% larger. At 8192 dimensions whole AoS entries leave
a tenth of each page empty, which should more than make up for it.

## Not yet implemented

### SIMD pruning without AVX-512

Pruning runs a vector at a time on AVX2 and NEON. It costs about the same
per group at any dimension while the kernels get cheaper as dimensions drop,
so an AVX2 or NEON version would matter most for small vectors.

## Costs and open questions

- **Kernel calls.** A group split across pages takes one extra kernel call
  per page boundary, plus adding 32 sums into the running total.
- **Deleted pages.** A fully deleted page can't always be skipped, since a
  live group's codes may pass through it.
- **Probe re-ranking** reads a spilled list's END page twice:
  once for the reference when ranking the candidate lists, and again for
  the entries if the list is scanned, as it does an inline list's head.
- **Recall at 4096–8192 dimensions** is unmeasured, for lack of a real
  dataset that wide. Up to 3072 dimensions the 16-bit lookup table keeps
  recall at the AoS level (see [Validation](#validation)), though it sums
  up to 2048 subquantizers per vector at 8192.

## Measuring

`vectorsearch bench search` builds a standalone index with the real builder
and page layout, and queries it through the real routing, scan and rerank,
so it measures the layout end to end. Run it with `--fastscan 16` (or `8`)
and without, against a real dataset:

```
vectorsearch bench search --hdf5 cohere-wikipedia-22-12-1M-angular.hdf5 \
    --metric angular --queries 1000 --runs 3 --nprobe 32 --fastscan 16
```

The build log gives the index's page count, and the results give recall,
latency and QPS. The parallel build does not cluster identically from run
to run, so recall moves by about a hundredth between builds. Repeat each
run a few times, and compare against the AoS path as a control.

## Validation

- Unit tests cover the stream at 16–8192 dimensions, spilled references
  included: metadata never comes before its group's codes are complete,
  waste stays under a chunk except where metadata is deferred, and every
  vector scores as it does in AoS. A page with a shortened code run makes
  the scan report corruption.
- Range sums match whole-group sums at 8–8192 dimensions on every SIMD
  variant.
- Serial and parallel standalone builds at 4096 dimensions write fastscan
  lists from the head's END page on, and every stored vector finds
  itself.
- On 500K OpenAI `text-embedding-3-large` embeddings at 3072 dimensions,
  and their 2048- and 1536-dimension shortenings, fastscan recall@10 matches
  AoS within the spread between builds at nprobe 16, 32 and 64.
- The PostgreSQL regression tests check that 1968- and 8192-dimension
  indexes are fastscan and scanned through the index both before and after
  an insert adds an AoS page, that a 4096-dimension AoS list keeps its
  first entries on its head's END page and converts to fastscan, and
  they run the mutation scenarios on fastscan lists with spilled
  references.
