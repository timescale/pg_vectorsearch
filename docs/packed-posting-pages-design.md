# Packed posting pages

Status: the packed fastscan layout is implemented (`posting_page.h`,
`posting_build.c`, `posting_scan.c`). The reference vector still sits whole
on the head page, so AoS lists and the 1968-dimension limit are unchanged;
[Not yet implemented](#not-yet-implemented) covers letting it cross pages.
Space figures for the packed layout come from the real builder; figures for
0.1.0 come from its layout constants. Neither comes from built indexes, and
no speed figures are measured here.

## Summary

Fastscan is the posting format that searches fastest, and it should be the
default at every dimension the index supports. The 0.1.0 layout prevents
that in two ways:

1. **Fastscan stops at 944 dimensions.** A fastscan group (32 vectors) must
   fit whole on one page, and the first page of a posting list must hold a
   group next to the list's reference vector. At 1024 and 1536 dimensions,
   both common embedding sizes, the build falls back to the slower AoS
   format. Above 1888 dimensions a group no longer fits even an empty page.
2. **Fastscan wastes space.** A group that doesn't fit in the rest of a page
   starts on the next one and leaves the gap empty. The gap reaches 22% of a
   page at 384 dimensions and 47% at 944. At 100M vectors this costs 7.5% of
   posting storage at 768 dimensions and 35% at 944.

The packed layout treats a list's fastscan codes as one stream that
continues across pages. A group's codes may break at any 64-byte boundary,
which matches the SIMD kernels' step sizes, so the kernels run unchanged and
read the codes in place. Each page then wastes under 64 bytes, and fastscan
works at every dimension the index supports. At 100M vectors and 768
dimensions this saves **1,195,330,739 bytes (7.5%)** of posting storage
compared to 0.1.0, and 35% at 944 dimensions.

AoS lists already pack tightly and keep their layout.

## Terms

- **Posting list**: the vectors assigned to one cluster, stored as a chain of
  8 KB pages.
- **Head page**: the first page of a posting list.
- **Reference vector**: the cluster's full-precision centroid (`4 × dims`
  bytes). Search uses it to compute `query − centroid` before scoring any
  vector in the list.
- **AoS entry**: one vector in the AoS format. A 20-byte header (TID, flags,
  three correction factors) followed by one bit per dimension.
- **Fastscan group**: 32 vectors in the fastscan format. The 1-bit codes of
  all 32 are interleaved so one SIMD lookup scores 32 vectors at once.
- **Chunk**: 64 bytes of a group's codes, covering 16 dimensions of all
  32 vectors.
- **Group metadata**: a group's 32 TIDs, 32 × 3 correction factors, and its
  deleted-vector mask (580 bytes).

## The 0.1.0 layout

### Page frame

Every posting page is an 8 KB PostgreSQL page with 8144 bytes for data.

```
 0      24                                         8168   8192
 ┌──────┬──────────────────────────────────────────┬──────┐
 │ page │ content: 8144 bytes                      │opaque│
 │header│ (reference, AoS entries, or groups)      │ 24 B │
 └──────┴──────────────────────────────────────────┴──────┘
```

The 24-byte opaque area holds the next page in the chain, the cluster id, the
entry count and capacity, and flags. On the head page it also holds the live
vector count and the last page of the chain.

### AoS posting lists

The head page starts with the reference vector. AoS entries fill the rest of
the head and every following page. Entries never cross a page, but they're
small, so less than one entry's worth of space goes unused.

```
 one AoS entry (116 B at 768 dims)
 ┌──────────────────────────────┬──────────────────────────┐
 │ header 20 B: TID, flags,     │ 1-bit codes 96 B         │
 │ f_add, f_rescale, f_error    │ (one bit per dimension)  │
 └──────────────────────────────┴──────────────────────────┘

 HEAD PAGE (768 dims)                                       8144 B
 ┌────────────────────┬────────────────────────────────────┬─┐
 │ reference          │ AoS entries 0–42                   │░│
 │ 768 floats, 3072 B │ 43 × 116 B = 4988 B                │░│ 84 B
 └────────────────────┴────────────────────────────────────┴─┘ unused

 FOLLOWING PAGES
 ┌──────────────────────────────────────────────────────────┬┐
 │ AoS entries: 70 × 116 B = 8120 B                         ││ 24 B
 └──────────────────────────────────────────────────────────┴┘ unused
```

Unused bytes per AoS page in 0.1.0:

| Dimensions | Entry size | Head page: entries | Head page: unused | Other pages: entries | Other pages: unused |
|---:|---:|---:|---:|---:|---:|
| 96 | 32 B | 242 | 16 B (0.2%) | 254 | 16 B (0.2%) |
| 128 | 36 B | 212 | 0 B (0.0%) | 226 | 8 B (0.1%) |
| 256 | 52 B | 136 | 48 B (0.6%) | 156 | 32 B (0.4%) |
| 384 | 68 B | 97 | 12 B (0.1%) | 119 | 52 B (0.6%) |
| 512 | 84 B | 72 | 48 B (0.6%) | 96 | 80 B (1.0%) |
| 768 | 116 B | 43 | 84 B (1.0%) | 70 | 24 B (0.3%) |
| 1024 | 148 B | 27 | 52 B (0.6%) | 55 | 4 B (0.0%) |
| 1536 | 212 B | 9 | 92 B (1.1%) | 38 | 88 B (1.1%) |
| 1968 | 268 B | 1 | 4 B (0.0%) | 30 | 104 B (1.3%) |

1968 dimensions is the 0.1.0 maximum. Beyond it, the reference vector plus
one entry no longer fit on the head page.

### Fastscan posting lists

A fastscan group holds 32 vectors, and its size grows with the dimension:
`580 + 4 × dims` bytes.

```
 one fastscan group, 0.1.0 order (3652 B at 768 dims)
 ┌──────────┬───────┬───────────┬─────────┬──────────────────────┬────┐
 │ 32 TIDs  │ f_add │ f_rescale │ f_error │ codes                │mask│
 │  192 B   │ 128 B │   128 B   │  128 B  │ 3072 B = 48 chunks   │ 4 B│
 └──────────┴───────┴───────────┴─────────┴──────────────────────┴────┘

 one 64-byte chunk = 16 dimensions of all 32 vectors
 ┌────────────────┬────────────────┬────────────────┬────────────────┐
 │ dims 0–3       │ dims 4–7       │ dims 8–11      │ dims 12–15     │
 │ 32 × 4-bit code│ 32 × 4-bit code│ 32 × 4-bit code│ 32 × 4-bit code│
 │ 16 B           │ 16 B           │ 16 B           │ 16 B           │
 └────────────────┴────────────────┴────────────────┴────────────────┘
```

A group must fit whole on a page. Whatever is left after the last group that
fits stays empty:

```
 HEAD PAGE (768 dims)                                       8144 B
 ┌──────────────────┬──────────────────────┬──────────────────┐
 │ reference 3072 B │ group 0  3652 B      │░░░░░░░░░░░░░░░░░░│ 1420 B
 └──────────────────┴──────────────────────┴──────────────────┘ unused
 PAGE 2
 ┌──────────────────────┬──────────────────────┬──────┐
 │ group 1  3652 B      │ group 2  3652 B      │░░░░░░│ 840 B unused
 └──────────────────────┴──────────────────────┴──────┘
 PAGE 3
 ┌──────────────────────┬──────────────────────┬──────┐
 │ group 3              │ group 4              │░░░░░░│ 840 B unused
 └──────────────────────┴──────────────────────┴──────┘
```

Unused bytes per fastscan page in 0.1.0:

| Dimensions | Group size | Head page: groups | Head page: unused | Other pages: groups | Other pages: unused |
|---:|---:|---:|---:|---:|---:|
| 96 | 964 B | 8 | 48 B (0.6%) | 8 | 432 B (5.3%) |
| 100 | 996 B | 7 | 772 B (9.5%) | 8 | 176 B (2.2%) |
| 128 | 1092 B | 6 | 1080 B (13.3%) | 7 | 500 B (6.1%) |
| 256 | 1604 B | 4 | 704 B (8.6%) | 5 | 124 B (1.5%) |
| 384 | 2116 B | 3 | 260 B (3.2%) | 3 | 1796 B (22.1%) |
| 512 | 2628 B | 2 | 840 B (10.3%) | 3 | 260 B (3.2%) |
| 768 | 3652 B | 1 | 1420 B (17.4%) | 2 | 840 B (10.3%) |
| 944 | 4356 B | 1 | 12 B (0.1%) | 1 | 3788 B (46.5%) |

### Where 0.1.0 can use fastscan

`fastscan = auto` picks fastscan only where a group fits on the head page
next to the reference vector (`prism_resolve_fastscan`). `fastscan = on`
fails everywhere else.

| Dimensions | Posting format in 0.1.0 | Why |
|---|---|---|
| up to 944 | fastscan | a group fits next to the reference |
| 945–1888 | AoS only | a group fits a page, but not next to the reference |
| 1889–1968 | AoS only | a group is bigger than a page |
| above 1968 | not supported | the reference plus one AoS entry exceed the head page |

### Cost at 100M vectors

The figures below assume:

- The build's default list sizing: about 256 vectors per list at 768
  dimensions, more at small dimensions.
- List sizes spread evenly from 50% to 150% of that target.
- No replication.

Only posting pages are counted, not centroid pages or the table.

| Dimensions | Posting lists | 0.1.0 fastscan: pages per list | 0.1.0 fastscan: total | 0.1.0 AoS: total |
|---:|---:|---:|---:|---:|
| 96 | 131,234 | 3.47 | 3.48 GiB | 3.55 GiB |
| 128 | 147,493 | 3.67 | 4.12 GiB | 4.01 GiB |
| 256 | 213,675 | 3.63 | 5.91 GiB | 5.91 GiB |
| 384 | 280,112 | 4.22 | 9.03 GiB | 7.86 GiB |
| 512 | 347,222 | 3.83 | 10.14 GiB | 9.92 GiB |
| 768 | 390,625 | 4.99 | 14.88 GiB | 13.50 GiB |
| 944 | 390,625 | 8.48 | 25.28 GiB | 15.96 GiB |

From 128 dimensions up, fastscan in 0.1.0 takes as much space as AoS or more,
even though each fastscan vector is smaller than an AoS entry. The gaps
between groups cause most of the difference. The rest comes from each list's
last group, which is usually only partly full.

## The packed layout

### Page layout

The codes of a list's groups form one stream, in group order, that runs
across the list's fastscan pages. Each page holds a run of that stream, plus
the metadata blocks of the groups scored on it:

```
 ┌────────┬────────┬─ ─ ─ ─┬──────────────────────────────┬────┐
 │ meta   │ meta   │       │ codes: a run of the stream,  │░░░░│
 │ 580 B  │ 580 B  │  …    │ cut at chunks                │    │
 └────────┴────────┴─ ─ ─ ─┴──────────────────────────────┴────┘
  metadata blocks, in group order                          unused
```

On the head page all of this follows the reference vector. Nothing is added
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
| `entry_count` | vectors whose metadata is on the page; the page holds `ceil(entry_count / 32)` metadata blocks |
| `code_bytes` | length of the page's code run, at most 8144. It shares storage with an AoS page's `max_entries`: fastscan pages take no inserts, so they have no capacity to track, and `PRISM_POSTING_PAGE_FASTSCAN` in `flags` says which reading applies. |

Where a page sits in the stream is not stored. A scan reads a list from its
head, so the position and group it reached on one page are where the next
page continues: whether the page opens with the rest of a group, a new
group, or deferred metadata follows from that state. Only the list's last
group can be partly full, so readers that only need metadata
(deleted-vector marking, split, inspect) walk the blocks as before.

### Where the stream may break

| Part | May break at | Why |
|---|---|---|
| Fastscan codes | any 64-byte chunk | every SIMD kernel steps in 16, 32 or 64 bytes |
| Fastscan metadata | never | it must be on the page where the group is scored |
| AoS entry | never | entries are small, so the loss stays under one entry |

Where the codes are not a whole number of chunks (dimensions not divisible
by 16), the last 32 bytes stay together as one piece.

A metadata block goes on the page where its group's codes end. When it no
longer fits there, it waits for the next page, and the **next group's codes**
use the space meanwhile. At most two blocks wait at once: with two waiting,
the next group's codes may start on the page but not finish there. So a scan
has at most three groups in flight, two waiting for their metadata and one
adding up its codes.

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
Four pages hold groups 0–7 and start group 8, leaving 32 bytes empty. In
0.1.0 the same four pages hold groups 0–6 and leave 3940 bytes empty.

### Metadata waiting for the next page (96 dimensions)

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
the space, and both metadata blocks open page n+1. With whole groups, a page at
this dimension holds eight of them and leaves 432 bytes empty.

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

Every group crosses at least one page boundary, and each metadata block
opens the page after its codes end.

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
  list that ends inside a group, is reported as corruption.
- A group is pruned once its last chunk and its metadata have been read,
  as in 0.1.0.
- A page whose vectors are all deleted is skipped only when nothing else
  passes through it: its codes end on a group boundary and every group they
  touch has its metadata there.

### Inserts

Inserts never write into fastscan pages, in 0.1.0 or here. A new vector goes
onto an AoS page after the list's fastscan pages, and the scan reads those
with the AoS kernel. Fastscan pages are written only when a list is built,
converted or split, all of which write the whole list in order.

### AoS pages

AoS entries keep their size, stay whole, and keep the 0.1.0 layout: the
reference vector on the head page and whole entries filling the rest.

## Space with the packed layout

### Unused bytes per page

Averaged over a long list, so the head page and the list's end don't count.

| Dimensions | Group size | 0.1.0 fastscan | Packed fastscan | 0.1.0 and packed AoS |
|---:|---:|---:|---:|---:|
| 96 | 964 B | 432 B (5.3%) | 46 B (0.6%) | 16 B (0.2%) |
| 100 | 996 B | 176 B (2.2%) | 34 B (0.4%) | 8 B (0.1%) |
| 128 | 1092 B | 500 B (6.1%) | 50 B (0.6%) | 8 B (0.1%) |
| 256 | 1604 B | 124 B (1.5%) | 60 B (0.7%) | 32 B (0.4%) |
| 384 | 2116 B | 1796 B (22.1%) | 1 B (0.0%) | 52 B (0.6%) |
| 512 | 2628 B | 260 B (3.2%) | 4 B (0.0%) | 80 B (1.0%) |
| 768 | 3652 B | 840 B (10.3%) | 7 B (0.1%) | 24 B (0.3%) |
| 944 | 4356 B | 3788 B (46.5%) | 9 B (0.1%) | 24 B (0.3%) |
| 1024 | 4676 B | no fastscan | 9 B (0.1%) | 4 B (0.0%) |
| 1536 | 6724 B | no fastscan | 11 B (0.1%) | 88 B (1.1%) |
| 1968 | 8452 B | no fastscan | 12 B (0.1%) | 104 B (1.3%) |

Letting the next group's codes use the space a waiting metadata block leaves
matters at small dimensions. With a strict "metadata after its own codes"
rule, packed fastscan would still leave about 432 bytes per page at 96
dimensions and 500 at 128, as in 0.1.0.

### Savings at 100M vectors

Fastscan posting storage, with the same assumptions as the 0.1.0 table.
Packed totals come from the builder's page counts.

| Dimensions | 0.1.0 | Packed | Saved | Saved, long lists |
|---:|---:|---:|---:|---:|
| 96 | 3.48 GiB | 3.44 GiB | 45,088,081 B (1.2%) | 4.8% |
| 100 | 3.65 GiB | 3.55 GiB | 101,430,002 B (2.6%) | 1.8% |
| 128 | 4.12 GiB | 3.91 GiB | 227,772,636 B (5.1%) | 5.6% |
| 256 | 5.91 GiB | 5.91 GiB | 0 B (0.0%) | 0.8% |
| 384 | 9.03 GiB | 8.12 GiB | 967,866,768 B (10.0%) | 22.0% |
| 512 | 10.14 GiB | 10.14 GiB | 0 B (0.0%) | 3.1% |
| **768** | **14.88 GiB** | **13.76 GiB** | **1,195,330,739 B (7.5%)** | **10.2%** |
| 944 | 25.28 GiB | 16.36 GiB | 9,575,097,276 B (35.3%) | 46.5% |

"Saved, long lists" is the limit for lists many pages long, as with heavy
replication or a large `target_pages`. Short lists save less because they
still round up to whole pages. At 256 and 512 dimensions, rounding hides the
gain completely at the default list size.

### Fastscan above 944 dimensions

These dimensions use AoS in 0.1.0. Packed fastscan at 100M vectors, with AoS
for comparison:

| Dimensions | Packed fastscan | AoS (0.1.0 and packed) |
|---:|---:|---:|
| 1024 | 17.48 GiB | 16.88 GiB |
| 1536 | 24.55 GiB | 23.81 GiB |
| 1968 | 31.24 GiB | 29.76 GiB |

Packed fastscan is 3–5% larger than AoS. A list's last group is usually only
partly full, and its empty lanes still take space. That cost is the same in
0.1.0 and is not affected by this layout.

## Kernel overflow

The fastscan kernels add in 16-bit SIMD lanes. With the default 16-bit
lookup table (`fastscan_bits = 16`), worst-case input overflows those lanes
above 1024 dimensions in the AVX2 kernel and above 2056 in the AVX-512
kernel. The 8-bit table puts up to 255 into every subquantizer, so its
16-bit result overflows above 1028 dimensions in every kernel. 0.1.0 never
runs fastscan above 944 dimensions, so it never hit either.

The scan now calls the kernels through range wrappers
(`vs_fastscan_accumulate_range` and `vs_fastscan_accumulate_hacc_range`).
Each scores a range of columns, passing at most 128 columns (1024
dimensions) to one kernel call, and adds the results in 32 bits. The same
wrappers score a group a page at a time. The SIMD code is unchanged.

## Not yet implemented

### The reference vector across pages

Above 1968 dimensions the reference vector no longer fits on the head page
with room for an entry. Letting the reference continue across pages, splitting
at any float, would lift that limit for both formats: `query − centroid`
can be computed a piece at a time, straight from each page. AoS entries
would fill whatever the reference leaves free on its last page:

```
 4096 dims, AoS
 HEAD PAGE, PAGE 2
 ┌──────────────────────────────────────────────────────────┐
 │ reference floats (2036 per page)                 8144 B ►│
 └──────────────────────────────────────────────────────────┘
 PAGE 3
 ┌─┬──────────────────────────────────────────────────────┬─┐
 │►│ AoS entries 0–14: 15 × 532 B = 7980 B                │░│ 68 B
 └─┴──────────────────────────────────────────────────────┴─┘ unused
  reference floats 4072–4095 (96 B)
```

Other parts of the index set their own dimension limits too: centroid pages,
the fast rotation, and the accuracy of the lookup tables with many
subquantizers.

## Costs and open questions

- **Kernel calls.** A group split across pages takes one extra kernel call
  per page boundary, plus adding 32 sums into the running total. See
  [Measuring](#measuring).
- **Groups in flight.** The scan keeps running sums for up to three groups
  and its position in the stream.
- **Deleted pages.** A fully deleted page can no longer always be skipped,
  since a live group's codes may pass through it.
- **New on-disk format.** Indexes built by 0.1.0 need `REINDEX`.
- **Recall and speed above 944 dimensions** are unmeasured. Fastscan now
  runs there by default.

## Measuring

`vectorsearch bench fastscan-kernel` measures the layout in isolation:

- **Split accumulate** scores each group in one kernel call, then split at
  a chunk boundary as a page break splits it, and reports the cost per
  split.
- **Posting scan** writes posting lists with the real builder into memory
  pages and scans them with the real scan path, reporting pages per list,
  bytes per vector and throughput, for fastscan and AoS lists. It uses only
  APIs that 0.1.0 also has, so the same benchmark can run against both
  layouts.

```
vectorsearch bench fastscan-kernel --dim 768 --count 200000 --list 256
```

On a Zen 4 machine, a page break costs a few nanoseconds per split group.
Where the layout saves pages, from 768 dimensions up, fastscan scans
proportionally faster. Where it saves none (96 to 384 dimensions at the
default list size), scans run within a few percent of 0.1.0; long lists
at 128 dimensions run up to about 7% slower, since a group's kernel call
there is short and the stream bookkeeping shows. End-to-end search speed
and recall are still to be measured with `vectorsearch bench search`.

## Validation

- Unit tests cover the stream at 16–1968 dimensions: metadata never comes
  before its group's codes are complete, waste stays under a chunk except
  where metadata waits, and every vector scores as it does in AoS. A page
  with a shortened code run makes the scan report corruption. Range sums
  match whole-group sums at 8–4096 dimensions on every SIMD variant.
- Still to do: rebuild real 100-, 384-, 768- and 1024-dimension datasets with
  0.1.0 and with the packed layout. Compare index size, recall, QPS and tail
  latency in release builds, with warm and cold shared buffers.
