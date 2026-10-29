# Packed posting pages

Status: design proposal. Nothing here is implemented. Space figures come
from a model built on the 0.1.0 layout constants (`posting_page.h`,
`fastscan.h`), not from built indexes. No speed figures are measured.

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

The proposal stores each posting list as one byte stream that continues
across pages. Fastscan groups may break at any 64-byte boundary in their
codes, which matches the SIMD kernels' step sizes, so the kernels run
unchanged and read the codes in place. Each page then wastes under 64 bytes,
and posting pages no longer limit which dimensions can use fastscan. At 100M
vectors and 768 dimensions this saves **1,195,330,739 bytes (7.5%)** of
posting storage compared to 0.1.0, and 35% at 944 dimensions.

AoS lists already pack tightly and keep their layout. The stream lets the
reference vector continue across pages as well, which removes the posting
pages' 1968-dimension limit for both formats. Two things remain: a kernel
overflow must be fixed before fastscan runs above 1024 dimensions, and other
parts of the index set their own dimension limits.

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

### One stream per posting list

A posting list becomes one byte stream, written into the content area of its
pages in chain order. Page boundaries cut the stream wherever they fall,
subject to the rules below.

```
 ┌───────────┬──────────────────────┬──────────────────────┬─── ─ ─
 │ reference │ group 0              │ group 1              │
 │ vector    │ codes      │ metadata│ codes      │ metadata│ …
 └───────────┴──────────────────────┴──────────────────────┴─── ─ ─
   4×dims B    4×dims B     580 B
```

Each group stores its codes first and its metadata after them. The
deleted-vector mask moves into the metadata block.

### Where the stream may break

| Part | May break at | Why |
|---|---|---|
| Reference vector | any float (4 B) | `query − centroid` is computed in pieces, straight from the page |
| Fastscan codes | any 64-byte chunk | every SIMD kernel steps in 16, 32 or 64 bytes |
| Fastscan metadata | never | it must be on the page where the group is scored |
| AoS entry | never | entries are small, so the loss stays under one entry |

Where the codes are not a whole number of chunks (dimensions not divisible
by 16), the last 32 bytes stay together as one piece.

When a group's codes end with less than 580 bytes left on the page, its
metadata moves to the start of the next page. The **next group's codes** fill
the gap meanwhile, so it doesn't go to waste. The scan then has two groups in
flight: one waiting for its metadata, and one adding up its codes.

### Fastscan pages at 768 dimensions

```
 HEAD PAGE                                                  8144 B
 ┌──────────────────┬──────────────────┬───────┬──────────────┬─┐
 │ reference        │ group 0 codes    │ grp 0 │ group 1 codes│░│
 │ 768 floats       │ 48 chunks        │ meta  │ chunks 0–21  │░│
 │ 3072 B           │ 3072 B           │ 580 B │ 1408 B ─────►│░│ 12 B
 └──────────────────┴──────────────────┴───────┴──────────────┴─┘ unused
 0               3072               6144    6724           8132

 PAGE 2
 ┌──────────────┬───────┬──────────────────┬───────┬──────────────┬─┐
 │ group 1 codes│ grp 1 │ group 2 codes    │ grp 2 │ group 3 codes│░│
 │ chunks 22–47 │ meta  │ 48 chunks        │ meta  │ chunks 0–34  │░│
 │►1664 B       │ 580 B │ 3072 B           │ 580 B │ 2240 B ─────►│░│ 8 B
 └──────────────┴───────┴──────────────────┴───────┴──────────────┴─┘
 0           1664    2244               5316    5896           8136

 PAGE 3
 ┌──────────┬───────┬──────────────────┬───────┬─ ─ ─
 │ grp 3    │ grp 3 │ group 4 codes    │ grp 4 │ …
 │ ch 35–47 │ meta  │ 48 chunks        │ meta  │
 │►832 B    │ 580 B │ 3072 B           │ 580 B │
 └──────────┴───────┴──────────────────┴───────┴─ ─ ─
 0        832     1412               4484    5064
```

In 0.1.0, three pages hold groups 0–4 and 3100 bytes stay empty. With the
packed layout, the first two pages leave 20 bytes empty, and page 3 still
has 3080 bytes free after group 4, enough for all of group 5's codes.

### Metadata moving to the next page (96 dimensions)

At 96 dimensions a group is 384 bytes of codes and 580 bytes of metadata.
Here are two pages from the middle of a long list. Page n opens with the
metadata of the last group from page n−1, then holds seven whole groups:

```
 PAGE n                                                     8144 B
 ┌───────┬─ ─ ─ ─ ─ ─ ─ ─┬────────────┬────────────┬──┐
 │ grp 8 │ groups 9–15   │ group 16   │ group 17   │░░│
 │ meta  │ (whole)       │ codes 384 B│ codes 384 B│░░│ 48 B
 └───────┴─ ─ ─ ─ ─ ─ ─ ─┴────────────┴────────────┴──┘ unused
 0     580            7328         7712         8096

 PAGE n+1
 ┌───────────┬───────────┬────────────────────── ─ ─
 │ group 16  │ group 17  │ group 18 codes …
 │ meta 580 B│ meta 580 B│
 └───────────┴───────────┴────────────────────── ─ ─
 0           580         1160
```

Group 16's metadata doesn't fit after its codes, so group 17's codes take the
space and both metadata blocks open page n+1. The whole-group rule of 0.1.0
would end page n after group 15 and leave 432 bytes empty.

### High dimensions (2048): the reference crosses a page too

```
 HEAD PAGE
 ┌──────────────────────────────────────────────────────────┐
 │ reference floats 0–2035                          8144 B ►│
 └──────────────────────────────────────────────────────────┘
 PAGE 2
 ┌─┬───────────────────────────────────────────────────────┬┐
 │►│ group 0 codes, chunks 0–125                    8064 B►││ 32 B
 └─┴───────────────────────────────────────────────────────┴┘ unused
  reference floats 2036–2047 (48 B)
 PAGE 3
 ┌──┬───────┬───────────────────────────────────────────────┬┐
 │► │ grp 0 │ group 1 codes, chunks 0–115            7424 B►││ 12 B
 │  │ meta  │                                               ││ unused
 └──┴───────┴───────────────────────────────────────────────┴┘
  group 0 chunks 126–127 (128 B)
 PAGE 4
 ┌────────┬───────┬───────────────────────────── ─ ─
 │►grp 1  │ grp 1 │ group 2 codes …
 │ chunks │ meta  │
 │ 116–127│       │
 └────────┴───────┴───────────────────────────── ─ ─
```

At 2048 dimensions a group (8772 B) is bigger than a page, so every group
crosses at least one page boundary.

### AoS pages

AoS entries keep their size and stay whole. Up to 1968 dimensions the packed
AoS layout is the same as in 0.1.0: the reference sits on the head page and
whole entries fill the rest.

Above 1968 dimensions the reference continues onto the next pages, and AoS
entries fill whatever the reference leaves free on its last page:

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
 FOLLOWING PAGES
 ┌──────────────────────────────────────────────────────────┬┐
 │ AoS entries: 15 × 532 B = 7980 B                         ││ 164 B
 └──────────────────────────────────────────────────────────┴┘ unused
```

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
- Between pages the scan keeps the running sums of at most two groups
  (32 × 4 bytes each) and the chunk where it stopped.
- A group is pruned once its last chunk and its metadata have been read,
  as in 0.1.0.

### Inserts

A new vector takes a free lane in the list's last group. Its codes go into
that lane in every chunk, and its TID and factors into the metadata.

```
 PAGE n                          PAGE n+1
 ┌───┬─────────────────────┐   ┌────────────────┬───────┐
 │ … │ last group ch 0–k   │──►│ chunks k+1–47  │ meta  │
 └───┴─────────────────────┘   └────────────────┴───────┘
        ▲ write lane 20             ▲ write lane 20   ▲ TID, factors,
          in each chunk               in each chunk     clear deleted bit
```

If that group crosses a page, the insert changes both pages. It locks them in
chain order and WAL-logs them in one record. Inserts into a group on a single
page touch one page, as in 0.1.0.

## Space with the packed layout

### Unused bytes per page

Averaged over a long list, so the head page doesn't count:

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
| 2048 | 8772 B | no fastscan | 12 B (0.2%) | 140 B (1.7%) |
| 4096 | 16964 B | no fastscan | 14 B (0.2%) | 164 B (2.0%) |

AoS above 1968 dimensions needs the packed layout.

Moving the metadata ahead of the next group's codes matters at small
dimensions. With a strict "metadata after its own codes" rule, packed
fastscan would still leave 432 bytes per page at 96 dimensions and 500 at 128,
as in 0.1.0.

### Savings at 100M vectors

Fastscan posting storage, with the same assumptions as the 0.1.0 table:

| Dimensions | 0.1.0 | Packed | Saved | Saved, long lists |
|---:|---:|---:|---:|---:|
| 96 | 3.48 GiB | 3.44 GiB | 45,088,081 B (1.2%) | 4.8% |
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

These dimensions have no fastscan in 0.1.0. Packed fastscan at 100M vectors,
with packed AoS for comparison:

| Dimensions | Packed fastscan | Packed AoS | 0.1.0 |
|---:|---:|---:|---|
| 1024 | 17.48 GiB | 16.88 GiB | AoS only |
| 1536 | 24.55 GiB | 23.81 GiB | AoS only |
| 2048 | 31.24 GiB | 30.72 GiB | not supported |
| 3072 | 46.11 GiB | 44.19 GiB | not supported |
| 4096 | 59.87 GiB | 58.21 GiB | not supported |

Packed fastscan is 2–4% larger than AoS. A list's last group is usually only
partly full, and its empty lanes still take space. That cost is the same in
0.1.0 and is not affected by this proposal.

## Required kernel fix

The default 16-bit lookup table (`fastscan_bits = 16`) adds its low bytes in
16-bit SIMD lanes. With worst-case input these lanes overflow above 1024
dimensions in the AVX2 kernel and above 2056 dimensions in the AVX-512
kernel. With typical input, where the low bytes average about half their
range, the AVX2 kernel is estimated to go wrong from about 2048 dimensions.
The scalar kernel adds in 32 bits and is correct. NEON has not been checked.
0.1.0 never runs fastscan above 944 dimensions, so this is not a bug there
today.

The kernels must widen their sums to 32 bits often enough that no 16-bit
lane can overflow, for example every 128 columns (1024 dimensions), before
fastscan is enabled at higher dimensions. Running sums carried from one page
to the next are then 32-bit in both table modes.

## Costs and open questions

- **Kernel entry point.** Kernels take a range of chunks and starting sums
  instead of a whole group. This adds one call per page boundary, which
  should be negligible next to the 48–256 chunks of a group. To be confirmed
  with a standalone micro-benchmark.
- **Two groups in flight.** The scan keeps running sums for up to two groups
  across a page boundary, plus its position in the stream.
- **Page addressing.** Group `g` no longer sits at `g × group size` in a
  page. A page needs its stream offset in the opaque area so a reader can
  find the first group boundary on it, for example when a split or vacuum
  reads a page out of order.
- **Inserts across a page boundary** write two pages, as described above.
- **Fastscan conversion, splits, and vacuum** move from page-local group
  indexing to the stream layout.
- **New on-disk format.** Indexes built by 0.1.0 need `REINDEX`.
- **Other limits on high dimensions** are separate from this proposal:
  centroid pages, the fast rotation, and accuracy of the 16-bit lookup table
  with many subquantizers.

## Validation

- Standalone first: compare contiguous and split groups on the same data for
  recall (identical) and kernel speed (no measurable difference expected).
- Rebuild real 100-, 384-, 768- and 1024-dimension datasets with 0.1.0 and
  with the packed layout. Compare index size, recall, QPS and tail latency in
  release builds, with warm and cold shared buffers.
- Add a 4096-dimension dataset once the reference can continue across pages.
- Check plans as well as results, so a fallback to sequential scan doesn't
  pass as a working index scan.
