# Fastscan Posting Page Design

VPSHUFB-based fast scan for RaBitQ posting lists. Replaces the
per-entry 1-bit masked float-add inner product kernel with a
VPSHUFB table-lookup kernel that processes 32 vectors per
instruction pair.

## Background

RaBitQ encodes each vector as 1-bit sign codes (dim/8 bytes) plus
three scalar factors (f_add, f_rescale, f_error). The inner product
between a transformed query and the binary codes is the core
operation in the posting scan hot loop.

**Current 1-bit kernel:** `mkt_rabitq_inner_product_multi` processes
vectors one at a time (or 4 at a time with SIMD), computing
`binary_ip = Σ transformed[d] where bit[d]=1` via masked float
adds. At dim=768, each vector requires 96 bytes of bit data and
~768 float operations.

**Fastscan kernel:** Groups 4 dimensions into a "subquantizer" (sq).
Each sq has 2^4=16 possible sign combinations. A 16-entry uint8
lookup table (LUT) per sq maps each combination to a quantized
partial distance. The 1-bit codes are repacked into 4-bit nibbles
(two sq per byte), and VPSHUFB looks up 32 partial distances in
one instruction. At dim=768: 192 subquantizers, 96 sq-pairs, one
VPSHUFB per sq-pair → 96 VPSHUFB instructions for 32 vectors.

**Prior results** (cohere-1M, nprobe=10):
- 1-bit kernel: 5296 QPS @ 0.799 recall
- Fastscan: 7789 QPS @ 0.791 recall (+47%)
- Inner product time: 136μs → 46μs (3× faster)

## Design

### Mixed-page approach

The scan path handles AoS and fastscan pages transparently within
the same cluster chain. Each page's opaque flags indicate its
format (`PRISM_POSTING_PAGE_FASTSCAN`). This enables:

1. **Bulk build** writes AoS pages (streaming, one entry at a time)
2. **Inserts** append to AoS pages (same path, no batching needed)
3. **Background conversion** repacks full AoS pages into fastscan
   format in-place (future maintenance operation)

This keeps the write path simple. The initial implementation
validates the performance gain using a build-time conversion: after
the standard AoS build completes, a post-processing pass converts
all posting pages to fastscan format. If the gain is confirmed, a
background conversion path can be added later.

A `fastscan` boolean reloption (default false) controls whether
the post-build conversion runs. The metadata page records the
format so the scan path can pre-allocate fastscan scratch buffers.

### Fastscan page layout

Fastscan pages use SoA layout organized into 32-vector group
sections. Each group section is self-contained: metadata arrays
followed by VPSHUFB-packed codes.

```
┌──────────────────────────────────────────┐
│ PageHeaderData                      24B  │
├──────────────────────────────────────────┤
│ [pt_centroid — FIRST page only]  3072B   │
├──────────────────────────────────────────┤
│ Group 0 section:                         │
│   ItemPointerData tids[32]       192B    │
│   float f_add[32]                128B    │
│   float f_rescale[32]            128B    │
│   float f_error[32]              128B    │
│   uint8_t codes[96 × 32]       3072B    │
│                            total 3648B   │
├──────────────────────────────────────────┤
│ Group 1 section:                 3648B   │
│   (same layout as Group 0)               │
├──────────────────────────────────────────┤
│ (unused gap)                             │
├──────────────────────────────────────────┤
│ PrismPostingPageOpaque               24B   │
└──────────────────────────────────────────┘
```

At dim=768 (nsq=192, nsq_pairs=96):
- Group section: 32×6 + 32×4×3 + 96×32 = 192+384+3072 = 3648B
- Overflow page: 2 groups = 64 vectors (usable 8144B, used 7296B)
- First page: 1 group = 32 vectors (usable 5072B, used 3648B)

The `PRISM_POSTING_PAGE_FASTSCAN` flag (0x0004, already reserved in
posting_page.h) distinguishes fastscan pages from AoS pages.
`max_entries` and `entry_count` in the opaque work as before.

### Why per-group sections (not all-meta-then-all-codes)

When the VPSHUFB kernel finishes a 32-vector group, the prune loop
immediately needs that group's f_add, f_rescale, f_error, and TID
arrays. Colocating them in one 3648B section keeps the working set
in L1/L2. Cross-group data is never needed simultaneously.

### LUT construction

For each cluster (once per query):

1. Compute global range from the transformed query vector:
   `global_min = Σ min(0, transformed[d])`,
   `global_max = Σ max(0, transformed[d])`
2. Scale: `scale = range / 255`, `inv_scale = 1 / scale`
3. For each sq (4 dims), build 16 entries via incremental sums
   from pre-scaled values. Entry c is the quantized partial sum
   for sign combination c. Bias-folded so each entry is 1-2 float
   adds + truncation.
4. After accumulation: `float_ip = accum * scale + nsq * bias`

### Accumulation kernel

The kernel processes one 32-vector group:

```
for each sq_pair (96 iterations at dim=768):
    load 32B packed codes
    split into low/high nibbles
    broadcast lut0, lut1 to both 128-bit lanes
    VPSHUFB: 32 partial distances for sq0
    VPSHUFB: 32 partial distances for sq1
    widen uint8 → uint16 via maddubs
    accumulate into uint16 registers
```

Output: 32 uint16 accumulated partial distances.

**uint16 overflow check:** Max accumulator value = nsq_pairs × 510
(two sq each contributing max 255). At dim=768: 96 × 510 = 48,960.
uint16 max = 65,535. Safe for dim ≤ 1280. Add a static assertion.

### AVX2 vs AVX-512

AVX2: processes one sq_pair per iteration with two 256-bit VPSHUFB.
AVX-512: packs both LUTs and both nibble sets into 512-bit registers,
doing both lookups in one 512-bit VPSHUFB. Similar throughput on Zen 4
(bottleneck is VPSHUFB throughput), may help on other architectures.

### Code packing (1-bit → nibble)

At build time, after RaBitQ encoding produces 1-bit codes for a
group of 32 vectors:

1. For each sq_pair (two subquantizers, 8 dims total):
2. For each of 32 vectors:
3. Extract 4 sign bits for sq0 → low nibble (bits 0-3)
4. Extract 4 sign bits for sq1 → high nibble (bits 4-7)
5. Pack into one byte: `out[v] = nibble0 | (nibble1 << 4)`

The packed codes for 32 vectors × 96 sq_pairs = 3072 bytes.

### Build integration

The build always writes standard AoS pages via the existing
streaming builder. When `fastscan=true`, a post-processing pass
converts all posting pages to fastscan format after the AoS build
completes:

1. For each cluster, walk the posting page chain
2. Read entries from AoS pages in order
3. Batch into 32-vector groups
4. Pack 1-bit codes into VPSHUFB nibble layout
5. Write fastscan group sections to new pages
6. Replace the cluster's posting chain with the new pages

This avoids changing the builder's streaming one-entry-at-a-time
flow. The conversion is O(N) over all posting entries and runs
once at build time.

Future optimization: a background worker that converts AoS pages
to fastscan incrementally, enabling inserts to append AoS pages
that are later converted during maintenance.

### Scan integration

The posting scan dispatches on the page flag:

```c
void prism_posting_scan_cluster(PrismPostingScan *scan, MktTopK *topk)
{
    if (scan->fastscan)
        prism_posting_scan_cluster_fastscan(scan, topk);
    else
        prism_posting_scan_cluster_aos(scan, topk);
}
```

The fastscan scan function:
1. Build LUT once per cluster from `qstate->transformed`
2. For each page, for each group on the page:
   a. Call `mkt_fastscan_accumulate()` on the group's codes
   b. De-quantize: `binary_ip = accum * scale + nsq * bias`
   c. Convert to distance: `dist = f_add + g_add - 2 * f_rescale * final_dot`
   d. Prune via f_error lower bounds
   e. Insert survivors into topk

### Page count comparison (dim=768)

| Vectors | AoS pages | Fastscan pages | Ratio |
|---------|-----------|----------------|-------|
| 100     | 2         | 3              | 1.50× |
| 500     | 8         | 9              | 1.12× |
| 1000    | 15        | 17             | 1.13× |
| 5000    | 72        | 79             | 1.10× |

~10% more pages due to 32-vector group granularity, but the kernel
processes ~8× more vectors per SIMD instruction.

### Metadata

`PrismMetaPage.flags` gains `MKT_META_FLAG_FASTSCAN` (0x02).
Old code ignores the flag and reads AoS pages normally (pages
without `PRISM_POSTING_PAGE_FASTSCAN` in their opaque are AoS).

### Implementation plan

1. Fastscan kernel (fastscan.h/c, AVX2/AVX-512) — pure SIMD math,
   no PG dependencies
2. Unit tests validating kernel correctness
3. Add fastscan page format to posting_page.h (accessors, capacity)
4. Add fastscan scan path to posting_scan.c (mixed-page dispatch)
5. Add post-build AoS→fastscan conversion pass
6. Wire reloption + metadata flag in PG code
7. Benchmark against AoS baseline and validate recall
