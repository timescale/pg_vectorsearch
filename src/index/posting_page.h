/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * posting_page.h - Posting list page format
 *
 * Stores per-cluster posting data using an AoS layout: each entry
 * is a fixed-size block packing meta + factors + bits together.
 *
 *   Paged mode (BLCKSZ pages):
 *     PageHeader (PG-compatible, 24B)
 *     (reference prefix — on a list's first entry page only)
 *     entry[0], entry[1], ...
 *     PrismPostingPageOpaque (16B at page end)
 *
 *   Flat mode (one buffer per cluster):
 *     PrismFlatPostingHeader (16B)
 *     entry[0], entry[1], ...
 *
 * Per-entry block (size = PRISM_POSTING_ENTRY_SIZE(dim) bytes):
 *     PrismPostingEntryHeader  20B  (tid + flags + f_add/rescale/error)
 *     uint8_t bits[packed_bytes]  (dim-dependent, 96B at dim=768)
 *
 * Why AoS: when the SIMD kernel strides through entries at
 * stride=PRISM_POSTING_ENTRY_SIZE(dim), consecutive pages' bit
 * regions are separated by only ~72 bytes of page-boundary overhead
 * (opaque + next PageHeader) instead of the ~1472 bytes that an
 * SoA layout interleaves (metas + f_add + f_rescale + f_error of
 * the next page before its bits). A small per-page perturbation is
 * much easier for the HW prefetcher to absorb than a multi-KB
 * discontinuity — see the BLCKSZ experiment that confirmed the gap
 * lives at those region transitions.
 *
 * Side benefit: reading bits at stride=entry_size also pulls each
 * entry's f_add/f_rescale/f_error/meta into L1 for free, so phase 3
 * (distance conversion + prune) finds them hot.
 */

#ifndef PRISM_POSTING_PAGE_H
#define PRISM_POSTING_PAGE_H

#include <assert.h> /* static_assert pre-C23 (e.g. gcc 11's -std=c2x) */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "core/types.h"
#include "index/storage.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------- */

#define PRISM_POSTING_PAGE_ID 0x4D50 /* "MP" */

/* Entry flags */
#define PRISM_POSTING_FLAG_DELETED	0x01
#define PRISM_POSTING_FLAG_BOUNDARY 0x02

/*
 * Page flags.
 *
 * The low two bits are a field, not two flags: the page's part of its
 * list's head. A list's head is the pages that carry its encode reference
 * (P^T * centroid, 4 * dim bytes, or 2 * dim as float16). An inline list's
 * head is its first page, which holds the whole reference followed by
 * entries. A spilled list (prism_posting_spilled, past 1968 dimensions, or
 * 3820 as float16) spreads the reference over
 * prism_posting_reference_pages(dim, ref_format) pages, which the bulk build
 * lays out contiguously after the first:
 *
 *   head part   inline list                    spilled list
 *   START       whole reference, then entries  a page of reference only
 *   MIDDLE      (none)                         a page of reference only
 *   END         (none)                         rest of reference, entries
 *   NONE        entries                        entries
 *
 * START means the same on every list: the list's first page, which carries
 * the head metadata in the opaque and is what a leaf centroid points at.
 * Whether it also holds entries follows from the dimension. The values share
 * bits (END includes START's), so read the field with
 * prism_posting_head_part and compare it; never test one of its bits alone.
 */
#define PRISM_POSTING_PAGE_HEAD_MASK 0x0003
#define PRISM_POSTING_HEAD_NONE		 0x0000
#define PRISM_POSTING_HEAD_START	 0x0001
#define PRISM_POSTING_HEAD_MIDDLE	 0x0002
#define PRISM_POSTING_HEAD_END		 0x0003

/* The page's entries are in the fastscan format, not AoS. */
#define PRISM_POSTING_PAGE_FASTSCAN 0x0004
/*
 * Every entry on the page is dead (AoS: all entries flagged; FASTSCAN: every
 * lane of every group's tombstone_mask set). Both formats mark entries
 * individually; this flag is the scan-side shortcut that lets a fully dead
 * page skip its scoring kernel outright. The page stays linked so compaction
 * can later reclaim it. Cleared if the page is ever reused for new entries.
 */
#define PRISM_POSTING_PAGE_TOMBSTONED 0x0008
/*
 * The page belongs to a chain that has been logically retired — no longer
 * reachable through the centroid tree — and is awaiting physical reclaim. A
 * posting-list split sets this on every page of the old chain (head included)
 * once it has repointed the leaf at the new heads.
 *
 * Distinct from TOMBSTONED: a tombstoned page is all-dead but still a live
 * part of its cluster, so scans skip it and follow the chain past it. A
 * DELETED chain is off the tree but stays physically LINKED and readable until
 * reclaim, so an in-flight scanner that already followed a stale leaf pointer
 * into it still sees a consistent list rather than a gap.
 *
 * When this flag is set on a head, the head-metadata overlay in the opaque
 * (see the union below) holds the deletion XID instead of
 * live_count/tail_blkno. The reclaim gate compares that XID against the global
 * visibility horizon before physically reclaiming the chain, so a scanner that
 * still holds a stale pointer can never have it repurposed underneath it.
 * Unlike TOMBSTONED, this flag therefore does apply to chain heads: a split
 * retires the whole old chain, head and all.
 */
#define PRISM_POSTING_PAGE_DELETED 0x0010

/*
 * The list stores its encode reference as float16 instead of float32
 * (reloption posting_reference_format), half the bytes. Set on every head page
 * of such a list, so a page alone says how wide its reference prefix is.
 * Pages past the head hold no reference and do not carry it. The index-wide
 * layout (whether lists spill, how many head pages they have) follows the
 * metapage's PRISM_META_FLAG_REF16.
 */
#define PRISM_POSTING_PAGE_REF16 0x0020

/*
 * Element type of a posting list's encode reference (reloption
 * posting_reference_format). The global mean is always float32.
 */
typedef enum PrismPostingReferenceFormat
{
	PRISM_POSTING_REFERENCE_FLOAT32 = 0,
	PRISM_POSTING_REFERENCE_FLOAT16,
} PrismPostingReferenceFormat;

static const char *const prism_posting_reference_format_names[] = {
		[PRISM_POSTING_REFERENCE_FLOAT32] = "float32",
		[PRISM_POSTING_REFERENCE_FLOAT16] = "float16",
};

/* The reference format's name, as the reloption spells it. */
static inline const char *
prism_posting_reference_format_name(PrismPostingReferenceFormat ref_format)
{
	return prism_posting_reference_format_names[ref_format];
}

/* ----------------------------------------------------------------
 * Structs
 * ---------------------------------------------------------------- */

/*
 * Per-entry metadata.
 *
 * In PG: tid is a heap TID for reranking via buffer cache.
 * In standalone: vector_id packed into tid via ItemPointerSet.
 */
typedef struct PrismPostingEntryMeta
{
	ItemPointerData tid; /* 6B */
	uint8_t			flags;
	uint8_t			reserved;
} PrismPostingEntryMeta; /* 8B */

/*
 * Per-entry header in the AoS layout: meta + the three RaBitQ
 * scalar factors, followed by a flexible array of quantized bits
 * (VS_RABITQ_BYTES(dim) bytes at runtime).
 *
 * sizeof(PrismPostingEntryHeader) is 20B — the FAM doesn't contribute
 * to the struct's static size, so PRISM_POSTING_ENTRY_HEADER_SIZE
 * stays a clean compile-time constant. The bits array extends past
 * the struct in memory; callers compute per-entry offsets via
 * PRISM_POSTING_ENTRY_SIZE(dim) and access bits as `hdr->bits`.
 */
typedef struct PrismPostingEntryHeader
{
	PrismPostingEntryMeta meta;		 /* 8B */
	float				  f_add;	 /* 4B */
	float				  f_rescale; /* 4B */
	float				  f_error;	 /* 4B */
	uint8_t				  bits[FLEXIBLE_ARRAY_MEMBER];
} PrismPostingEntryHeader; /* 20B + dim-dependent bits */

/*
 * Paged-mode opaque (at end of every BLCKSZ posting page).
 */
typedef struct PrismPostingPageOpaque
{
	BlockNumber next_blkno; /* next page in chain */
	uint32_t	cluster_id;
	uint16_t	entry_count; /* entries on this page */
	uint16_t	flags;		 /* PRISM_POSTING_PAGE_* */
	uint16_t	page_id;	 /* PRISM_POSTING_PAGE_ID */

	/*
	 * An AoS page's capacity. A fastscan page takes no inserts, so it holds
	 * the length of its run of the list's code stream instead (see the
	 * fastscan page format below).
	 */
	union
	{
		uint16_t max_entries;
		uint16_t code_bytes;
	};

	/*
	 * Overlay (8 bytes). On a live page these are the per-cluster head
	 * metadata; on a page flagged PRISM_POSTING_PAGE_DELETED they instead
	 * carry the deletion XID for the reclaim gate. The DELETED flag is the
	 * sole discriminator — always read live_count/tail_blkno only when it is
	 * clear and delete_xid only when it is set. A posting-list split retires
	 * the whole old chain, its START page included, so a DELETED head does
	 * carry delete_xid here rather than head metadata; readers of head
	 * metadata (insert, split) first check the flag and treat a retired head
	 * as gone, so the two uses never collide. The anonymous struct/union keeps
	 * op->live_count, op->tail_blkno, and op->delete_xid all directly
	 * accessible. Storing the XID as a backend-neutral uint64_t (not PG's
	 * FullTransactionId) keeps this header usable by the standalone engine;
	 * the PG side converts via U64FromFullTransactionId / the inverse.
	 */
	union
	{
		struct
		{
			/*
			 * Lets the runtime insert path find the chain tail in O(1) and
			 * track per-cluster live size (for LIRE split/merge in a later
			 * phase). tail_blkno == InvalidBlockNumber means "not yet
			 * computed": the first insert walks the chain to fill both fields,
			 * then maintains them incrementally. Left zero / Invalid on
			 * overflow (non-first) pages.
			 */
			uint32_t	live_count;
			BlockNumber tail_blkno;
		};

		/* Valid only when PRISM_POSTING_PAGE_DELETED is set (see flag
		 * comment).
		 */
		uint64_t delete_xid;
	};
} PrismPostingPageOpaque; /* 24B */

/*
 * True when this page's entries are gone for good, as opposed to moved.
 *
 * TOMBSTONED means "nothing here worth scoring" and the scan skips the page.
 * The reclaim pass sets the same bit on a retired chain, whose entries are not
 * dead at all -- they were rewritten into the chain's replacements -- so
 * skipping there drops results that a stale-pointer scan is entitled to see.
 * DELETED separates the two: it is set only by the retire pass, so a page
 * carrying both moved rather than died, and its contents are still intact and
 * linked until #224 makes the pages reusable.
 */
static inline bool
prism_posting_page_all_dead(const PrismPostingPageOpaque *op)
{
	return (op->flags & PRISM_POSTING_PAGE_TOMBSTONED) != 0 &&
		   (op->flags & PRISM_POSTING_PAGE_DELETED) == 0;
}

/*
 * Flat-mode header (at start of flat page buffer).
 * No PG PageHeaderData — avoids the uint16_t page size limit.
 */
typedef struct PrismFlatPostingHeader
{
	uint32_t max_entries; /* == capacity (sized for count) */
	uint32_t entry_count;
	uint32_t cluster_id;
	uint32_t _pad;
} PrismFlatPostingHeader; /* 16B */

/* ----------------------------------------------------------------
 * Size calculations
 * ---------------------------------------------------------------- */

/* Bytes in the per-entry header (meta + 3 floats) */
#define PRISM_POSTING_ENTRY_HEADER_SIZE \
	sizeof(PrismPostingEntryHeader) /* 20B */

/* Byte offset of bits[] within one entry. Equal to the header size
 * because bits[] is the FAM right after the header. */
#define PRISM_POSTING_ENTRY_BITS_OFFSET offsetof(PrismPostingEntryHeader, bits)

/* Bytes per entry in bits region */
#define PRISM_POSTING_BITS_PER_ENTRY(dim) VS_RABITQ_BYTES(dim)

/* Entry stride must be a multiple of PrismPostingEntryHeader's alignment
 * (4 bytes — the float fields) so entry i's float members land on
 * properly aligned addresses regardless of `i`. For dim values where
 * VS_RABITQ_BYTES(dim) isn't already a multiple of 4 (i.e., dim not
 * a multiple of 32, such as dim=16 or dim=100), we pad up. */
#define PRISM_POSTING_ENTRY_ALIGN 4u

/* Total bytes per entry: header + bits, padded up to entry alignment
 * (= 116 at dim=768; 24 at dim=16). */
#define PRISM_POSTING_ENTRY_SIZE(dim)                                         \
	(((PRISM_POSTING_ENTRY_HEADER_SIZE + PRISM_POSTING_BITS_PER_ENTRY(dim)) + \
	  PRISM_POSTING_ENTRY_ALIGN - 1u) &                                       \
	 ~(PRISM_POSTING_ENTRY_ALIGN - 1u))

/*
 * A posting list's first page holds a full-precision encode reference (a
 * vector of 32-bit floats). The reference fits into the first page, together
 * with at least one entry, up to dimension PRISM_INDEX_NARROW_MAX_DIM, or
 * up to 3820 dimensions as float16 (PRISM_POSTING_PAGE_REF16).
 * Vectors of higher dimensions spill over to the following page(s) in a
 * chain of reference pages (prism_posting_spilled).
 *
 * Derived at 8-byte alignment, the largest MAXALIGN PostgreSQL uses, so it
 * is one constant on every platform; where MAXALIGN is smaller the head has
 * a few bytes of slack and the threshold is merely conservative. The static
 * asserts keep the number honest against layout changes.
 */
#define PRISM_INDEX_NARROW_MAX_DIM 1968

#define PRISM_CEILING_ALIGN(len) (((size_t)(len) + 7) & ~(size_t)7)

static_assert(
		BLCKSZ - PRISM_CEILING_ALIGN(SizeOfPageHeaderData) -
						sizeof(PrismPostingPageOpaque) -
						PRISM_CEILING_ALIGN(
								PRISM_INDEX_NARROW_MAX_DIM * sizeof(float)) >=
				PRISM_POSTING_ENTRY_SIZE(PRISM_INDEX_NARROW_MAX_DIM),
		"an inline head must fit the encode reference plus one entry");
static_assert(
		BLCKSZ - PRISM_CEILING_ALIGN(SizeOfPageHeaderData) -
						sizeof(PrismPostingPageOpaque) -
						PRISM_CEILING_ALIGN(
								(PRISM_INDEX_NARROW_MAX_DIM + 1) *
								sizeof(float)) <
				PRISM_POSTING_ENTRY_SIZE(PRISM_INDEX_NARROW_MAX_DIM + 1),
		"the inline threshold is tight: one dimension more must not fit");

/*
 * Hard dimension ceiling for the index layout, which is imposed by the
 * Hadamard fast-rotation bound.
 *
 * A dimension above PRISM_INDEX_NARROW_MAX_DIM is admitted only when
 * vs_fast_rotate_supported() holds (see prism_index_dim_supported), and
 * the Hadamard transform's inline sign storage stops at
 * VS_FAST_ROTATE_MAX_DIM.
 */
#define PRISM_INDEX_MAX_DIM 8192

static_assert(
		PRISM_INDEX_MAX_DIM <= VS_FAST_ROTATE_MAX_DIM,
		"every admitted dimension must rotate through the fast path");

/*
 * The one gate every build applies: an inline layout up to the narrow
 * threshold, and past it a spilled layout only where the fast rotation
 * applies (dim = 2^m * k with 2^m >= 4 and odd k <= 8: 2048, 3072, 4096,
 * 6144, 8192 pass; 3000 does not). Takes a uint32_t so a wider dimension
 * is range-checked before it narrows to a Dimension.
 */
static inline bool
prism_index_dim_supported(uint32_t dim)
{
	if (dim == 0 || dim > PRISM_INDEX_MAX_DIM)
		return false;
	if (dim <= PRISM_INDEX_NARROW_MAX_DIM)
		return true;
	return vs_fast_rotate_supported((Dimension)dim);
}

static_assert(
		BLCKSZ - PRISM_CEILING_ALIGN(SizeOfPageHeaderData) -
						sizeof(PrismPostingPageOpaque) >=
				PRISM_POSTING_ENTRY_SIZE(PRISM_INDEX_MAX_DIM),
		"an overflow page must hold one entry at the ceiling");

/* Usable space on a BLCKSZ page (between content start and opaque) */
static inline uint32_t
prism_posting_page_usable(void)
{
	return BLCKSZ - (uint32_t)MAXALIGN(SizeOfPageHeaderData) -
		   sizeof(PrismPostingPageOpaque);
}

/* Bytes of one reference element. */
static inline uint32_t
prism_posting_ref_elem_size(PrismPostingReferenceFormat ref_format)
{
	static const uint32_t sizes[] = {
			[PRISM_POSTING_REFERENCE_FLOAT32] = sizeof(float),
			[PRISM_POSTING_REFERENCE_FLOAT16] = sizeof(uint16_t),
	};

	return sizes[ref_format];
}

/* The reference format a head page's flags mark. */
static inline PrismPostingReferenceFormat
prism_posting_flags_ref_format(uint16_t flags)
{
	return (flags & PRISM_POSTING_PAGE_REF16) != 0
				 ? PRISM_POSTING_REFERENCE_FLOAT16
				 : PRISM_POSTING_REFERENCE_FLOAT32;
}

/* The head-page flag that marks a reference format. */
static inline uint16_t
prism_posting_ref_format_flag(PrismPostingReferenceFormat ref_format)
{
	return ref_format == PRISM_POSTING_REFERENCE_FLOAT16
				 ? PRISM_POSTING_PAGE_REF16
				 : 0;
}

/* Space occupied by P^T * centroid on an inline first page (MAXALIGN'd) */
static inline uint32_t
prism_posting_pt_centroid_size(
		Dimension dim, PrismPostingReferenceFormat ref_format)
{
	return (uint32_t)MAXALIGN(dim * prism_posting_ref_elem_size(ref_format));
}

/*
 * True when the encode reference no longer fits beside one entry on the
 * first page, so the list spreads it over several head pages. Equivalent to
 * dim > PRISM_INDEX_NARROW_MAX_DIM at 8-byte MAXALIGN; computed from the
 * layout so a different alignment cannot disagree with the pages.
 */
static inline bool
prism_posting_spilled(Dimension dim, PrismPostingReferenceFormat ref_format)
{
	return prism_posting_pt_centroid_size(dim, ref_format) +
				   PRISM_POSTING_ENTRY_SIZE(dim) >
		   prism_posting_page_usable();
}

/* Floats of the reference on a full head page: its whole usable area. */
#define PRISM_POSTING_REFERENCE_PAGE_FLOATS                \
	((BLCKSZ - PRISM_CEILING_ALIGN(SizeOfPageHeaderData) - \
	  sizeof(PrismPostingPageOpaque)) /                    \
	 sizeof(float))

/* Reference elements on a full head page: its whole usable area. */
static inline uint32_t
prism_posting_reference_page_capacity(PrismPostingReferenceFormat ref_format)
{
	return (uint32_t)PRISM_POSTING_REFERENCE_PAGE_FLOATS *
		   (uint32_t)sizeof(float) / prism_posting_ref_elem_size(ref_format);
}

/*
 * Pages that carry the encode reference, the list's head pages: one for an
 * inline layout, else START, MIDDLE and END. This is also the head stride of
 * the bulk-built head region (prism_leaf_head_blkno), so a leaf's head pages
 * are contiguous.
 *
 * A spilled list always has a START and an END page. A float16 reference
 * spills past 3820 dimensions but fits a page up to 4072, and then the END
 * page holds none of it, only entries.
 */
static inline uint32_t
prism_posting_reference_pages(
		Dimension dim, PrismPostingReferenceFormat ref_format)
{
	uint32_t cap = prism_posting_reference_page_capacity(ref_format);
	uint32_t n	 = ((uint32_t)dim + cap - 1) / cap;

	if (!prism_posting_spilled(dim, ref_format))
		return 1;
	return n < 2 ? 2 : n;
}

/*
 * Elements of the reference on head page i of a spilled list (0 = START).
 */
static inline uint32_t
prism_posting_reference_page_elems(
		Dimension dim, uint32_t i, PrismPostingReferenceFormat ref_format)
{
	uint32_t cap   = prism_posting_reference_page_capacity(ref_format);
	uint32_t start = i * cap;

	if (start >= dim)
		return 0;
	return (dim - start) < cap ? (dim - start) : cap;
}

/* The page's part of its list's head (see the page flags above). */
static inline uint16_t
prism_posting_head_part(uint16_t flags)
{
	return flags & PRISM_POSTING_PAGE_HEAD_MASK;
}

/* Whether the page is its list's first page, the one with head metadata. */
static inline bool
prism_posting_is_head_start(uint16_t flags)
{
	return prism_posting_head_part(flags) == PRISM_POSTING_HEAD_START;
}

/*
 * Whether the page holds part of the reference and no entries: a spilled
 * list's START page, or a MIDDLE page. Code that looks at a page on its
 * own uses this to tell it from a page with entries: its capacity is 0, the
 * builder writes no entries to it, and conversion and inspection count it
 * as reference. An END page also holds reference, but entries follow it.
 */
static inline bool
prism_posting_is_reference_only_page(uint16_t flags, Dimension dim)
{
	uint16_t part = prism_posting_head_part(flags);

	if (part == PRISM_POSTING_HEAD_MIDDLE)
		return true;
	return part == PRISM_POSTING_HEAD_START &&
		   prism_posting_spilled(dim, prism_posting_flags_ref_format(flags));
}

/*
 * The reference prefix: the part of the reference a list's first entry page
 * holds ahead of its entries. That is the whole reference on an inline
 * START page and the rest of it on an END page. Entries, AoS or fastscan,
 * start this many bytes into the page content, so the prefix sets where a
 * reader finds them and how many fit. MAXALIGN'd so the entries start
 * aligned.
 */
static inline uint32_t
prism_posting_ref_prefix_size(
		Dimension dim, PrismPostingReferenceFormat ref_format)
{
	if (!prism_posting_spilled(dim, ref_format))
		return prism_posting_pt_centroid_size(dim, ref_format);

	uint32_t last = prism_posting_reference_pages(dim, ref_format) - 1;

	return (uint32_t)MAXALIGN(
			prism_posting_reference_page_elems(dim, last, ref_format) *
			prism_posting_ref_elem_size(ref_format));
}

/*
 * Whether the page has a reference prefix, so its entries start after it:
 * an inline START page or an END page. Every other page that holds entries
 * has them from the start of its content.
 */
static inline bool
prism_posting_has_ref_prefix(uint16_t flags, Dimension dim)
{
	uint16_t part = prism_posting_head_part(flags);

	if (part == PRISM_POSTING_HEAD_END)
		return true;
	return part == PRISM_POSTING_HEAD_START &&
		   !prism_posting_spilled(dim, prism_posting_flags_ref_format(flags));
}

/*
 * Head block of a leaf, and back. Heads sit in a region at first_posting
 * with a fixed stride: 1 for an inline layout,
 * prism_posting_reference_pages(dim, ref_format) for a spilled one, so the
 * reference pages follow their head. The stride is passed rather than derived
 * from dim because the standalone in-RAM build addresses leaves by index
 * (first_posting 0, stride 1).
 */
static inline BlockNumber
prism_leaf_head_blkno(
		BlockNumber first_posting, uint32_t leaf, uint32_t stride)
{
	return first_posting + (BlockNumber)leaf * stride;
}

static inline uint32_t
prism_route_head_to_leaf(
		BlockNumber first_posting, BlockNumber head, uint32_t stride)
{
	return (uint32_t)(head - first_posting) / stride;
}

/* Max entries on a page with no reference ahead of its entries */
static inline uint32_t
prism_posting_max_entries(Dimension dim)
{
	return prism_posting_page_usable() / PRISM_POSTING_ENTRY_SIZE(dim);
}

/*
 * Max entries on a list's first entry page, after its reference prefix
 * (prism_posting_ref_prefix_size): an inline START page or an END page.
 */
static inline uint32_t
prism_posting_max_entries_first(
		Dimension dim, PrismPostingReferenceFormat ref_format)
{
	return (prism_posting_page_usable() -
			prism_posting_ref_prefix_size(dim, ref_format)) /
		   PRISM_POSTING_ENTRY_SIZE(dim);
}

/* Buffer size for a flat page with count entries */
static inline size_t
prism_posting_flat_page_size(Dimension dim, uint32_t count)
{
	return sizeof(PrismFlatPostingHeader) +
		   (size_t)count * PRISM_POSTING_ENTRY_SIZE(dim);
}

/* ----------------------------------------------------------------
 * Paged-mode access — opaque via PG PageGetSpecialPointer
 * ---------------------------------------------------------------- */

static inline PrismPostingPageOpaque *
prism_posting_opaque(Page page)
{
	return (PrismPostingPageOpaque *)PageGetSpecialPointer(page);
}

/*
 * Is this page a posting page at all (not a wrong-kind page, and not one
 * only extended but never initialized)? Guard on the special-area size
 * before reading the opaque, so a page of another kind is never misread
 * through the posting layout. MAXALIGN, not a bare sizeof: PageGetSpecialSize
 * reports the on-disk (MAXALIGN'd) special-area size PageInit reserved, and a
 * struct whose size isn't already a multiple of MAXIMUM_ALIGNOF would
 * otherwise silently stop matching the moment a field is added.
 */
static inline bool
prism_page_is_posting(Page page)
{
	return !PageIsNew(page) &&
		   PageGetSpecialSize(page) ==
				   MAXALIGN(sizeof(PrismPostingPageOpaque)) &&
		   prism_posting_opaque(page)->page_id == PRISM_POSTING_PAGE_ID;
}

static inline uint32_t
prism_posting_page_count(Page page)
{
	return prism_posting_opaque(page)->entry_count;
}

/* Room for one more AoS entry. A fastscan page never takes one. */
static inline bool
prism_posting_page_has_room(Page page)
{
	PrismPostingPageOpaque *op = prism_posting_opaque(page);
	return !(op->flags & PRISM_POSTING_PAGE_FASTSCAN) &&
		   op->entry_count < op->max_entries;
}

/*
 * Per-cluster head metadata (read from the START page). tail_blkno ==
 * InvalidBlockNumber means it has not been computed yet — see
 * prism_posting_insert_one, which fills it lazily on the first insert.
 */
static inline uint32_t
prism_posting_head_live_count(Page head)
{
	return prism_posting_opaque(head)->live_count;
}

static inline BlockNumber
prism_posting_head_tail(Page head)
{
	return prism_posting_opaque(head)->tail_blkno;
}

/* ----------------------------------------------------------------
 * Content-based AoS access — header-agnostic
 *
 * Entries are laid out contiguously: entry i starts at
 *   content + i * PRISM_POSTING_ENTRY_SIZE(dim)
 * and consists of an PrismPostingEntryHeader followed by
 * VS_RABITQ_BYTES(dim) bytes of bits. The SIMD kernel strides
 * through bits[] at stride = PRISM_POSTING_ENTRY_SIZE(dim), starting
 * from prism_posting_first_bits(content).
 * ---------------------------------------------------------------- */

/* Header for entry i (access fields via the struct: hdr->meta,
 * hdr->f_add, hdr->f_rescale, hdr->f_error). */
static inline PrismPostingEntryHeader *
prism_posting_entry_at(char *content, uint32_t i, Dimension dim)
{
	return (PrismPostingEntryHeader *)(content +
									   (size_t)i *
											   PRISM_POSTING_ENTRY_SIZE(dim));
}

/* Bits pointer for entry i (immediately follows entry i's header). */
static inline uint8_t *
prism_posting_entry_bits_at(
		char *content, uint32_t max_entries, Dimension dim, uint32_t i)
{
	(void)max_entries;
	return prism_posting_entry_at(content, i, dim)->bits;
}

/* Pointer to entry 0's bits — the base the SIMD kernel uses, with
 * stride = PRISM_POSTING_ENTRY_SIZE(dim). */
static inline uint8_t *
prism_posting_first_bits(char *content)
{
	return ((PrismPostingEntryHeader *)content)->bits;
}

/* ----------------------------------------------------------------
 * Paged-mode convenience wrappers
 *
 * These use PageGetContents for the content pointer and read
 * max_entries from the opaque. Used by page init/add and tests.
 * ---------------------------------------------------------------- */

static inline char *
prism_posting_content(Page page)
{
	return PageGetContents(page);
}

/*
 * On an inline first page, P^T * centroid is stored right after PageHeader,
 * before the entries. On a spilled list's head pages it is that page's
 * floats of the reference; prism_posting_reference_read gathers the whole
 * vector.
 */
static inline const float *
prism_posting_pt_centroid(Page page)
{
	return (const float *)PageGetContents(page);
}

static inline float *
prism_posting_pt_centroid_mut(Page page)
{
	return (float *)PageGetContents(page);
}

/*
 * Copy the first n elements of a head page's reference into out as floats,
 * widening a float16 reference (PRISM_POSTING_PAGE_REF16).
 */
void prism_posting_reference_load(Page page, uint32_t n, float *out);

/* Store n floats of ref as the page's reference, as float16 if it is one. */
void prism_posting_reference_store(Page page, const float *ref, uint32_t n);

/*
 * Round a reference to the float16 values a REF16 page stores, in place.
 * Errors out on an element float16 cannot represent (beyond ±65504).
 */
void prism_posting_reference_round16(float *ref, Dimension dim);

/*
 * Gather a posting list's encode reference, inline or spilled, into `out`
 * (dim floats). Reads the head pages one at a time, releasing each before
 * the next (the PG storage pins a single buffer). Returns the first page
 * that holds entries: the START page for an inline layout, the END page for
 * a spilled one. Errors out on a chain that is not what the layout says it
 * must be.
 */
BlockNumber prism_posting_reference_read(
		VsStorage				   *storage,
		BlockNumber					head,
		Dimension					dim,
		PrismPostingReferenceFormat ref_format,
		float					   *out);

/*
 * prism_posting_reference_read without the floats on the head's END page:
 * returns that page, for a caller that reads it for its entries anyway and
 * takes the floats from it with prism_posting_reference_read_end. Sets
 * *cluster_id to the list's. For an inline layout this is the whole gather.
 */
BlockNumber prism_posting_reference_read_to_end(
		VsStorage				   *storage,
		BlockNumber					head,
		Dimension					dim,
		PrismPostingReferenceFormat ref_format,
		float					   *out,
		uint32_t				   *cluster_id);

/*
 * Check that `page`, at blk, is the head END page of the list at head
 * with cluster_id, and copy its floats of the reference into out.
 */
void prism_posting_reference_read_end(
		BlockNumber					head,
		BlockNumber					blk,
		Page						page,
		uint32_t					cluster_id,
		Dimension					dim,
		PrismPostingReferenceFormat ref_format,
		float					   *out);

/*
 * The same gather for a caller that already copied the START page's floats
 * while it had the head pinned: continues at `next`, the head's next_blkno,
 * with head page 1. Inline layouts have nothing left to read and get `next`
 * back unchanged.
 */
BlockNumber prism_posting_reference_read_rest(
		VsStorage				   *storage,
		BlockNumber					head,
		BlockNumber					next,
		uint32_t					cluster_id,
		Dimension					dim,
		PrismPostingReferenceFormat ref_format,
		float					   *out);

/*
 * Lay out head page i of a spilled list in `page`: init it as the head's
 * START page for i = 0, its END page for the last, else a MIDDLE page, plus
 * `extra_flags`, and copy its floats of `ref`. An END page starts with no
 * entries. next_blkno is left Invalid for the caller to link.
 */
void prism_posting_reference_page_fill(
		Page		 page,
		uint32_t	 cluster_id,
		Dimension	 dim,
		uint32_t	 i,
		const float *ref,
		uint16_t	 extra_flags);

/* ----------------------------------------------------------------
 * Chain walks
 *
 * A posting list is a chain of pages linked by next_blkno. Every caller
 * that traverses one has to read the link before releasing the page and
 * release it before moving on; these keep that order in one place.
 *
 * The scan path deliberately does not use them. It holds a page across the
 * SIMD kernels that consume it, interleaves the next page's prefetch, and
 * handles flat mode (a single page, no chain) and page_base mode (memory it
 * does not release) -- so its stepping is a different shape, not a
 * duplicate of this one.
 * ---------------------------------------------------------------- */

/*
 * Where a read walk currently is. The callback may release the page early
 * and keep working: the next block is already read, so the walk does not
 * need it afterwards. That matters for work which must not hold a page --
 * fetching heap tuples through the same storage, which holds one page at a
 * time and would take over the slot.
 */
typedef struct PrismPostingChainPos
{
	VsStorage  *storage;
	BlockNumber blkno;
	Page		page; /* NULL once released */
	BlockNumber next; /* read before the callback runs */
	bool		first;
} PrismPostingChainPos;

static inline void
prism_posting_chain_release(PrismPostingChainPos *pos)
{
	if (pos->page != NULL)
	{
		vs_storage_release_page(pos->storage, pos->blkno);
		pos->page = NULL;
	}
}

/* Return false to stop the walk. */
typedef bool (*PrismPostingChainCb)(PrismPostingChainPos *pos, void *state);

void prism_posting_chain_walk(
		VsStorage		   *storage,
		BlockNumber			head,
		PrismPostingChainCb cb,
		void			   *state);

/*
 * Mutating walk: the callback gets the opaque of a page held for write, and
 * every page is committed. For chain-wide flag changes, where the body is a
 * line or two and the walk is all of the code.
 */
typedef void (*PrismPostingChainMutateCb)(
		PrismPostingPageOpaque *op, void *state);

void prism_posting_chain_mutate(
		VsStorage				 *storage,
		BlockNumber				  head,
		PrismPostingChainMutateCb cb,
		void					 *state);

/*
 * Entry content of any posting page: past the reference prefix on an inline
 * START page or an END page, else the start of the page. A page that holds
 * only reference has no entries, so its content is the (empty) start of the
 * page.
 */
static inline char *
prism_posting_page_content(Page page, Dimension dim)
{
	uint16_t flags = prism_posting_opaque(page)->flags;
	char	*start = prism_posting_content(page);

	return prism_posting_has_ref_prefix(flags, dim)
				 ? start + prism_posting_ref_prefix_size(
								   dim, prism_posting_flags_ref_format(flags))
				 : start;
}

/* Paged-mode convenience wrappers — these assume pages with no reference
 * prefix (content starts right after PageHeader). For the others, go
 * through prism_posting_page_content. */
static inline PrismPostingEntryHeader *
prism_posting_entry(Page page, uint32_t i, Dimension dim)
{
	return prism_posting_entry_at(PageGetContents(page), i, dim);
}

static inline uint8_t *
prism_posting_entry_bits(
		Page page, uint32_t max_entries, Dimension dim, uint32_t i)
{
	return prism_posting_entry_bits_at(
			PageGetContents(page), max_entries, dim, i);
}

/* ----------------------------------------------------------------
 * Flat-mode access
 * ---------------------------------------------------------------- */

static inline PrismFlatPostingHeader *
prism_flat_posting_header(char *buf)
{
	return (PrismFlatPostingHeader *)buf;
}

static inline char *
prism_flat_posting_content(char *buf)
{
	return buf + sizeof(PrismFlatPostingHeader);
}

/* ----------------------------------------------------------------
 * Standalone helper: pack/extract vector_id in ItemPointerData
 * ---------------------------------------------------------------- */

static inline void
prism_posting_set_vector_id(ItemPointerData *tid, uint32_t vector_id)
{
	ItemPointerSet(tid, (BlockNumber)vector_id, 0);
}

static inline uint32_t
prism_posting_get_vector_id(const ItemPointerData *tid)
{
	return (uint32_t)ItemPointerGetBlockNumber(tid);
}

/* ----------------------------------------------------------------
 * TID ↔ uint64_t encoding for VsTopK
 *
 * Both standalone and PG store TIDs in VsTopK's uint64_t id field.
 * For standalone: vector_id is packed in BlockNumber with offset=0,
 * so encode yields (vector_id << 16) and decode_vector_id shifts
 * back. For PG: full ItemPointerData (block + offset) fits in 48
 * bits of the uint64_t.
 * ---------------------------------------------------------------- */

static inline uint64_t
prism_posting_encode_tid(const ItemPointerData *tid)
{
	return ((uint64_t)ItemPointerGetBlockNumber(tid) << 16) |
		   (uint64_t)ItemPointerGetOffsetNumber(tid);
}

static inline ItemPointerData
prism_posting_decode_tid(uint64_t id)
{
	ItemPointerData tid;
	ItemPointerSet(&tid, (BlockNumber)(id >> 16), (OffsetNumber)(id & 0xFFFF));
	return tid;
}

/* Convenience for standalone: extract vector_id from encoded TID.
 * Works because prism_posting_set_vector_id stores vid in BlockNumber
 * with offset=0, so (id >> 16) == vid. */
static inline uint32_t
prism_posting_decode_vector_id(uint64_t id)
{
	return (uint32_t)(id >> 16);
}

/* ----------------------------------------------------------------
 * Paged-mode page operations
 * ---------------------------------------------------------------- */

/*
 * Initialize an empty BLCKSZ posting page.
 */
void prism_posting_page_init(
		Page page, uint32_t cluster_id, Dimension dim, uint16_t flags);

/*
 * Add one entry to a BLCKSZ page. Returns false if page is full.
 */
bool prism_posting_page_add(
		Page			page,
		Dimension		dim,
		ItemPointerData tid,
		float			f_add,
		float			f_rescale,
		float			f_error,
		const uint8_t  *bits,
		uint8_t			entry_flags);

/* ----------------------------------------------------------------
 * Flat-mode page operations
 * ---------------------------------------------------------------- */

/*
 * Initialize a flat posting page buffer. buf must be at least
 * prism_posting_flat_page_size(dim, max_entries) bytes.
 */
void
prism_posting_flat_init(char *buf, uint32_t max_entries, uint32_t cluster_id);

/*
 * Add one entry to a flat page. Returns false if full.
 */
bool prism_posting_flat_add(
		char		   *buf,
		Dimension		dim,
		ItemPointerData tid,
		float			f_add,
		float			f_rescale,
		float			f_error,
		const uint8_t  *bits,
		uint8_t			entry_flags);

/* ----------------------------------------------------------------
 * Fastscan page format
 *
 * Fastscan scores 32 vectors at once with in-register table lookups on
 * their 1-bit codes, which is much faster than scoring AoS entries one at a
 * time. A fastscan list stores its vectors in 32-vector groups. Each group
 * has codes, VS_FASTSCAN_GROUP_BYTES(dim) bytes interleaved for the kernel,
 * and 580 bytes of group metadata:
 *   ItemPointerData tids[32]         192B
 *   float f_add[32]                  128B
 *   float f_rescale[32]              128B
 *   float f_error[32]                128B
 *   uint32_t tombstone_mask            4B   (bit v set => lane v is dead)
 *
 * Page content (after the reference prefix on a list's first entry page):
 *   group metadata[ngroups]          ngroups × 580B
 *   codes[code_bytes]                the page's run of the code stream
 *
 * A spilled list's stream starts on its head's END page, after the rest of
 * the reference.
 *
 * The codes of all groups form one stream that continues from page to page.
 * A group's codes can break at any PRISM_FASTSCAN_CHUNK_BYTES (64-byte)
 * boundary, so a page wastes less than 64 bytes instead of the gap a whole
 * group does not fit. Every kernel steps in sizes that divide 64 bytes, so
 * the scan scores a split group in place, one page at a time.
 *
 * Group metadata never splits. It goes on the page where its group's codes
 * end, because that is where the scan has the complete sums to prune with,
 * and the scan holds one page at a time. When it does not fit there, it is
 * deferred to the start of the next page and the next group's codes use the
 * space.
 *
 * Metadata comes first on the page so readers that need only metadata
 * (deleted-vector marking, split, inspect) find group g's at a fixed offset,
 * without knowing where groups start in the code stream.
 *
 * In the opaque, entry_count counts the vectors whose metadata is on the
 * page and code_bytes is the length of its code run. Only the list's last
 * group can be partial, so a page holds ceil(entry_count / 32) groups'
 * metadata. A page's position in the stream is not stored: the scan reads a
 * list from its head and carries the position from page to page.
 *
 * tombstone_mask is the per-lane analog of an AoS entry's
 * PRISM_POSTING_FLAG_DELETED (see fastscan-design.md for why the codes
 * can't be marked in place). The builder writes it as zero.
 *
 * Pages are identified by PRISM_POSTING_PAGE_FASTSCAN in opaque flags.
 * ---------------------------------------------------------------- */

#include "quant/fastscan.h"

static_assert(
		VS_FASTSCAN_GROUP == 32,
		"tombstone_mask packs one bit per lane into a uint32_t");

/*
 * Code bytes per split unit: 16 dimensions of 32 vectors. The widest kernel
 * step (AVX-512, 64 bytes) and the 16-bit lookup table's 128-byte block of
 * two columns both line up with it, so a kernel call can start at any chunk.
 */
#define PRISM_FASTSCAN_CHUNK_BYTES 64u

/*
 * Most groups a scan can have in flight, meaning it has started summing
 * their codes but not yet reached their metadata. That is the group whose
 * codes it is reading plus up to two whose metadata was deferred to the next
 * page. The scan keeps running sums for each, so the bound keeps its state
 * fixed. Two deferred groups are enough to fill the space deferred metadata
 * leaves with codes at every dimension above about 72.
 */
#define PRISM_FASTSCAN_MAX_INFLIGHT_GROUPS 3u

/* Bytes of one group's metadata. */
#define PRISM_FASTSCAN_META_BYTES                             \
	((uint32_t)(VS_FASTSCAN_GROUP * sizeof(ItemPointerData) + \
				VS_FASTSCAN_GROUP * 3 * sizeof(float) + sizeof(uint32_t)))

static_assert(
		PRISM_FASTSCAN_META_BYTES % sizeof(float) == 0,
		"group metadata keeps the floats of the next group's aligned");

/* Bytes per 32-vector group, metadata and codes together */
static inline uint32_t
prism_fastscan_group_section_bytes(Dimension dim)
{
	return PRISM_FASTSCAN_META_BYTES + VS_FASTSCAN_GROUP_BYTES(dim);
}

/*
 * Bytes for metadata and codes on a fastscan page, after the reference
 * prefix when the page is a list's first entry page (ref_prefix), whose
 * width ref_format sets.
 */
static inline uint32_t
prism_fastscan_page_space(
		Dimension dim, bool ref_prefix, PrismPostingReferenceFormat ref_format)
{
	uint32_t usable = prism_posting_page_usable();

	return ref_prefix ? usable - prism_posting_ref_prefix_size(dim, ref_format)
					  : usable;
}

/* Groups with metadata on a fastscan page, from its vector count. */
static inline uint32_t
prism_fastscan_page_groups(const PrismPostingPageOpaque *op)
{
	return ((uint32_t)op->entry_count + VS_FASTSCAN_GROUP - 1) /
		   VS_FASTSCAN_GROUP;
}

/*
 * The max number of groups' metadata a page can hold. Besides the metadata
 * deferred from the previous page, a page holds metadata only for groups
 * whose codes end on it, and its code bytes reach the end of at most
 * space / group-code-size + 1 groups.
 */
static inline uint32_t
prism_fastscan_max_groups(
		Dimension dim, bool ref_prefix, PrismPostingReferenceFormat ref_format)
{
	uint32_t space	  = prism_fastscan_page_space(dim, ref_prefix, ref_format);
	uint32_t deferred = PRISM_FASTSCAN_MAX_INFLIGHT_GROUPS - 1;
	uint32_t by_meta  = space / PRISM_FASTSCAN_META_BYTES;
	/* An index's dimension is never 0, so a group has code bytes; the
	 * analyzer cannot see that from a scan's dim field. */
	/* NOLINTNEXTLINE(clang-analyzer-core.DivideZero) */
	uint32_t by_code = deferred + 1 + space / VS_FASTSCAN_GROUP_BYTES(dim);

	return by_meta < by_code ? by_meta : by_code;
}

/* Max entries on a fastscan page with no reference prefix */
static inline uint32_t
prism_fastscan_max_entries(Dimension dim)
{
	return prism_fastscan_max_groups(
				   dim, false, PRISM_POSTING_REFERENCE_FLOAT32) *
		   VS_FASTSCAN_GROUP;
}

/* Max entries on a list's first fastscan entry page, after the prefix */
static inline uint32_t
prism_fastscan_max_entries_first(
		Dimension dim, PrismPostingReferenceFormat ref_format)
{
	return prism_fastscan_max_groups(dim, true, ref_format) *
		   VS_FASTSCAN_GROUP;
}

/*
 * Entries a full page holds on average, for cost estimates. Fastscan pages
 * waste under a chunk each, so the code and metadata bytes per vector set
 * the density.
 */
static inline double
prism_fastscan_entries_per_page(Dimension dim)
{
	return (double)prism_fastscan_page_space(
				   dim, false, PRISM_POSTING_REFERENCE_FLOAT32) *
		   VS_FASTSCAN_GROUP / prism_fastscan_group_section_bytes(dim);
}

/*
 * Upper bound on the entries a single posting page can hold, whatever format
 * it is in. Neither format's figure bounds the other: fastscan packs more
 * than AoS at low dimension and allows for deferred group metadata, while
 * AoS packs more where a page holds under one group's codes. The figures
 * for a page without a reference prefix bound the others too. Callers sizing a
 * buffer for one page's worth of entries want this rather than either half.
 */
static inline uint32_t
prism_posting_max_entries_any_format(Dimension dim)
{
	uint32_t aos = prism_posting_max_entries(dim);
	uint32_t fs	 = prism_fastscan_max_entries(dim);
	return aos > fs ? aos : fs;
}

/* Entries a full page holds in the denser of the two formats. */
static inline double
prism_posting_entries_per_page(Dimension dim)
{
	double aos = (double)prism_posting_max_entries(dim);
	double fs  = prism_fastscan_entries_per_page(dim);
	return aos > fs ? aos : fs;
}

/*
 * The entry ceiling a page of this format and position really has. All four
 * combinations differ: a fastscan page packs more entries than an AoS one,
 * and a list's first entry page gives up room to the reference prefix.
 * Bounding an AoS page by the fastscan figure (or by the larger of the two)
 * would still let prism_posting_entry_at walk off the page.
 */
static inline uint32_t
prism_posting_page_cap(const PrismPostingPageOpaque *op, Dimension dim)
{
	bool first = prism_posting_has_ref_prefix(op->flags, dim);
	PrismPostingReferenceFormat ref_format = prism_posting_flags_ref_format(
			op->flags);

	if (prism_posting_is_reference_only_page(op->flags, dim))
		return 0;
	if (op->flags & PRISM_POSTING_PAGE_FASTSCAN)
		return first ? prism_fastscan_max_entries_first(dim, ref_format)
					 : prism_fastscan_max_entries(dim);

	return first ? prism_posting_max_entries_first(dim, ref_format)
				 : prism_posting_max_entries(dim);
}

/*
 * Reject an on-disk entry_count past that ceiling before it drives a loop, so
 * a corrupt or truncated page cannot read past the page end. A fastscan page
 * also has its group metadata and code run checked to fit.
 */
void prism_posting_check_count(BlockNumber blkno, Page page, Dimension dim);

/*
 * The fastscan half of prism_posting_check_count, against the entry ceiling
 * and content space of a page in this one's position. The scan passes bounds
 * it computed once, since it checks every page it reads.
 */
void prism_fastscan_check_page(
		BlockNumber					  blkno,
		const PrismPostingPageOpaque *op,
		uint32_t					  cap,
		uint32_t					  space);

/* ----------------------------------------------------------------
 * Fastscan page accessors
 *
 * All take the content pointer (past PageHeader, past pt_centroid on first
 * pages). The group index g counts the groups with metadata on that page.
 * ---------------------------------------------------------------- */

static inline char *
prism_fastscan_group_base(char *content, uint32_t g)
{
	return content + (size_t)g * PRISM_FASTSCAN_META_BYTES;
}

/* The page's run of the code stream, after the metadata of ngroups groups. */
static inline uint8_t *
prism_fastscan_page_codes(char *content, uint32_t ngroups)
{
	return (uint8_t *)content + (size_t)ngroups * PRISM_FASTSCAN_META_BYTES;
}

static inline ItemPointerData *
prism_fastscan_group_tids(char *content, uint32_t g)
{
	return (ItemPointerData *)prism_fastscan_group_base(content, g);
}

static inline float *
prism_fastscan_group_f_add(char *content, uint32_t g)
{
	return (float *)(prism_fastscan_group_base(content, g) +
					 VS_FASTSCAN_GROUP * sizeof(ItemPointerData));
}

static inline float *
prism_fastscan_group_f_rescale(char *content, uint32_t g)
{
	return prism_fastscan_group_f_add(content, g) + VS_FASTSCAN_GROUP;
}

static inline float *
prism_fastscan_group_f_error(char *content, uint32_t g)
{
	return prism_fastscan_group_f_rescale(content, g) + VS_FASTSCAN_GROUP;
}

/*
 * Per-group dead-lane bitmap: bit v set means lane v's TID is dead. Read and
 * write through this pointer directly (it aliases the live page); there is no
 * copy to write back. Only valid lanes (v < the group's entry count) are ever
 * set -- the builder writes the word as zero and nothing else touches it, so
 * an unmarked lane, including every lane past entry_count on a partial final
 * group, reads as live.
 */
static inline uint32_t *
prism_fastscan_group_tombstone_mask(char *content, uint32_t g)
{
	return (uint32_t *)(prism_fastscan_group_f_error(content, g) +
						VS_FASTSCAN_GROUP);
}

/*
 * Bit test and set for one lane of a tombstone_mask word already read out
 * of (or about to be written back into) a page via the accessor above.
 * Every reader and writer of the mask goes through these two rather than
 * writing out the bit math itself, so there is exactly one place that
 * says "lane v is bit 1u << v".
 */
static inline bool
prism_fastscan_lane_is_tombstoned(uint32_t mask, uint32_t lane)
{
	return (mask & (1u << lane)) != 0;
}

static inline uint32_t
prism_fastscan_lane_set_tombstone(uint32_t mask, uint32_t lane)
{
	return mask | (1u << lane);
}

#endif /* PRISM_POSTING_PAGE_H */
