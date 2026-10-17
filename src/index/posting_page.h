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
 *     (pt_centroid — on first page only)
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

/* Page flags */
#define PRISM_POSTING_PAGE_FIRST	0x0001
#define PRISM_POSTING_PAGE_OVERFLOW 0x0002
#define PRISM_POSTING_PAGE_FASTSCAN 0x0004 /* reserved for phase 2 */
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
 * The page carries a chunk of a full-precision float reference instead of
 * entries: entry_count and max_entries are 0. A posting list whose encode
 * reference does not fit beside an entry on its head (prism_posting_spilled)
 * stores it as a chain of these: the head (FIRST | REFERENCE) holds floats
 * [0, PRISM_POSTING_REF_CHUNK_FLOATS), each following page (OVERFLOW |
 * REFERENCE) the next chunk, and the entries start on the page after the
 * last chunk. The metapage's global mean spills the same way (see meta.h).
 */
#define PRISM_POSTING_PAGE_REFERENCE 0x0020

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
	uint16_t	flags;		 /* FIRST | OVERFLOW | FASTSCAN | TOMBSTONED |
							  * DELETED */
	uint16_t page_id;		 /* PRISM_POSTING_PAGE_ID */
	uint16_t max_entries;	 /* capacity of this page */

	/*
	 * Overlay (8 bytes). On a live page these are the per-cluster head
	 * metadata; on a page flagged PRISM_POSTING_PAGE_DELETED they instead
	 * carry the deletion XID for the reclaim gate. The DELETED flag is the
	 * sole discriminator — always read live_count/tail_blkno only when it is
	 * clear and delete_xid only when it is set. A posting-list split retires
	 * the whole old chain, its FIRST head included, so a DELETED head does
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
 * The widest dimension whose posting-list head holds the float encode
 * reference (MAXALIGN(dim * sizeof(float)) bytes) and at least one entry on
 * one page. Up to here the layout is the inline one; past it the reference
 * spills into a chain of REFERENCE pages (prism_posting_spilled). This is
 * also the widest dimension the float-format centroid page (~2036) and the
 * metapage's inline mean (2032) would have allowed, so it pins the
 * on-disk layout every index built before spilling existed has.
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
 * Hard dimension ceiling for the index layout. Spilling removes the head
 * page as a bound; what remains is that an overflow page holds at least one
 * entry and a RaBitQ centroid entry fits a centroid page, both true far
 * past this. The ceiling is the fast-rotation bound instead: a dimension
 * above PRISM_INDEX_NARROW_MAX_DIM is admitted only when
 * vs_fast_rotate_supported() holds (see prism_index_dim_supported), and
 * the Hadamard transform's inline sign storage stops at
 * VS_FAST_ROTATE_MAX_DIM. Above the narrow threshold the dense fallback
 * would cost O(dim^2) memory per backend and an O(dim^3) build on first
 * use, which is a footgun rather than a feature.
 */
#define PRISM_INDEX_MAX_DIM 8192

static_assert(
		PRISM_INDEX_MAX_DIM <= VS_FAST_ROTATE_MAX_DIM,
		"every admitted dimension must rotate through the fast path");

/*
 * The one gate every build applies: an inline layout up to the narrow
 * threshold, and past it a spilled layout only where the fast rotation
 * applies (dim = 2^m * k with 2^m >= 4 and odd k <= 8: 2048, 3072, 4096,
 * 6144, 8192 pass; 3000 does not).
 */
static inline bool
prism_index_dim_supported(Dimension dim)
{
	if (dim == 0 || dim > PRISM_INDEX_MAX_DIM)
		return false;
	if (dim <= PRISM_INDEX_NARROW_MAX_DIM)
		return true;
	return vs_fast_rotate_supported(dim);
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

/* Space occupied by P^T * centroid on an inline first page (MAXALIGN'd) */
static inline uint32_t
prism_posting_pt_centroid_size(Dimension dim)
{
	return (uint32_t)MAXALIGN(dim * sizeof(float));
}

/*
 * True when the encode reference no longer fits beside one entry on the
 * head, so the list stores it as a REFERENCE chain. Equivalent to
 * dim > PRISM_INDEX_NARROW_MAX_DIM at 8-byte MAXALIGN; computed from the
 * layout so a different alignment cannot disagree with the pages.
 */
static inline bool
prism_posting_spilled(Dimension dim)
{
	return prism_posting_pt_centroid_size(dim) +
				   PRISM_POSTING_ENTRY_SIZE(dim) >
		   prism_posting_page_usable();
}

/* Floats one REFERENCE page carries: the whole usable area. */
#define PRISM_POSTING_REF_CHUNK_FLOATS                     \
	((BLCKSZ - PRISM_CEILING_ALIGN(SizeOfPageHeaderData) - \
	  sizeof(PrismPostingPageOpaque)) /                    \
	 sizeof(float))

/*
 * Pages that carry the encode reference: the head alone for an inline
 * layout, else the head plus its REFERENCE continuations. This is also the
 * head stride of the bulk-built head region (prism_leaf_head_blkno), so a
 * leaf's reference pages are contiguous with its head.
 */
static inline uint32_t
prism_posting_ref_pages(Dimension dim)
{
	if (!prism_posting_spilled(dim))
		return 1;
	return ((uint32_t)dim + PRISM_POSTING_REF_CHUNK_FLOATS - 1) /
		   PRISM_POSTING_REF_CHUNK_FLOATS;
}

/* Floats held by reference page `chunk` (0 = head) of a spilled list. */
static inline uint32_t
prism_posting_ref_chunk_floats(Dimension dim, uint32_t chunk)
{
	uint32_t start = chunk * (uint32_t)PRISM_POSTING_REF_CHUNK_FLOATS;

	if (start >= dim)
		return 0;
	return (dim - start) < PRISM_POSTING_REF_CHUNK_FLOATS
				 ? (dim - start)
				 : (uint32_t)PRISM_POSTING_REF_CHUNK_FLOATS;
}

/*
 * Head block of a leaf, and back. Heads sit in a region at first_posting
 * with a fixed stride: 1 for an inline layout, prism_posting_ref_pages(dim)
 * for a spilled one, so the reference pages follow their head. The stride
 * is passed rather than derived from dim because the standalone in-RAM
 * build addresses leaves by index (first_posting 0, stride 1).
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

/* Max entries on an overflow page (no pt_centroid) */
static inline uint32_t
prism_posting_max_entries(Dimension dim)
{
	return prism_posting_page_usable() / PRISM_POSTING_ENTRY_SIZE(dim);
}

/* Max entries on a first page (with pt_centroid) */
static inline uint32_t
prism_posting_max_entries_first(Dimension dim)
{
	/* A spilled head carries only a reference chunk. Guarded rather than
	 * subtracted blindly: past the inline threshold the reference alone can
	 * exceed the page, and an unsigned subtraction would report a capacity
	 * of millions -- which the count checks use as their bound. */
	if (prism_posting_spilled(dim))
		return 0;

	return (prism_posting_page_usable() -
			prism_posting_pt_centroid_size(dim)) /
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

static inline bool
prism_posting_page_has_room(Page page)
{
	PrismPostingPageOpaque *op = prism_posting_opaque(page);
	return op->entry_count < op->max_entries;
}

/*
 * Per-cluster head metadata (read from the FIRST page). tail_blkno ==
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
 * before the entries. On a REFERENCE page it is that page's chunk of the
 * reference; prism_posting_reference_read gathers the whole vector.
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
 * Entry content of a first page. A REFERENCE head has no entries, so its
 * content is the (empty) start of the page rather than a pointer formed
 * past its end.
 */
static inline char *
prism_posting_content_first(Page page, Dimension dim)
{
	if (prism_posting_opaque(page)->flags & PRISM_POSTING_PAGE_REFERENCE)
		return PageGetContents(page);
	return PageGetContents(page) + prism_posting_pt_centroid_size(dim);
}

/*
 * Gather a posting list's encode reference, inline or spilled, into `out`
 * (dim floats). Reads the head and any REFERENCE continuations one page at
 * a time, releasing each before the next (the PG storage pins a single
 * buffer). Returns the first page that holds entries: the head itself for
 * an inline layout, the page after the last chunk for a spilled one, or
 * InvalidBlockNumber when the chain ends with the reference. Errors out on
 * a chain that is not what the layout says it must be.
 */
BlockNumber prism_posting_reference_read(
		VsStorage *storage, BlockNumber head, Dimension dim, float *out);

/*
 * The same gather for a caller that already copied the head's chunk (chunk
 * 0) while it had the head pinned: continues at `next`, the head's
 * next_blkno, with chunk 1. Inline layouts have nothing left to read and
 * get `next` back unchanged.
 */
BlockNumber prism_posting_reference_read_rest(
		VsStorage  *storage,
		BlockNumber head,
		BlockNumber next,
		uint32_t	cluster_id,
		Dimension	dim,
		float	   *out);

/*
 * Lay out reference page `chunk` of a spilled list in `page`: init as a
 * REFERENCE page (FIRST for chunk 0, OVERFLOW otherwise, plus `extra_flags`)
 * and copy its chunk of `ref`. next_blkno is left Invalid for the caller to
 * link.
 */
void prism_posting_reference_page_fill(
		Page		 page,
		uint32_t	 cluster_id,
		Dimension	 dim,
		uint32_t	 chunk,
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
 * Content of any posting page, first or not.
 *
 * A first page carries pt_centroid ahead of its entries, so the offset
 * differs; every caller that walks a chain meets both kinds and was
 * repeating this test. Inline, so the scan pays nothing for it.
 */
static inline char *
prism_posting_page_content(Page page, Dimension dim)
{
	uint16_t flags = prism_posting_opaque(page)->flags;

	if (flags & PRISM_POSTING_PAGE_REFERENCE)
		return prism_posting_content(page);
	return (flags & PRISM_POSTING_PAGE_FIRST)
				 ? prism_posting_content_first(page, dim)
				 : prism_posting_content(page);
}

/* Paged-mode convenience wrappers — these assume non-first pages
 * (content starts right after PageHeader). For first pages with
 * pt_centroid, go through prism_posting_content_first. */
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
 * SoA layout organized into 32-vector group sections.
 * Each group section:
 *   ItemPointerData tids[32]         192B
 *   float f_add[32]                  128B
 *   float f_rescale[32]              128B
 *   float f_error[32]                128B
 *   uint8_t codes[nsq_pairs × 32]   variable (3072B at dim=768)
 *   uint32_t tombstone_mask            4B   (bit v set => lane v is dead)
 *
 * tombstone_mask is the per-lane analog of an AoS entry's
 * PRISM_POSTING_FLAG_DELETED (see fastscan-design.md for why the codes
 * themselves can't be marked in place). A page starts with every mask word
 * zero (PageInit zeroes the page; a fresh group's tail is never written
 * otherwise), so an unmarked lane reads as live.
 *
 * Pages are identified by PRISM_POSTING_PAGE_FASTSCAN in opaque flags.
 * ---------------------------------------------------------------- */

#include "quant/fastscan.h"

static_assert(
		VS_FASTSCAN_GROUP == 32,
		"tombstone_mask packs one bit per lane into a uint32_t");

/* Bytes per 32-vector group section (metadata + packed codes + tombstone
 * mask) */
static inline uint32_t
prism_fastscan_group_section_bytes(Dimension dim)
{
	return (uint32_t)(VS_FASTSCAN_GROUP * sizeof(ItemPointerData) +
					  VS_FASTSCAN_GROUP * 3 * sizeof(float) +
					  VS_FASTSCAN_GROUP_BYTES(dim) + sizeof(uint32_t));
}

/*
 * Groups a fastscan page holds. A spilled head holds none (it carries a
 * reference chunk); an inline first page gives up the reference's room.
 * Guarded, not subtracted blindly: past the inline threshold the reference
 * exceeds the page and an unsigned subtraction would wrap to a capacity of
 * millions and let fastscan=auto select a layout no page can hold.
 */
static inline uint32_t
prism_fastscan_max_groups(Dimension dim, bool is_first)
{
	uint32_t section = prism_fastscan_group_section_bytes(dim);
	uint32_t usable	 = prism_posting_page_usable();

	if (is_first)
	{
		if (prism_posting_spilled(dim))
			return 0;
		usable -= prism_posting_pt_centroid_size(dim);
	}
	return usable / section;
}

/* Max entries on a fastscan overflow page */
static inline uint32_t
prism_fastscan_max_entries(Dimension dim)
{
	return prism_fastscan_max_groups(dim, false) * VS_FASTSCAN_GROUP;
}

/* Max entries on a fastscan first page (with pt_centroid) */
static inline uint32_t
prism_fastscan_max_entries_first(Dimension dim)
{
	return prism_fastscan_max_groups(dim, true) * VS_FASTSCAN_GROUP;
}

/*
 * Upper bound on the entries a single posting page can hold, whatever format
 * it is in. Neither format's figure bounds the other: fastscan packs more
 * than AoS at low dimension (416 vs 339 at dim 4) and fewer at high (64 vs 67
 * at dim 768, and none at all once a group stops fitting). The overflow-page
 * figures bound a first page too, since that gives up room to the encode
 * reference. Callers sizing a buffer for one page's worth of entries want
 * this rather than either half.
 */
static inline uint32_t
prism_posting_max_entries_any_format(Dimension dim)
{
	uint32_t aos = prism_posting_max_entries(dim);
	uint32_t fs	 = prism_fastscan_max_entries(dim);
	return aos > fs ? aos : fs;
}

/*
 * The entry ceiling a page of this format and position really has. All four
 * combinations differ: a fastscan page packs more entries than an AoS one,
 * and a first page gives up room to the encode reference. Bounding an AoS
 * page by the fastscan figure (or by the larger of the two) would still let
 * prism_posting_entry_at walk off the page.
 */
static inline uint32_t
prism_posting_page_cap(const PrismPostingPageOpaque *op, Dimension dim)
{
	bool first = (op->flags & PRISM_POSTING_PAGE_FIRST) != 0;

	if (op->flags & PRISM_POSTING_PAGE_REFERENCE)
		return 0;
	if (op->flags & PRISM_POSTING_PAGE_FASTSCAN)
		return first ? prism_fastscan_max_entries_first(dim)
					 : prism_fastscan_max_entries(dim);

	return first ? prism_posting_max_entries_first(dim)
				 : prism_posting_max_entries(dim);
}

/*
 * Reject an on-disk entry_count past that ceiling before it drives a loop, so
 * a corrupt or truncated page cannot read past the page end. The bound the
 * scan path already applies (posting_scan.c), in the form the shared read
 * paths need.
 */
void prism_posting_check_count(
		BlockNumber blkno, const PrismPostingPageOpaque *op, Dimension dim);

/* ----------------------------------------------------------------
 * Fastscan group accessors
 *
 * All take content pointer (past PageHeader, past pt_centroid on
 * first pages) and group index g.
 * ---------------------------------------------------------------- */

static inline char *
prism_fastscan_group_base(char *content, uint32_t g, Dimension dim)
{
	return content + (size_t)g * prism_fastscan_group_section_bytes(dim);
}

static inline ItemPointerData *
prism_fastscan_group_tids(char *content, uint32_t g, Dimension dim)
{
	return (ItemPointerData *)prism_fastscan_group_base(content, g, dim);
}

static inline float *
prism_fastscan_group_f_add(char *content, uint32_t g, Dimension dim)
{
	return (float *)(prism_fastscan_group_base(content, g, dim) +
					 VS_FASTSCAN_GROUP * sizeof(ItemPointerData));
}

static inline float *
prism_fastscan_group_f_rescale(char *content, uint32_t g, Dimension dim)
{
	return prism_fastscan_group_f_add(content, g, dim) + VS_FASTSCAN_GROUP;
}

static inline float *
prism_fastscan_group_f_error(char *content, uint32_t g, Dimension dim)
{
	return prism_fastscan_group_f_rescale(content, g, dim) + VS_FASTSCAN_GROUP;
}

static inline uint8_t *
prism_fastscan_group_codes(char *content, uint32_t g, Dimension dim)
{
	return (uint8_t *)(prism_fastscan_group_f_error(content, g, dim) +
					   VS_FASTSCAN_GROUP);
}

/*
 * Per-group dead-lane bitmap: bit v set means lane v's TID is dead. Read and
 * write through this pointer directly (it aliases the live page); there is no
 * copy to write back. Only valid lanes (v < the group's entry count) are ever
 * set -- PageInit zeroes a fresh page and nothing else touches this word, so
 * an unmarked lane, including every lane past entry_count on a partial final
 * group, reads as live.
 */
static inline uint32_t *
prism_fastscan_group_tombstone_mask(char *content, uint32_t g, Dimension dim)
{
	return (uint32_t *)(prism_fastscan_group_codes(content, g, dim) +
						VS_FASTSCAN_GROUP_BYTES(dim));
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
