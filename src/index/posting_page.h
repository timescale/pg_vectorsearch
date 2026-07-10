/*
 * posting_page.h - Posting list page format
 *
 * Stores per-cluster posting data using an AoS layout: each entry
 * is a fixed-size block packing meta + factors + bits together.
 *
 *   Paged mode (BLCKSZ pages):
 *     PageHeader (PG-compatible, 24B)
 *     (pt_centroid — on first page only)
 *     entry[0], entry[1], ...
 *     MktPostingPageOpaque (16B at page end)
 *
 *   Flat mode (one buffer per cluster):
 *     MktFlatPostingHeader (16B)
 *     entry[0], entry[1], ...
 *
 * Per-entry block (size = MKT_POSTING_ENTRY_SIZE(dim) bytes):
 *     MktPostingEntryHeader  20B  (tid + flags + f_add/rescale/error)
 *     uint8_t bits[packed_bytes]  (dim-dependent, 96B at dim=768)
 *
 * Why AoS: when the SIMD kernel strides through entries at
 * stride=MKT_POSTING_ENTRY_SIZE(dim), consecutive pages' bit
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

#ifndef MKT_POSTING_PAGE_H
#define MKT_POSTING_PAGE_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "index/storage.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------- */

#define MKT_POSTING_PAGE_ID 0x4D50 /* "MP" */

/* Entry flags */
#define MKT_POSTING_FLAG_DELETED  0x01
#define MKT_POSTING_FLAG_BOUNDARY 0x02

/* Page flags */
#define MKT_POSTING_PAGE_FIRST	  0x0001
#define MKT_POSTING_PAGE_OVERFLOW 0x0002
#define MKT_POSTING_PAGE_FASTSCAN 0x0004 /* reserved for phase 2 */
/*
 * Every entry on the page is dead. Set by the VACUUM tombstone pass when a
 * page's whole contents are deleted (AoS: all entries flagged; FASTSCAN: all
 * group TIDs dead — the only way a packed page's deletes are recorded, since
 * its entries can't be flagged individually). The scan skips the page's
 * scoring kernel entirely; the page stays linked so compaction can later
 * reclaim it. Cleared if the page is ever reused for new entries.
 */
#define MKT_POSTING_PAGE_TOMBSTONED 0x0008
/*
 * The page has been unlinked from its cluster chain and is awaiting physical
 * reclaim (future page-recycle work; not yet emitted by any code path).
 *
 * Distinct from TOMBSTONED: a tombstoned page is all-dead but still LINKED, so
 * scans skip it and follow the chain past it. A DELETED page has additionally
 * been spliced out (prev->next_blkno swung past it) and recorded in the FSM;
 * it is off the chain entirely.
 *
 * When this flag is set, the head-metadata overlay in the opaque (see the
 * union below) holds the deletion XID instead of live_count/tail_blkno. The
 * recycle gate compares that XID against the oldest snapshot before letting an
 * insert reinitialize the page for a new chain, so a scanner that still holds
 * a stale pointer to the page can never have it repurposed underneath it.
 * Only overflow pages are ever DELETED — chain heads are pinned by the
 * centroid tree and never unlinked.
 */
#define MKT_POSTING_PAGE_DELETED 0x0010

/* ----------------------------------------------------------------
 * Structs
 * ---------------------------------------------------------------- */

/*
 * Per-entry metadata.
 *
 * In PG: tid is a heap TID for reranking via buffer cache.
 * In standalone: vector_id packed into tid via ItemPointerSet.
 */
typedef struct MktPostingEntryMeta
{
	ItemPointerData tid; /* 6B */
	uint8_t			flags;
	uint8_t			reserved;
} MktPostingEntryMeta; /* 8B */

/*
 * Per-entry header in the AoS layout: meta + the three RaBitQ
 * scalar factors, followed by a flexible array of quantized bits
 * (MKT_RABITQ_BYTES(dim) bytes at runtime).
 *
 * sizeof(MktPostingEntryHeader) is 20B — the FAM doesn't contribute
 * to the struct's static size, so MKT_POSTING_ENTRY_HEADER_SIZE
 * stays a clean compile-time constant. The bits array extends past
 * the struct in memory; callers compute per-entry offsets via
 * MKT_POSTING_ENTRY_SIZE(dim) and access bits as `hdr->bits`.
 */
typedef struct MktPostingEntryHeader
{
	MktPostingEntryMeta meta;	   /* 8B */
	float				f_add;	   /* 4B */
	float				f_rescale; /* 4B */
	float				f_error;   /* 4B */
	uint8_t				bits[FLEXIBLE_ARRAY_MEMBER];
} MktPostingEntryHeader; /* 20B + dim-dependent bits */

/*
 * Paged-mode opaque (at end of every BLCKSZ posting page).
 */
typedef struct MktPostingPageOpaque
{
	BlockNumber next_blkno; /* next page in chain */
	uint32_t	cluster_id;
	uint16_t	entry_count; /* entries on this page */
	uint16_t	flags;		 /* FIRST | OVERFLOW | FASTSCAN | TOMBSTONED |
							  * DELETED */
	uint16_t page_id;		 /* MKT_POSTING_PAGE_ID */
	uint16_t max_entries;	 /* capacity of this page */

	/*
	 * Overlay (8 bytes). On a live page these are the per-cluster head
	 * metadata; on a page flagged MKT_POSTING_PAGE_DELETED they instead carry
	 * the deletion XID for the recycle gate. The two uses never collide: head
	 * metadata is meaningful only on FIRST pages, which are never DELETED
	 * (heads are pinned by the centroid tree), and the deletion XID is set
	 * only on unlinked overflow pages, which by definition hold no live
	 * entries — so live_count/tail_blkno are already dead weight there. The
	 * anonymous struct/union keeps op->live_count, op->tail_blkno, and
	 * op->delete_xid all directly accessible; gate the latter on the DELETED
	 * flag. Storing the XID as a backend-neutral uint64_t (not PG's
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

		/* Valid only when MKT_POSTING_PAGE_DELETED is set (see flag comment).
		 */
		uint64_t delete_xid;
	};
} MktPostingPageOpaque; /* 24B */

/*
 * Flat-mode header (at start of flat page buffer).
 * No PG PageHeaderData — avoids the uint16_t page size limit.
 */
typedef struct MktFlatPostingHeader
{
	uint32_t max_entries; /* == capacity (sized for count) */
	uint32_t entry_count;
	uint32_t cluster_id;
	uint32_t _pad;
} MktFlatPostingHeader; /* 16B */

/* ----------------------------------------------------------------
 * Size calculations
 * ---------------------------------------------------------------- */

/* Bytes in the per-entry header (meta + 3 floats) */
#define MKT_POSTING_ENTRY_HEADER_SIZE sizeof(MktPostingEntryHeader) /* 20B */

/* Byte offset of bits[] within one entry. Equal to the header size
 * because bits[] is the FAM right after the header. */
#define MKT_POSTING_ENTRY_BITS_OFFSET offsetof(MktPostingEntryHeader, bits)

/* Bytes per entry in bits region */
#define MKT_POSTING_BITS_PER_ENTRY(dim) MKT_RABITQ_BYTES(dim)

/* Entry stride must be a multiple of MktPostingEntryHeader's alignment
 * (4 bytes — the float fields) so entry i's float members land on
 * properly aligned addresses regardless of `i`. For dim values where
 * MKT_RABITQ_BYTES(dim) isn't already a multiple of 4 (i.e., dim not
 * a multiple of 32, such as dim=16 or dim=100), we pad up. */
#define MKT_POSTING_ENTRY_ALIGN 4u

/* Total bytes per entry: header + bits, padded up to entry alignment
 * (= 116 at dim=768; 24 at dim=16). */
#define MKT_POSTING_ENTRY_SIZE(dim)                                       \
	(((MKT_POSTING_ENTRY_HEADER_SIZE + MKT_POSTING_BITS_PER_ENTRY(dim)) + \
	  MKT_POSTING_ENTRY_ALIGN - 1u) &                                     \
	 ~(MKT_POSTING_ENTRY_ALIGN - 1u))

/*
 * Hard dimension ceiling for the index layout. A posting list's first page
 * carries the list's float encode reference (MAXALIGN(dim * sizeof(float))
 * bytes) and must still hold at least one posting entry; that binds before
 * the float-format centroid page (one dim * sizeof(float) entry, ~2036) and
 * the metadata page's inline mean. Past the ceiling the first-page capacity
 * arithmetic underflows, so index creation must reject the dimension up
 * front. The static assert keeps the number honest against layout changes.
 */
#define MKT_INDEX_MAX_DIM 1968

static_assert(
		BLCKSZ - MAXALIGN(SizeOfPageHeaderData) -
						sizeof(MktPostingPageOpaque) -
						MAXALIGN(MKT_INDEX_MAX_DIM * sizeof(float)) >=
				MKT_POSTING_ENTRY_SIZE(MKT_INDEX_MAX_DIM),
		"posting first page must fit the encode reference plus one entry");
static_assert(
		BLCKSZ - MAXALIGN(SizeOfPageHeaderData) -
						sizeof(MktPostingPageOpaque) -
						MAXALIGN((MKT_INDEX_MAX_DIM + 1) * sizeof(float)) <
				MKT_POSTING_ENTRY_SIZE(MKT_INDEX_MAX_DIM + 1),
		"the ceiling is tight: one dimension more must not fit");

/* Usable space on a BLCKSZ page (between content start and opaque) */
static inline uint32_t
mkt_posting_page_usable(void)
{
	return BLCKSZ - (uint32_t)MAXALIGN(SizeOfPageHeaderData) -
		   sizeof(MktPostingPageOpaque);
}

/* Space occupied by P^T * centroid on first pages (MAXALIGN'd) */
static inline uint32_t
mkt_posting_pt_centroid_size(Dimension dim)
{
	return (uint32_t)MAXALIGN(dim * sizeof(float));
}

/* Max entries on an overflow page (no pt_centroid) */
static inline uint32_t
mkt_posting_max_entries(Dimension dim)
{
	return mkt_posting_page_usable() / MKT_POSTING_ENTRY_SIZE(dim);
}

/* Max entries on a first page (with pt_centroid) */
static inline uint32_t
mkt_posting_max_entries_first(Dimension dim)
{
	uint32_t usable = mkt_posting_page_usable() -
					  mkt_posting_pt_centroid_size(dim);
	return usable / MKT_POSTING_ENTRY_SIZE(dim);
}

/* Buffer size for a flat page with count entries */
static inline size_t
mkt_posting_flat_page_size(Dimension dim, uint32_t count)
{
	return sizeof(MktFlatPostingHeader) +
		   (size_t)count * MKT_POSTING_ENTRY_SIZE(dim);
}

/* ----------------------------------------------------------------
 * Paged-mode access — opaque via PG PageGetSpecialPointer
 * ---------------------------------------------------------------- */

static inline MktPostingPageOpaque *
mkt_posting_opaque(Page page)
{
	return (MktPostingPageOpaque *)PageGetSpecialPointer(page);
}

static inline uint32_t
mkt_posting_page_count(Page page)
{
	return mkt_posting_opaque(page)->entry_count;
}

static inline bool
mkt_posting_page_has_room(Page page)
{
	MktPostingPageOpaque *op = mkt_posting_opaque(page);
	return op->entry_count < op->max_entries;
}

/*
 * Per-cluster head metadata (read from the FIRST page). tail_blkno ==
 * InvalidBlockNumber means it has not been computed yet — see
 * mkt_posting_insert_one, which fills it lazily on the first insert.
 */
static inline uint32_t
mkt_posting_head_live_count(Page head)
{
	return mkt_posting_opaque(head)->live_count;
}

static inline BlockNumber
mkt_posting_head_tail(Page head)
{
	return mkt_posting_opaque(head)->tail_blkno;
}

/* ----------------------------------------------------------------
 * Content-based AoS access — header-agnostic
 *
 * Entries are laid out contiguously: entry i starts at
 *   content + i * MKT_POSTING_ENTRY_SIZE(dim)
 * and consists of an MktPostingEntryHeader followed by
 * MKT_RABITQ_BYTES(dim) bytes of bits. The SIMD kernel strides
 * through bits[] at stride = MKT_POSTING_ENTRY_SIZE(dim), starting
 * from mkt_posting_first_bits(content).
 * ---------------------------------------------------------------- */

/* Header for entry i (access fields via the struct: hdr->meta,
 * hdr->f_add, hdr->f_rescale, hdr->f_error). */
static inline MktPostingEntryHeader *
mkt_posting_entry_at(char *content, uint32_t i, Dimension dim)
{
	return (MktPostingEntryHeader *)(content +
									 (size_t)i * MKT_POSTING_ENTRY_SIZE(dim));
}

/* Bits pointer for entry i (immediately follows entry i's header). */
static inline uint8_t *
mkt_posting_entry_bits_at(
		char *content, uint32_t max_entries, Dimension dim, uint32_t i)
{
	(void)max_entries;
	return mkt_posting_entry_at(content, i, dim)->bits;
}

/* Pointer to entry 0's bits — the base the SIMD kernel uses, with
 * stride = MKT_POSTING_ENTRY_SIZE(dim). */
static inline uint8_t *
mkt_posting_first_bits(char *content)
{
	return ((MktPostingEntryHeader *)content)->bits;
}

/* ----------------------------------------------------------------
 * Paged-mode convenience wrappers
 *
 * These use PageGetContents for the content pointer and read
 * max_entries from the opaque. Used by page init/add and tests.
 * ---------------------------------------------------------------- */

static inline char *
mkt_posting_content(Page page)
{
	return PageGetContents(page);
}

/*
 * On first pages, P^T * centroid is stored right after PageHeader,
 * before the SoA content. Content starts after pt_centroid area.
 */
static inline const float *
mkt_posting_pt_centroid(Page page)
{
	return (const float *)PageGetContents(page);
}

static inline float *
mkt_posting_pt_centroid_mut(Page page)
{
	return (float *)PageGetContents(page);
}

static inline char *
mkt_posting_content_first(Page page, Dimension dim)
{
	return PageGetContents(page) + mkt_posting_pt_centroid_size(dim);
}

/* Paged-mode convenience wrappers — these assume non-first pages
 * (content starts right after PageHeader). For first pages with
 * pt_centroid, go through mkt_posting_content_first. */
static inline MktPostingEntryHeader *
mkt_posting_entry(Page page, uint32_t i, Dimension dim)
{
	return mkt_posting_entry_at(PageGetContents(page), i, dim);
}

static inline uint8_t *
mkt_posting_entry_bits(
		Page page, uint32_t max_entries, Dimension dim, uint32_t i)
{
	return mkt_posting_entry_bits_at(
			PageGetContents(page), max_entries, dim, i);
}

/* ----------------------------------------------------------------
 * Flat-mode access
 * ---------------------------------------------------------------- */

static inline MktFlatPostingHeader *
mkt_flat_posting_header(char *buf)
{
	return (MktFlatPostingHeader *)buf;
}

static inline char *
mkt_flat_posting_content(char *buf)
{
	return buf + sizeof(MktFlatPostingHeader);
}

/* ----------------------------------------------------------------
 * Standalone helper: pack/extract vector_id in ItemPointerData
 * ---------------------------------------------------------------- */

static inline void
mkt_posting_set_vector_id(ItemPointerData *tid, uint32_t vector_id)
{
	ItemPointerSet(tid, (BlockNumber)vector_id, 0);
}

static inline uint32_t
mkt_posting_get_vector_id(const ItemPointerData *tid)
{
	return (uint32_t)ItemPointerGetBlockNumber(tid);
}

/* ----------------------------------------------------------------
 * TID ↔ uint64_t encoding for MktTopK
 *
 * Both standalone and PG store TIDs in MktTopK's uint64_t id field.
 * For standalone: vector_id is packed in BlockNumber with offset=0,
 * so encode yields (vector_id << 16) and decode_vector_id shifts
 * back. For PG: full ItemPointerData (block + offset) fits in 48
 * bits of the uint64_t.
 * ---------------------------------------------------------------- */

static inline uint64_t
mkt_posting_encode_tid(const ItemPointerData *tid)
{
	return ((uint64_t)ItemPointerGetBlockNumber(tid) << 16) |
		   (uint64_t)ItemPointerGetOffsetNumber(tid);
}

static inline ItemPointerData
mkt_posting_decode_tid(uint64_t id)
{
	ItemPointerData tid;
	ItemPointerSet(&tid, (BlockNumber)(id >> 16), (OffsetNumber)(id & 0xFFFF));
	return tid;
}

/* Convenience for standalone: extract vector_id from encoded TID.
 * Works because mkt_posting_set_vector_id stores vid in BlockNumber
 * with offset=0, so (id >> 16) == vid. */
static inline uint32_t
mkt_posting_decode_vector_id(uint64_t id)
{
	return (uint32_t)(id >> 16);
}

/* ----------------------------------------------------------------
 * Paged-mode page operations
 * ---------------------------------------------------------------- */

/*
 * Initialize an empty BLCKSZ posting page.
 */
void mkt_posting_page_init(
		Page page, uint32_t cluster_id, Dimension dim, uint16_t flags);

/*
 * Add one entry to a BLCKSZ page. Returns false if page is full.
 */
bool mkt_posting_page_add(
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
 * mkt_posting_flat_page_size(dim, max_entries) bytes.
 */
void
mkt_posting_flat_init(char *buf, uint32_t max_entries, uint32_t cluster_id);

/*
 * Add one entry to a flat page. Returns false if full.
 */
bool mkt_posting_flat_add(
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
 *
 * Pages are identified by MKT_POSTING_PAGE_FASTSCAN in opaque flags.
 * ---------------------------------------------------------------- */

#include "quant/fastscan.h"

/* Bytes per 32-vector group section (metadata + packed codes) */
static inline uint32_t
mkt_fastscan_group_section_bytes(Dimension dim)
{
	return (uint32_t)(MKT_FASTSCAN_GROUP * sizeof(ItemPointerData) +
					  MKT_FASTSCAN_GROUP * 3 * sizeof(float) +
					  MKT_FASTSCAN_GROUP_BYTES(dim));
}

/* Max entries on a fastscan overflow page */
static inline uint32_t
mkt_fastscan_max_entries(Dimension dim)
{
	uint32_t section = mkt_fastscan_group_section_bytes(dim);
	uint32_t usable	 = mkt_posting_page_usable();
	uint32_t ngroups = usable / section;
	return ngroups * MKT_FASTSCAN_GROUP;
}

/* Max entries on a fastscan first page (with pt_centroid) */
static inline uint32_t
mkt_fastscan_max_entries_first(Dimension dim)
{
	uint32_t section = mkt_fastscan_group_section_bytes(dim);
	uint32_t usable	 = mkt_posting_page_usable() -
					  mkt_posting_pt_centroid_size(dim);
	uint32_t ngroups = usable / section;
	return ngroups * MKT_FASTSCAN_GROUP;
}

/* Max groups on a page */
static inline uint32_t
mkt_fastscan_max_groups(Dimension dim, bool is_first)
{
	uint32_t section = mkt_fastscan_group_section_bytes(dim);
	uint32_t usable	 = mkt_posting_page_usable();
	if (is_first)
		usable -= mkt_posting_pt_centroid_size(dim);
	return usable / section;
}

/* ----------------------------------------------------------------
 * Fastscan group accessors
 *
 * All take content pointer (past PageHeader, past pt_centroid on
 * first pages) and group index g.
 * ---------------------------------------------------------------- */

static inline char *
mkt_fastscan_group_base(char *content, uint32_t g, Dimension dim)
{
	return content + (size_t)g * mkt_fastscan_group_section_bytes(dim);
}

static inline ItemPointerData *
mkt_fastscan_group_tids(char *content, uint32_t g, Dimension dim)
{
	return (ItemPointerData *)mkt_fastscan_group_base(content, g, dim);
}

static inline float *
mkt_fastscan_group_f_add(char *content, uint32_t g, Dimension dim)
{
	return (float *)(mkt_fastscan_group_base(content, g, dim) +
					 MKT_FASTSCAN_GROUP * sizeof(ItemPointerData));
}

static inline float *
mkt_fastscan_group_f_rescale(char *content, uint32_t g, Dimension dim)
{
	return mkt_fastscan_group_f_add(content, g, dim) + MKT_FASTSCAN_GROUP;
}

static inline float *
mkt_fastscan_group_f_error(char *content, uint32_t g, Dimension dim)
{
	return mkt_fastscan_group_f_rescale(content, g, dim) + MKT_FASTSCAN_GROUP;
}

static inline uint8_t *
mkt_fastscan_group_codes(char *content, uint32_t g, Dimension dim)
{
	return (uint8_t *)(mkt_fastscan_group_f_error(content, g, dim) +
					   MKT_FASTSCAN_GROUP);
}

#endif /* MKT_POSTING_PAGE_H */
