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
	uint16_t	flags;		 /* FIRST | OVERFLOW | FASTSCAN */
	uint16_t	page_id;	 /* MKT_POSTING_PAGE_ID */
	uint16_t	max_entries; /* capacity of this page */
} MktPostingPageOpaque;		 /* 16B */

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

#endif /* MKT_POSTING_PAGE_H */
