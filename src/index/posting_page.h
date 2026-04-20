/*
 * posting_page.h - Posting list page format
 *
 * Stores per-cluster posting data using an all-forward SoA layout.
 * Two storage modes share the same layout and scan code:
 *
 *   Paged mode (BLCKSZ pages):
 *     PageHeader (PG-compatible, 24B)
 *     SoA data (pre-sized to max_entries)
 *     MktPostingPageOpaque (16B at page end)
 *
 *   Flat mode (one buffer per cluster):
 *     MktFlatPostingHeader (16B)
 *     SoA data (exactly sized to entry count)
 *
 * SoA data layout (identical for both modes):
 *     MktPostingEntryMeta[max]   8B each  (TID + flags)
 *     float f_add[max]           4B each
 *     float f_rescale[max]       4B each
 *     float f_error[max]         4B each
 *     uint8_t bits[max * packed_bytes]    (contiguous 1-bit codes)
 *
 * Access helpers take a content pointer and max_entries, making
 * them agnostic to the page header format. The scan iterator
 * stores these per-page and works identically for both modes.
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

/* Bytes per entry in forward region: meta + 3 floats */
#define MKT_POSTING_FWD_PER_ENTRY \
	(sizeof(MktPostingEntryMeta) + 3 * sizeof(float)) /* 20B */

/* Bytes per entry in bits region */
#define MKT_POSTING_BITS_PER_ENTRY(dim) MKT_RABITQ_BYTES(dim)

/* Total bytes per entry */
#define MKT_POSTING_ENTRY_SIZE(dim) \
	(MKT_POSTING_FWD_PER_ENTRY + MKT_POSTING_BITS_PER_ENTRY(dim))

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
 * Content-based SoA access — header-agnostic
 *
 * These helpers take a content pointer (start of SoA data) and
 * max_entries. They work identically for both paged and flat modes.
 *
 * Layout within content area:
 *   offset 0:                      meta[max]
 *   offset max * sizeof(meta):     f_add[max]
 *   offset + max * sizeof(float):  f_rescale[max]
 *   offset + max * sizeof(float):  f_error[max]
 *   offset + max * sizeof(float):  bits[max * packed_bytes]
 * ---------------------------------------------------------------- */

static inline MktPostingEntryMeta *
mkt_posting_metas_at(char *content)
{
	return (MktPostingEntryMeta *)content;
}

static inline float *
mkt_posting_f_add_at(char *content, uint32_t max_entries)
{
	return (float *)(content +
					 (size_t)max_entries * sizeof(MktPostingEntryMeta));
}

static inline float *
mkt_posting_f_rescale_at(char *content, uint32_t max_entries)
{
	return mkt_posting_f_add_at(content, max_entries) + max_entries;
}

static inline float *
mkt_posting_f_error_at(char *content, uint32_t max_entries)
{
	return mkt_posting_f_rescale_at(content, max_entries) + max_entries;
}

static inline uint8_t *
mkt_posting_bits_at(char *content, uint32_t max_entries)
{
	return (uint8_t *)(mkt_posting_f_error_at(content, max_entries) +
					   max_entries);
}

static inline uint8_t *
mkt_posting_entry_bits_at(
		char *content, uint32_t max_entries, Dimension dim, uint32_t index)
{
	return mkt_posting_bits_at(content, max_entries) +
		   (size_t)index * MKT_RABITQ_BYTES(dim);
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

static inline MktPostingEntryMeta *
mkt_posting_metas(Page page)
{
	return mkt_posting_metas_at(PageGetContents(page));
}

static inline float *
mkt_posting_f_add(Page page, uint32_t max_entries)
{
	return mkt_posting_f_add_at(PageGetContents(page), max_entries);
}

static inline float *
mkt_posting_f_rescale(Page page, uint32_t max_entries)
{
	return mkt_posting_f_rescale_at(PageGetContents(page), max_entries);
}

static inline float *
mkt_posting_f_error(Page page, uint32_t max_entries)
{
	return mkt_posting_f_error_at(PageGetContents(page), max_entries);
}

static inline uint8_t *
mkt_posting_bits(Page page, uint32_t max_entries)
{
	return mkt_posting_bits_at(PageGetContents(page), max_entries);
}

static inline uint8_t *
mkt_posting_entry_bits(
		Page page, uint32_t max_entries, Dimension dim, uint32_t index)
{
	return mkt_posting_entry_bits_at(
			PageGetContents(page), max_entries, dim, index);
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
