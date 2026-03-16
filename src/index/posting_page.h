/*
 * posting_page.h - Posting list page layout
 *
 * Posting pages store RaBitQ-encoded vectors with heap TIDs using
 * bidirectional growth, mirroring centroid_page.h:
 *
 *   [PageHeaderData (24B)]
 *   [MktPostingEntryMeta[0]  ]  <- metadata grows forward
 *   [MktPostingEntryMeta[1]  ]
 *   [       ...               ]
 *   [       free space         ]
 *   [       ...               ]
 *   [   RaBitQData[1]         ]  <- quantized data grows backward
 *   [   RaBitQData[0]         ]
 *   [MktPostingPageOpaque(16B)]
 *
 * Metadata (8B fixed) grows from the top; RaBitQData (variable-size,
 * 8 + ceil(dim/8) bytes) grows from the bottom. The page is full when
 * the two regions would overlap.
 *
 * Pages are chained via next_blkno to form a posting list for one
 * cluster. The first page has the FIRST flag set.
 */

#ifndef MKT_POSTING_PAGE_H
#define MKT_POSTING_PAGE_H

#include <stdint.h>
#include <string.h>

#include "core/memory.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

#ifdef MKT_STANDALONE
#include "core/pg_compat.h"
#else
#include <postgres.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <storage/bufpage.h>
#pragma GCC diagnostic pop
#include <storage/itemptr.h>
#endif

/* ----------------------------------------------------------------
 * Page type identifier
 * ---------------------------------------------------------------- */
#define MKT_POSTING_PAGE_ID ((uint16_t)0x4D50) /* "MP" */

/* ----------------------------------------------------------------
 * Posting entry flags
 * ---------------------------------------------------------------- */
#define MKT_POSTING_FLAG_DELETED  ((uint8_t)0x01)
#define MKT_POSTING_FLAG_BOUNDARY ((uint8_t)0x02)

/* ----------------------------------------------------------------
 * Posting page flags
 * ---------------------------------------------------------------- */
#define MKT_POSTING_PAGE_FIRST	  ((uint16_t)0x0001)
#define MKT_POSTING_PAGE_OVERFLOW ((uint16_t)0x0002)

/* ----------------------------------------------------------------
 * Per-entry metadata (8 bytes, grows forward from page header)
 * ---------------------------------------------------------------- */
typedef struct MktPostingEntryMeta
{
	ItemPointerData tid;	  /* 6B heap TID */
	uint8_t			flags;	  /* DELETED, BOUNDARY */
	uint8_t			reserved; /* alignment padding */
} MktPostingEntryMeta;

/* ----------------------------------------------------------------
 * Page special area (16 bytes, at page end per PG convention)
 * ---------------------------------------------------------------- */
typedef struct MktPostingPageOpaque
{
	BlockNumber next_blkno;	 /* 4B - next page in chain */
	uint32_t	cluster_id;	 /* 4B - owning cluster */
	uint16_t	entry_count; /* 2B - entries on this page */
	uint16_t	flags;		 /* 2B - FIRST, OVERFLOW */
	uint16_t	page_id;	 /* 2B - MKT_POSTING_PAGE_ID */
	uint16_t	reserved;	 /* 2B - alignment */
} MktPostingPageOpaque;

/* ----------------------------------------------------------------
 * Capacity calculation
 * ---------------------------------------------------------------- */

/* Usable bytes on a posting page (between header and opaque) */
#define MKT_POSTING_PAGE_USABLE      \
	(BLCKSZ - SizeOfPageHeaderData - \
	 (size_t)MAXALIGN(sizeof(MktPostingPageOpaque)))

/* Bytes consumed per entry: metadata + RaBitQData */
static inline uint32_t
mkt_posting_entry_bytes(Dimension dim)
{
	return sizeof(MktPostingEntryMeta) + MKT_RABITQ_DATA_SIZE(dim);
}

/* Maximum entries that fit on one page for a given dimension */
static inline uint32_t
mkt_posting_max_entries(Dimension dim)
{
	return (uint32_t)(MKT_POSTING_PAGE_USABLE / mkt_posting_entry_bytes(dim));
}

/* ----------------------------------------------------------------
 * Page access helpers
 * ---------------------------------------------------------------- */

/* Opaque area via PG-standard PageGetSpecialPointer */
#define MKT_POSTING_OPAQUE(page) \
	((MktPostingPageOpaque *)PageGetSpecialPointer(page))

/* Get mutable pointer to the i-th metadata entry (write path) */
static inline MktPostingEntryMeta *
mkt_posting_meta_mut(Page page, uint32_t index)
{
	return (MktPostingEntryMeta *)PageGetContents(page) + index;
}

/* Get pointer to the i-th metadata entry (read-only) */
static inline const MktPostingEntryMeta *
mkt_posting_meta(const Page page, uint32_t index)
{
	return mkt_posting_meta_mut(page, index);
}

/*
 * Get pointer to the i-th RaBitQData entry (backward region).
 *
 * Entry 0 is at the highest address (just before opaque), entry 1
 * is below it, etc.
 */
static inline const RaBitQData *
mkt_posting_data(const Page page, uint32_t index, Dimension dim)
{
	uint32_t data_size = MKT_RABITQ_DATA_SIZE(dim);
	return (const RaBitQData *)(PageGetSpecialPointer(page) -
								(size_t)(index + 1) * data_size);
}

/* ----------------------------------------------------------------
 * Page operations
 * ---------------------------------------------------------------- */

/*
 * Initialize a posting page. Zeroes the page, sets up the opaque
 * area with cluster_id and flags.
 */
void mkt_posting_page_init(Page page, uint32_t cluster_id, uint16_t flags);

/*
 * Add a posting entry to a page. Writes metadata forward and
 * RaBitQData backward. Returns true if the entry fits, false if
 * the page is full.
 */
bool mkt_posting_page_add(
		Page			  page,
		Dimension		  dim,
		BlockNumber		  block,
		OffsetNumber	  offset,
		const RaBitQData *data,
		uint8_t			  flags);

/*
 * Check if a page has room for one more entry.
 */
static inline bool
mkt_posting_page_has_room(Page page, Dimension dim)
{
	PageHeader header	= (PageHeader)page;
	size_t	   need_fwd = sizeof(MktPostingEntryMeta);
	size_t	   need_bwd = MKT_RABITQ_DATA_SIZE(dim);

	return header->pd_lower + need_fwd <= header->pd_upper - need_bwd;
}

#endif /* MKT_POSTING_PAGE_H */
