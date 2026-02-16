/*
 * centroid_page.h - Centroid tree page layout
 *
 * Centroid pages store RaBitQ-encoded medoid vectors using bidirectional
 * growth, inspired by PostgreSQL's standard page layout:
 *
 *   [PageHeaderData (24B)]
 *   [MktCentroidEntryMeta[0]  ]  ← metadata grows forward
 *   [MktCentroidEntryMeta[1]  ]
 *   [       ...               ]
 *   [       free space         ]
 *   [       ...               ]
 *   [   RaBitQData[1]         ]  ← quantized data grows backward
 *   [   RaBitQData[0]         ]
 *   [MktCentroidPageOpaque(12B)]
 *
 * Metadata (16B fixed) grows from the top; RaBitQData (variable-size,
 * 8 + ceil(dim/8) bytes) grows from the bottom. The page is full when
 * the two regions would overlap.
 *
 * All centroids are RaBitQ-encoded relative to the global data mean,
 * sharing a single orthogonal matrix P. The query is transformed once
 * and reused at every tree level.
 *
 * Centroids are medoids — actual data vectors referenced by heap TID.
 * This eliminates full-precision centroid storage in the index.
 */

#ifndef MKT_CENTROID_PAGE_H
#define MKT_CENTROID_PAGE_H

#include <stdint.h>
#include <string.h>

#include "core/memory.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/* Include PG compat for standalone, real PG headers for extension */
#ifdef MKT_STANDALONE
#include "core/pg_compat.h"
#else
#include <postgres.h>

#include <storage/bufpage.h>
#include <storage/itemptr.h>
#endif

/* ----------------------------------------------------------------
 * Page type identifier
 * ---------------------------------------------------------------- */
#define MKT_CENTROID_PAGE_ID ((uint16_t)0x4D43) /* "MC" */

/* ----------------------------------------------------------------
 * Centroid entry flags
 * ---------------------------------------------------------------- */
#define MKT_CENTROID_FLAG_LEAF ((uint16_t)0x0001)

/* ----------------------------------------------------------------
 * Per-centroid metadata (16 bytes, grows forward from page header)
 *
 * For internal nodes, child_blkno points to the child centroid page.
 * For leaf nodes, child_blkno points to the posting list head page.
 * ---------------------------------------------------------------- */
typedef struct MktCentroidEntryMeta
{
	BlockNumber		child_blkno; /* 4B - child page */
	uint16_t		child_count; /* 2B - children at next level */
	uint16_t		flags;		 /* 2B - MKT_CENTROID_FLAG_LEAF etc */
	ItemPointerData medoid_tid;	 /* 6B - heap TID of medoid vector */
	uint16_t		reserved;	 /* 2B - alignment padding */
} MktCentroidEntryMeta;

/* ----------------------------------------------------------------
 * Page special area (12 bytes, at page end per PG convention)
 * ---------------------------------------------------------------- */
typedef struct MktCentroidPageOpaque
{
	BlockNumber next_blkno;	 /* 4B - next page at same level */
	uint16_t	entry_count; /* 2B - centroids on this page */
	uint8_t		level;		 /* 1B - tree level (0 = root) */
	uint8_t		flags;		 /* 1B - page flags */
	uint16_t	page_id;	 /* 2B - MKT_CENTROID_PAGE_ID */
	uint16_t	padding;	 /* 2B - alignment */
} MktCentroidPageOpaque;

/* ----------------------------------------------------------------
 * Capacity calculation
 * ---------------------------------------------------------------- */

/* Usable bytes on a centroid page (between header and MAXALIGN'd opaque) */
#define MKT_CENTROID_PAGE_USABLE     \
	(BLCKSZ - SizeOfPageHeaderData - \
	 (size_t)MAXALIGN(sizeof(MktCentroidPageOpaque)))

/* Bytes consumed per entry: metadata + RaBitQData */
static inline uint32_t
mkt_centroid_entry_bytes(Dimension dim)
{
	return sizeof(MktCentroidEntryMeta) + MKT_RABITQ_DATA_SIZE(dim);
}

/* Maximum entries that fit on one page for a given dimension */
static inline uint32_t
mkt_centroid_max_entries(Dimension dim)
{
	return (uint32_t)(MKT_CENTROID_PAGE_USABLE /
					  mkt_centroid_entry_bytes(dim));
}

/* ----------------------------------------------------------------
 * Page access helpers
 *
 * Metadata grows forward from page header.
 * RaBitQData grows backward from the opaque area.
 * ---------------------------------------------------------------- */

/* Opaque area via PG-standard PageGetSpecialPointer */
#define MKT_CENTROID_OPAQUE(page) \
	((MktCentroidPageOpaque *)PageGetSpecialPointer(page))

/* Get mutable pointer to the i-th metadata entry (write path) */
static inline MktCentroidEntryMeta *
mkt_centroid_meta_mut(Page page, uint32_t index)
{
	return (MktCentroidEntryMeta *)PageGetContents(page) + index;
}

/* Get pointer to the i-th metadata entry (read-only) */
static inline const MktCentroidEntryMeta *
mkt_centroid_meta(const Page page, uint32_t index)
{
	return mkt_centroid_meta_mut(page, index);
}

/*
 * Get pointer to the i-th RaBitQData entry (backward region).
 *
 * Entry 0 is at the highest address (just before opaque), entry 1
 * is below it, etc. This keeps insertion order consistent: the most
 * recently added entry is at the lowest address in the data region.
 *
 * Layout (growing backward from opaque):
 *   opaque_start - 1*data_size = RaBitQData[0]
 *   opaque_start - 2*data_size = RaBitQData[1]
 *   ...
 */
static inline const RaBitQData *
mkt_centroid_data(const Page page, uint32_t index, Dimension dim)
{
	uint32_t data_size = MKT_RABITQ_DATA_SIZE(dim);
	return (const RaBitQData *)(PageGetSpecialPointer(page) -
								(size_t)(index + 1) * data_size);
}

/* ----------------------------------------------------------------
 * Page operations
 * ---------------------------------------------------------------- */

/*
 * Initialize a centroid page. Zeroes the page, sets up the opaque area.
 */
void mkt_centroid_page_init(Page page, uint8_t level);

/*
 * Add a centroid entry to a page. Writes metadata forward and
 * RaBitQData backward. Returns true if the entry fits, false if
 * the page is full.
 *
 * The RaBitQData is memcpy'd as a contiguous struct.
 */
bool mkt_centroid_page_add(
		Page				   page,
		Dimension			   dim,
		BlockNumber			   child_blkno,
		uint16_t			   child_count,
		uint16_t			   flags,
		const ItemPointerData *medoid_tid,
		const RaBitQData	  *data);

/*
 * Check if a page has room for one more entry.
 */
static inline bool
mkt_centroid_page_has_room(Page page, Dimension dim)
{
	PageHeader header	= (PageHeader)page;
	size_t	   need_fwd = sizeof(MktCentroidEntryMeta);
	size_t	   need_bwd = MKT_RABITQ_DATA_SIZE(dim);

	return header->pd_lower + need_fwd <= header->pd_upper - need_bwd;
}

#endif /* MKT_CENTROID_PAGE_H */
