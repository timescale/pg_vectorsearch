/*
 * centroid_page.h - Centroid tree page layout and I/O abstraction
 *
 * Centroid pages store RaBitQ-encoded medoid vectors in SoA layout for
 * SIMD-friendly batch distance computation. The page layout is:
 *
 *   [PageHeaderData (24B)]
 *   [MktCentroidEntryMeta[max] (max * 16B)]
 *   [float f_add[max]          (max * 4B)]
 *   [float f_rescale[max]      (max * 4B)]
 *   [uint8_t bits[max * D/8]   (max * D/8 B)]
 *   [padding]
 *   [MktCentroidPageOpaque (16B)]
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
 * Per-centroid metadata (16 bytes)
 *
 * Stored in the first SoA region of the page. For internal nodes,
 * child_blkno points to the child centroid page. For leaf nodes,
 * child_blkno points to the posting list head page.
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
 * Page special area (16 bytes, at page end per PG convention)
 * ---------------------------------------------------------------- */
typedef struct MktCentroidPageOpaque
{
	BlockNumber next_blkno;		  /* 4B - next page at same level */
	uint16_t	entry_count;	  /* 2B - centroids on this page */
	uint8_t		level;			  /* 1B - tree level (0 = root) */
	uint8_t		flags;			  /* 1B - page flags */
	uint16_t	page_id;		  /* 2B - MKT_CENTROID_PAGE_ID */
	uint16_t	first_global_idx; /* 2B - global index of first entry */
	uint32_t	reserved;		  /* 4B */
} MktCentroidPageOpaque;

/* ----------------------------------------------------------------
 * Capacity calculation
 * ---------------------------------------------------------------- */

/* Usable bytes on a centroid page */
#define MKT_CENTROID_PAGE_USABLE \
	(BLCKSZ - SizeOfPageHeaderData - sizeof(MktCentroidPageOpaque))

/* Maximum entries that fit on one page for a given dimension */
static inline uint32_t
mkt_centroid_max_entries(Dimension dim)
{
	uint32_t entry_bytes = sizeof(MktCentroidEntryMeta) + sizeof(float) +
						   sizeof(float) + MKT_RABITQ_BYTES(dim);
	return (uint32_t)(MKT_CENTROID_PAGE_USABLE / entry_bytes);
}

/* ----------------------------------------------------------------
 * SoA access macros
 *
 * These compute pointers into the pre-allocated SoA regions.
 * 'max' = mkt_centroid_max_entries(dim), computed once per index.
 * ---------------------------------------------------------------- */

#define MKT_CENTROID_META(page) \
	((MktCentroidEntryMeta *)((char *)(page) + SizeOfPageHeaderData))

#define MKT_CENTROID_F_ADD(page, max)                  \
	((float *)((char *)(page) + SizeOfPageHeaderData + \
			   (max) * sizeof(MktCentroidEntryMeta)))

#define MKT_CENTROID_F_RESCALE(page, max)              \
	((float *)((char *)(page) + SizeOfPageHeaderData + \
			   (max) * sizeof(MktCentroidEntryMeta) + (max) * sizeof(float)))

#define MKT_CENTROID_BITS(page, max)                     \
	((uint8_t *)((char *)(page) + SizeOfPageHeaderData + \
				 (max) * sizeof(MktCentroidEntryMeta) +  \
				 (max) * 2 * sizeof(float)))

#define MKT_CENTROID_OPAQUE(page)                        \
	((MktCentroidPageOpaque *)((char *)(page) + BLCKSZ - \
							   sizeof(MktCentroidPageOpaque)))

/* ----------------------------------------------------------------
 * I/O abstraction callbacks
 *
 * Standalone: ctx = array of malloc'd 8KB buffers
 * PG mode:    ctx = Relation, wraps ReadBuffer/ReleaseBuffer
 * ---------------------------------------------------------------- */
typedef struct MktPageAccessor
{
	Page (*read)(void *ctx, BlockNumber blkno);
	void (*release)(void *ctx, BlockNumber blkno);

	/* Write path: new page allocation + mark dirty */
	Page (*new_page)(void *ctx, BlockNumber *blkno_out);
	void (*mark_dirty)(void *ctx, BlockNumber blkno);

	void *ctx;
} MktPageAccessor;

/* ----------------------------------------------------------------
 * Vector fetch abstraction
 *
 * Standalone: ctx = { float *vectors, Dimension dim }
 * PG mode:    ctx = Relation (heap), uses heap_fetch(tid)
 * ---------------------------------------------------------------- */
typedef struct MktVectorAccessor
{
	VectorRef (*fetch)(void *ctx, ItemPointerData tid);
	void *ctx;
} MktVectorAccessor;

/* ----------------------------------------------------------------
 * WAL abstraction (optional, NULL = no WAL)
 * ---------------------------------------------------------------- */
typedef struct MktWALWriter
{
	void (*log_page_init)(
			void *ctx, BlockNumber blkno, uint8_t level, Dimension dim);
	void (*log_page_add)(
			void *ctx, BlockNumber blkno, Page page, Dimension dim);
	void *ctx;
} MktWALWriter;

/* ----------------------------------------------------------------
 * Page operations
 * ---------------------------------------------------------------- */

/*
 * Initialize a centroid page. Zeroes the page, sets up the opaque area.
 */
void
mkt_centroid_page_init(Page page, uint8_t level, uint16_t first_global_idx);

/*
 * Add a centroid entry to a page. Returns true if the entry fits,
 * false if the page is full.
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
	MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
	return opaque->entry_count < mkt_centroid_max_entries(dim);
}

#endif /* MKT_CENTROID_PAGE_H */
