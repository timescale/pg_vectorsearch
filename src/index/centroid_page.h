/*
 * centroid_page.h - Centroid tree page layout
 *
 * Centroid pages use bidirectional growth, inspired by PostgreSQL's
 * standard page layout:
 *
 *   [PageHeaderData (24B)]
 *   [MktCentroidEntryMeta[0]  ]  ← metadata grows forward
 *   [MktCentroidEntryMeta[1]  ]
 *   [       ...               ]
 *   [       free space         ]
 *   [       ...               ]
 *   [   data[1]               ]  ← vector data grows backward
 *   [   data[0]               ]
 *   [MktCentroidPageOpaque(12B)]
 *
 * Metadata grows from the top; vector data grows from the bottom.
 * The page is full when the two regions would overlap. Metadata
 * is 8 bytes per entry (uniform across all formats).
 *
 * The data format is selected at page initialization and stored in the
 * low 2 bits of opaque->flags:
 *
 *   RABITQ  (default) — RaBitQData (8 + ceil(dim/8) bytes per entry)
 *   FLOAT   — float32 vectors (dim * 4 bytes per entry)
 *   HALF    — float16 vectors (dim * 2 bytes per entry)
 *
 * RABITQ pages encode centroids relative to a global mean, enabling
 * fast approximate distance with error bounds. FLOAT/HALF pages store
 * full-precision vectors for exact routing at the cost of fewer
 * entries per page.
 */

#ifndef MKT_CENTROID_PAGE_H
#define MKT_CENTROID_PAGE_H

#include <stdint.h>
#include <string.h>

#include "core/memory.h"
#include "mkt_halfvec.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/* Include PG compat for standalone, real PG headers for extension */
#ifdef MKT_STANDALONE
#include "standalone/pg_compat.h"
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
#define MKT_CENTROID_PAGE_ID ((uint16_t)0x4D43) /* "MC" */

/* ----------------------------------------------------------------
 * Centroid entry flags
 * ---------------------------------------------------------------- */
#define MKT_CENTROID_FLAG_LEAF ((uint16_t)0x0001)

/* ----------------------------------------------------------------
 * Centroid data format (stored in low 2 bits of opaque->flags)
 * ---------------------------------------------------------------- */
#define MKT_CENTROID_FMT_MASK ((uint8_t)0x03)
#define MKT_CENTROID_OPAQUE_LEAF \
	((uint8_t)0x04) /* page contains leaf entries */

typedef enum MktCentroidFormat
{
	MKT_CENTROID_FMT_RABITQ = 0,
	MKT_CENTROID_FMT_FLOAT	= 1,
	MKT_CENTROID_FMT_HALF	= 2,
} MktCentroidFormat;

/* ----------------------------------------------------------------
 * Per-centroid metadata (grows forward from page header)
 *
 * Uniform 8-byte struct used by all formats. For internal nodes,
 * child_blkno points to the child centroid page. For leaf nodes,
 * child_blkno points to the posting list head page.
 * ---------------------------------------------------------------- */
typedef struct MktCentroidEntryMeta
{
	BlockNumber child_blkno; /* 4B - child page */
	uint16_t	child_count; /* 2B - children at next level */
	uint16_t	flags;		 /* 2B - MKT_CENTROID_FLAG_LEAF etc */
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

/* Metadata size per entry (uniform across all formats) */
static inline uint32_t
mkt_centroid_meta_size(MktCentroidFormat fmt)
{
	(void)fmt;
	return sizeof(MktCentroidEntryMeta);
}

/* Per-entry data size for routing (centroid vector or RaBitQ) */
static inline uint32_t
mkt_centroid_data_size(Dimension dim, MktCentroidFormat fmt)
{
	switch (fmt)
	{
	case MKT_CENTROID_FMT_FLOAT:
		return dim * sizeof(float);
	case MKT_CENTROID_FMT_HALF:
		return dim * sizeof(half);
	default:
		return MKT_RABITQ_DATA_SIZE(dim);
	}
}

/* Per-entry data size for leaf pages (routing + P^T * centroid) */
static inline uint32_t
mkt_centroid_leaf_data_size(Dimension dim, MktCentroidFormat fmt)
{
	return mkt_centroid_data_size(dim, fmt) + dim * sizeof(float);
}

/* Bytes consumed per entry (metadata + data) for a given format */
static inline uint32_t
mkt_centroid_entry_bytes_fmt(Dimension dim, MktCentroidFormat fmt)
{
	return mkt_centroid_meta_size(fmt) + mkt_centroid_data_size(dim, fmt);
}

/* Bytes consumed per leaf entry (metadata + routing + pt_centroid) */
static inline uint32_t
mkt_centroid_leaf_entry_bytes_fmt(Dimension dim, MktCentroidFormat fmt)
{
	return mkt_centroid_meta_size(fmt) + mkt_centroid_leaf_data_size(dim, fmt);
}

/* Maximum entries per page for a given format */
static inline uint32_t
mkt_centroid_max_entries_fmt(Dimension dim, MktCentroidFormat fmt)
{
	return (uint32_t)(MKT_CENTROID_PAGE_USABLE /
					  mkt_centroid_entry_bytes_fmt(dim, fmt));
}

/* Maximum entries per leaf page (includes pt_centroid per entry) */
static inline uint32_t
mkt_centroid_max_leaf_entries_fmt(Dimension dim, MktCentroidFormat fmt)
{
	return (uint32_t)(MKT_CENTROID_PAGE_USABLE /
					  mkt_centroid_leaf_entry_bytes_fmt(dim, fmt));
}

/* Backward-compatible wrappers (default to RaBitQ format) */
static inline uint32_t
mkt_centroid_entry_bytes(Dimension dim)
{
	return mkt_centroid_entry_bytes_fmt(dim, MKT_CENTROID_FMT_RABITQ);
}

static inline uint32_t
mkt_centroid_max_entries(Dimension dim)
{
	return mkt_centroid_max_entries_fmt(dim, MKT_CENTROID_FMT_RABITQ);
}

/* ----------------------------------------------------------------
 * Page access helpers
 *
 * Metadata grows forward from page header.
 * Vector data grows backward from the opaque area.
 * ---------------------------------------------------------------- */

/* Opaque area via PG-standard PageGetSpecialPointer */
#define MKT_CENTROID_OPAQUE(page) \
	((MktCentroidPageOpaque *)PageGetSpecialPointer(page))

/* Data format stored in the page */
static inline MktCentroidFormat
mkt_centroid_page_format(Page page)
{
	return (MktCentroidFormat)(MKT_CENTROID_OPAQUE(page)->flags &
							   MKT_CENTROID_FMT_MASK);
}

/*
 * Get mutable pointer to the i-th metadata entry (write path).
 * Uses byte-offset arithmetic since meta size is format-dependent.
 */
static inline MktCentroidEntryMeta *
mkt_centroid_meta_mut(Page page, uint32_t index)
{
	uint32_t meta_size = mkt_centroid_meta_size(
			mkt_centroid_page_format(page));
	return (MktCentroidEntryMeta *)((char *)PageGetContents(page) +
									index * meta_size);
}

/* Get pointer to the i-th metadata entry (read-only) */
static inline const MktCentroidEntryMeta *
mkt_centroid_meta(const Page page, uint32_t index)
{
	return mkt_centroid_meta_mut(page, index);
}

/*
 * Get pointer to the i-th data entry (backward region).
 *
 * Entry 0 is at the highest address (just before opaque), entry 1
 * is below it, etc. Data size is determined by the page format.
 *
 * Layout (growing backward from opaque):
 *   opaque_start - 1*data_size = data[0]
 *   opaque_start - 2*data_size = data[1]
 *   ...
 */
/*
 * Data accessor for the backward-growing data region.
 *
 * For leaf pages, each entry stores [routing_data | pt_centroid]
 * so the total data_size per entry is larger. For internal pages,
 * only routing_data is stored.
 */
static inline const void *
mkt_centroid_entry_data(const Page page, uint32_t index, Dimension dim)
{
	MktCentroidFormat fmt		= mkt_centroid_page_format(page);
	uint32_t		  data_size = mkt_centroid_data_size(dim, fmt);
	return (const void *)(PageGetSpecialPointer(page) -
						  (size_t)(index + 1) * data_size);
}

/* Typed accessors for routing data (at start of data region) */
static inline const RaBitQData *
mkt_centroid_data(const Page page, uint32_t index, Dimension dim)
{
	return (const RaBitQData *)mkt_centroid_entry_data(page, index, dim);
}

static inline const float *
mkt_centroid_float_data(const Page page, uint32_t index, Dimension dim)
{
	return (const float *)mkt_centroid_entry_data(page, index, dim);
}

static inline const half *
mkt_centroid_half_data(const Page page, uint32_t index, Dimension dim)
{
	return (const half *)mkt_centroid_entry_data(page, index, dim);
}

/* ----------------------------------------------------------------
 * Page operations
 * ---------------------------------------------------------------- */

/*
 * Initialize a centroid page with explicit data format.
 * Zeroes the page, sets up the opaque area with format flag.
 */
void
mkt_centroid_page_init_fmt(Page page, uint8_t level, MktCentroidFormat fmt);

/* Backward-compatible init (defaults to RaBitQ format) */
static inline void
mkt_centroid_page_init(Page page, uint8_t level)
{
	mkt_centroid_page_init_fmt(page, level, MKT_CENTROID_FMT_RABITQ);
}

/*
 * Reserve space for a centroid entry. Writes metadata forward,
 * reserves data space backward, returns a writable pointer to
 * the data region. The caller writes entry data directly into
 * the returned pointer (data_size bytes).
 *
 * Returns NULL if the page has no room.
 */
void *mkt_centroid_page_add_entry_begin(
		Page		page,
		Dimension	dim,
		BlockNumber child_blkno,
		uint16_t	child_count,
		uint16_t	flags);

/*
 * Add a centroid entry to a page. Writes metadata forward and
 * vector data backward. The data format is read from the page's
 * opaque flags to determine data size.
 *
 * data points to the entry payload:
 *   RABITQ → const RaBitQData *
 *   FLOAT  → const float * (dim elements)
 *   HALF   → const half * (dim elements)
 */
bool mkt_centroid_page_add_entry(
		Page		page,
		Dimension	dim,
		BlockNumber child_blkno,
		uint16_t	child_count,
		uint16_t	flags,
		const void *data);

/* Backward-compatible add (RaBitQ-typed parameter) */
static inline bool
mkt_centroid_page_add(
		Page			  page,
		Dimension		  dim,
		BlockNumber		  child_blkno,
		uint16_t		  child_count,
		uint16_t		  flags,
		const RaBitQData *data)
{
	return mkt_centroid_page_add_entry(
			page, dim, child_blkno, child_count, flags, data);
}

/*
 * Check if a page has room for one more entry.
 * Reads data format from the page to determine entry size.
 */
static inline bool
mkt_centroid_page_has_room(Page page, Dimension dim, bool is_leaf)
{
	PageHeader		  header   = (PageHeader)page;
	MktCentroidFormat fmt	   = mkt_centroid_page_format(page);
	size_t			  need_fwd = mkt_centroid_meta_size(fmt);
	size_t need_bwd = is_leaf ? mkt_centroid_leaf_data_size(dim, fmt)
							  : mkt_centroid_data_size(dim, fmt);

	return header->pd_lower + need_fwd + need_bwd <= header->pd_upper;
}

#endif /* MKT_CENTROID_PAGE_H */
