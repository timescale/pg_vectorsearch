/*
 * centroid_page.h - Centroid tree page layout
 *
 * Centroid pages use bidirectional growth, inspired by PostgreSQL's
 * standard page layout:
 *
 *   [PageHeaderData (24B)]
 *   [PrismCentroidEntryMeta[0]  ]  ← metadata grows forward
 *   [PrismCentroidEntryMeta[1]  ]
 *   [       ...               ]
 *   [       free space         ]
 *   [       ...               ]
 *   [   data[1]               ]  ← vector data grows backward
 *   [   data[0]               ]
 *   [PrismCentroidPageOpaque(12B)]
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

#ifndef PRISM_CENTROID_PAGE_H
#define PRISM_CENTROID_PAGE_H

#include <stdint.h>
#include <string.h>

#include "core/memory.h"
#include "core/types.h"
#include "quant/fastscan.h"
#include "quant/rabitq.h"
#include "types/vec16.h"

/* Include PG compat for standalone, real PG headers for extension */
#ifdef VS_STANDALONE
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
#define PRISM_CENTROID_PAGE_ID ((uint16_t)0x4D43) /* "MC" */

/* ----------------------------------------------------------------
 * Centroid entry flags
 * ---------------------------------------------------------------- */
#define PRISM_CENTROID_FLAG_LEAF ((uint16_t)0x0001)

/* ----------------------------------------------------------------
 * Centroid data format (stored in low 2 bits of opaque->flags)
 * ---------------------------------------------------------------- */
#define PRISM_CENTROID_FMT_MASK ((uint8_t)0x03)

typedef enum PrismCentroidFormat
{
	PRISM_CENTROID_FMT_RABITQ = 0,
	PRISM_CENTROID_FMT_FLOAT  = 1,
	PRISM_CENTROID_FMT_HALF	  = 2,
	/*
	 * FASTSCAN: same RaBitQ codes as the RABITQ format, but rearranged
	 * into 32-vector groups with kPerm0 interleaving so the centroid
	 * scoring path can use vs_fastscan_accumulate instead of the
	 * per-vector vs_rabitq_inner_product_multi kernel.
	 *
	 * Group section layout (per 32 entries):
	 *   BlockNumber child_blkno[32]      128 B
	 *   float       f_add[32]            128 B
	 *   float       f_rescale[32]        128 B
	 *   float       f_error[32]          128 B
	 *   uint8_t     codes[nsq_pairs*32]  variable
	 *
	 * Per-entry PrismCentroidEntryMeta (4 B child_blkno + 4 B
	 * child_count/flags) is replaced by the array-of-fields layout
	 * above; child_count and per-entry flags are dropped because the
	 * scan already knows from its descent level whether children are
	 * posting heads vs. nested centroid pages, and child_count is only
	 * used at build time. Final group may be partial; unused slots
	 * have child_blkno = InvalidBlockNumber and zero-padded codes.
	 */
	PRISM_CENTROID_FMT_FASTSCAN = 3,
} PrismCentroidFormat;

/* ----------------------------------------------------------------
 * Per-centroid metadata (grows forward from page header)
 *
 * Uniform 8-byte struct used by all formats. For internal nodes,
 * child_blkno points to the child centroid page. For leaf nodes,
 * child_blkno points to the posting list head page.
 * ---------------------------------------------------------------- */
typedef struct PrismCentroidEntryMeta
{
	BlockNumber child_blkno; /* 4B - child page */
	uint16_t	child_count; /* 2B - children at next level */
	uint16_t	flags;		 /* 2B - PRISM_CENTROID_FLAG_LEAF etc */
} PrismCentroidEntryMeta;

/* ----------------------------------------------------------------
 * Page special area (12 bytes, at page end per PG convention)
 * ---------------------------------------------------------------- */
typedef struct PrismCentroidPageOpaque
{
	BlockNumber next_blkno;	 /* 4B - next page at same level */
	uint16_t	entry_count; /* 2B - centroids on this page */
	uint8_t		level;		 /* 1B - tree level (0 = root) */
	uint8_t		flags;		 /* 1B - page flags */
	uint16_t	page_id;	 /* 2B - PRISM_CENTROID_PAGE_ID */
	uint16_t	padding;	 /* 2B - alignment */
} PrismCentroidPageOpaque;

/* ----------------------------------------------------------------
 * Capacity calculation
 * ---------------------------------------------------------------- */

/* Usable bytes on a centroid page (between header and MAXALIGN'd opaque) */
#define PRISM_CENTROID_PAGE_USABLE   \
	(BLCKSZ - SizeOfPageHeaderData - \
	 (size_t)MAXALIGN(sizeof(PrismCentroidPageOpaque)))

/* Metadata size per entry (uniform across all formats) */
static inline uint32_t
prism_centroid_meta_size(PrismCentroidFormat fmt)
{
	(void)fmt;
	return sizeof(PrismCentroidEntryMeta);
}

/* Per-entry data size for routing (centroid vector or RaBitQ) */
static inline uint32_t
prism_centroid_data_size(Dimension dim, PrismCentroidFormat fmt)
{
	switch (fmt)
	{
	case PRISM_CENTROID_FMT_FLOAT:
		return dim * sizeof(float);
	case PRISM_CENTROID_FMT_HALF:
		return dim * sizeof(half);
	case PRISM_CENTROID_FMT_FASTSCAN:
		/* Average per-entry overhead inside a fastscan group. Used
		 * only by the legacy "data_size × N" capacity check; the
		 * real layout is group-based — see prism_centroid_fastscan_*. */
		return (uint32_t)(VS_FASTSCAN_GROUP * 3 * sizeof(float) +
						  VS_FASTSCAN_GROUP_BYTES(dim)) /
			   VS_FASTSCAN_GROUP;
	default:
		return VS_RABITQ_DATA_SIZE(dim);
	}
}

/* Per-entry data size for leaf pages (routing + P^T * centroid) */
static inline uint32_t
prism_centroid_leaf_data_size(Dimension dim, PrismCentroidFormat fmt)
{
	return prism_centroid_data_size(dim, fmt) + dim * sizeof(float);
}

/* Bytes consumed per entry (metadata + data) for a given format */
static inline uint32_t
prism_centroid_entry_bytes_fmt(Dimension dim, PrismCentroidFormat fmt)
{
	return prism_centroid_meta_size(fmt) + prism_centroid_data_size(dim, fmt);
}

/* Bytes consumed per leaf entry (metadata + routing + pt_centroid) */
static inline uint32_t
prism_centroid_leaf_entry_bytes_fmt(Dimension dim, PrismCentroidFormat fmt)
{
	return prism_centroid_meta_size(fmt) +
		   prism_centroid_leaf_data_size(dim, fmt);
}

/* Maximum entries per page for a given format */
static inline uint32_t
prism_centroid_max_entries_fmt(Dimension dim, PrismCentroidFormat fmt)
{
	if (fmt == PRISM_CENTROID_FMT_FASTSCAN)
	{
		/* fastscan stores entries in 32-vector groups; max_entries is
		 * ngroups * 32. See prism_centroid_fastscan_group_bytes. */
		uint32_t group_bytes =
				(uint32_t)(VS_FASTSCAN_GROUP * sizeof(BlockNumber) +
						   VS_FASTSCAN_GROUP * 3 * sizeof(float) +
						   VS_FASTSCAN_GROUP_BYTES(dim));
		uint32_t ngroups = (uint32_t)PRISM_CENTROID_PAGE_USABLE / group_bytes;
		return ngroups * VS_FASTSCAN_GROUP;
	}
	return (uint32_t)(PRISM_CENTROID_PAGE_USABLE /
					  prism_centroid_entry_bytes_fmt(dim, fmt));
}

/* ----------------------------------------------------------------
 * Fastscan centroid page layout
 *
 * No PostgreSQL-style meta-grows-forward / data-grows-backward — the
 * whole page contents area is a sequence of fixed-size group sections
 * laid out forward from PageGetContents(). The number of valid
 * entries (which may be < ngroups * 32 for the last group) lives in
 * opaque->entry_count as for the other formats; ngroups is
 * ceil(entry_count / 32).
 *
 * Per-group section, in this order so that group index g maps to a
 * single contiguous span computable via g * group_bytes:
 *
 *   BlockNumber child_blkno[32]
 *   float       f_add[32]
 *   float       f_rescale[32]
 *   float       f_error[32]
 *   uint8_t     codes[nsq_pairs * 32]
 *
 * Centroid pages do NOT need leaf-mode pt_centroid (that lives on
 * the first posting page; see prism_posting_pt_centroid). The leaf bit
 * in opaque flags determines whether child_blkno[*] points to
 * posting heads or nested centroid pages.
 * ---------------------------------------------------------------- */

static inline uint32_t
prism_centroid_fastscan_group_bytes(Dimension dim)
{
	return (uint32_t)(VS_FASTSCAN_GROUP * sizeof(BlockNumber) +
					  VS_FASTSCAN_GROUP * 3 * sizeof(float) +
					  VS_FASTSCAN_GROUP_BYTES(dim));
}

static inline uint32_t
prism_centroid_fastscan_max_groups(Dimension dim)
{
	return (uint32_t)PRISM_CENTROID_PAGE_USABLE /
		   prism_centroid_fastscan_group_bytes(dim);
}

/* Group accessors: each takes the page contents pointer + group index. */
static inline char *
prism_centroid_fastscan_group_base(char *content, uint32_t g, Dimension dim)
{
	return content + (size_t)g * prism_centroid_fastscan_group_bytes(dim);
}

static inline BlockNumber *
prism_centroid_fastscan_group_child(char *content, uint32_t g, Dimension dim)
{
	return (BlockNumber *)prism_centroid_fastscan_group_base(content, g, dim);
}

static inline float *
prism_centroid_fastscan_group_f_add(char *content, uint32_t g, Dimension dim)
{
	return (float *)(prism_centroid_fastscan_group_base(content, g, dim) +
					 VS_FASTSCAN_GROUP * sizeof(BlockNumber));
}

static inline float *
prism_centroid_fastscan_group_f_rescale(
		char *content, uint32_t g, Dimension dim)
{
	return prism_centroid_fastscan_group_f_add(content, g, dim) +
		   VS_FASTSCAN_GROUP;
}

static inline float *
prism_centroid_fastscan_group_f_error(char *content, uint32_t g, Dimension dim)
{
	return prism_centroid_fastscan_group_f_rescale(content, g, dim) +
		   VS_FASTSCAN_GROUP;
}

static inline uint8_t *
prism_centroid_fastscan_group_codes(char *content, uint32_t g, Dimension dim)
{
	return (uint8_t *)(prism_centroid_fastscan_group_f_error(content, g, dim) +
					   VS_FASTSCAN_GROUP);
}

/* Maximum entries per leaf page (includes pt_centroid per entry) */
static inline uint32_t
prism_centroid_max_leaf_entries_fmt(Dimension dim, PrismCentroidFormat fmt)
{
	return (uint32_t)(PRISM_CENTROID_PAGE_USABLE /
					  prism_centroid_leaf_entry_bytes_fmt(dim, fmt));
}

/* Backward-compatible wrappers (default to RaBitQ format) */
static inline uint32_t
prism_centroid_entry_bytes(Dimension dim)
{
	return prism_centroid_entry_bytes_fmt(dim, PRISM_CENTROID_FMT_RABITQ);
}

static inline uint32_t
prism_centroid_max_entries(Dimension dim)
{
	return prism_centroid_max_entries_fmt(dim, PRISM_CENTROID_FMT_RABITQ);
}

/* ----------------------------------------------------------------
 * Page access helpers
 *
 * Metadata grows forward from page header.
 * Vector data grows backward from the opaque area.
 * ---------------------------------------------------------------- */

/* Opaque area via PG-standard PageGetSpecialPointer */
#define PRISM_CENTROID_OPAQUE(page) \
	((PrismCentroidPageOpaque *)PageGetSpecialPointer(page))

/* Data format stored in the page */
static inline PrismCentroidFormat
prism_centroid_page_format(Page page)
{
	return (PrismCentroidFormat)(PRISM_CENTROID_OPAQUE(page)->flags &
								 PRISM_CENTROID_FMT_MASK);
}

/*
 * Get mutable pointer to the i-th metadata entry (write path).
 * Uses byte-offset arithmetic since meta size is format-dependent.
 */
static inline PrismCentroidEntryMeta *
prism_centroid_meta_mut(Page page, uint32_t index)
{
	uint32_t meta_size = prism_centroid_meta_size(
			prism_centroid_page_format(page));
	return (PrismCentroidEntryMeta *)((char *)PageGetContents(page) +
									  index * meta_size);
}

/* Get pointer to the i-th metadata entry (read-only) */
static inline const PrismCentroidEntryMeta *
prism_centroid_meta(const Page page, uint32_t index)
{
	return prism_centroid_meta_mut(page, index);
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
prism_centroid_entry_data(const Page page, uint32_t index, Dimension dim)
{
	PrismCentroidFormat fmt		  = prism_centroid_page_format(page);
	uint32_t			data_size = prism_centroid_data_size(dim, fmt);
	return (const void *)(PageGetSpecialPointer(page) -
						  (size_t)(index + 1) * data_size);
}

/* Typed accessors for routing data (at start of data region) */
static inline const RaBitQData *
prism_centroid_data(const Page page, uint32_t index, Dimension dim)
{
	return (const RaBitQData *)prism_centroid_entry_data(page, index, dim);
}

static inline const float *
prism_centroid_float_data(const Page page, uint32_t index, Dimension dim)
{
	return (const float *)prism_centroid_entry_data(page, index, dim);
}

static inline const half *
prism_centroid_half_data(const Page page, uint32_t index, Dimension dim)
{
	return (const half *)prism_centroid_entry_data(page, index, dim);
}

/* ----------------------------------------------------------------
 * Page operations
 * ---------------------------------------------------------------- */

/*
 * Initialize a centroid page with explicit data format.
 * Zeroes the page, sets up the opaque area with format flag.
 */
void prism_centroid_page_init_fmt(
		Page page, uint8_t level, PrismCentroidFormat fmt);

/* Backward-compatible init (defaults to RaBitQ format) */
static inline void
prism_centroid_page_init(Page page, uint8_t level)
{
	prism_centroid_page_init_fmt(page, level, PRISM_CENTROID_FMT_RABITQ);
}

/*
 * Reserve space for a centroid entry. Writes metadata forward,
 * reserves data space backward, returns a writable pointer to
 * the data region. The caller writes entry data directly into
 * the returned pointer (data_size bytes).
 *
 * Returns NULL if the page has no room.
 */
void *prism_centroid_page_add_entry_begin(
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
bool prism_centroid_page_add_entry(
		Page		page,
		Dimension	dim,
		BlockNumber child_blkno,
		uint16_t	child_count,
		uint16_t	flags,
		const void *data);

/*
 * Overwrite an existing entry in place: replace its child_blkno and its
 * routing data (data must be data_size bytes in the page's format), leaving
 * child_count/flags, entry_count, and the page layout untouched. Used by the
 * incremental posting-list split to repoint a leaf entry at its first child
 * list and update its routing centroid.
 */
void prism_centroid_page_overwrite_entry(
		Page		page,
		Dimension	dim,
		uint32_t	index,
		BlockNumber child_blkno,
		const void *data);

/* Backward-compatible add (RaBitQ-typed parameter) */
static inline bool
prism_centroid_page_add(
		Page			  page,
		Dimension		  dim,
		BlockNumber		  child_blkno,
		uint16_t		  child_count,
		uint16_t		  flags,
		const RaBitQData *data)
{
	return prism_centroid_page_add_entry(
			page, dim, child_blkno, child_count, flags, data);
}

/*
 * Where the metadata region ends on a page holding `nentries`.
 *
 * Derived from entry_count rather than read from pd_lower, because pd_lower
 * does not survive a page write outside index build: prism keeps its data
 * in the region PostgreSQL treats as the free hole, and the storage layer
 * covers that hole (pd_lower = pd_upper) so a full-page image preserves it.
 * Every other page kind is indifferent -- centroid pages are the only ones
 * that grow a forward region -- so entry_count, which lives in the opaque
 * area and does survive, is the authoritative cursor. It is also what the
 * data-region reader already uses (prism_centroid_entry_data indexes off
 * pd_special), so the two regions stay consistent.
 */
static inline size_t
prism_centroid_meta_end(Page page, uint32_t nentries)
{
	PrismCentroidFormat fmt = prism_centroid_page_format(page);
	return (size_t)SizeOfPageHeaderData +
		   (size_t)nentries * prism_centroid_meta_size(fmt);
}

/*
 * Check if a page has room for one more entry.
 * Reads data format from the page to determine entry size.
 */
static inline bool
prism_centroid_page_has_room(Page page, Dimension dim, bool is_leaf)
{
	PageHeader			header	 = (PageHeader)page;
	PrismCentroidFormat fmt		 = prism_centroid_page_format(page);
	size_t				need_fwd = prism_centroid_meta_size(fmt);
	size_t need_bwd = is_leaf ? prism_centroid_leaf_data_size(dim, fmt)
							  : prism_centroid_data_size(dim, fmt);
	size_t lower	= prism_centroid_meta_end(
			   page, PRISM_CENTROID_OPAQUE(page)->entry_count);

	return lower + need_fwd + need_bwd <= header->pd_upper;
}

#endif /* PRISM_CENTROID_PAGE_H */
