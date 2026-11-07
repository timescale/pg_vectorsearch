/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * meta.h - Metadata page layout for prism index
 *
 * Block 0 of every prism index stores a PrismMetaPage in the
 * page special area. It records index parameters (dimension, tree
 * depth, centroid format, distance metric, RaBitQ seed) and the
 * global mean vector used for RaBitQ query preparation.
 */

#ifndef PRISM_META_H
#define PRISM_META_H

#include <postgres.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <storage/bufpage.h>
#pragma GCC diagnostic pop

#include "core/types.h"
#include "index/index_base.h"

/*
 * "PRI" + a format-version byte. Bump the low byte on any incompatible
 * metapage/layout change so an index built by an older format is rejected at
 * open rather than silently misread. v2 added PrismMetaPage.first_posting,
 * which shifted the struct layout; v3 removed the unused indexed-row count,
 * which shifted it again; v4 grew the fastscan group section by one
 * tombstone_mask word, which moves every group after the first on a page --
 * a posting page format change, not a metapage one, but this is the only
 * version gate the index has.
 *
 * The version is per index, chosen by its dimension. v7 is the current
 * format, written for the spilled layout (prism_meta_layout_spilled), which
 * an older binary rejects at open instead of reading a reference past its
 * page. An inline-layout index still gets v6, so existing indexes need no
 * REINDEX and still open under an older binary.
 */
#define PRISM_META_MAGIC_COMPAT_V6 ((uint32_t)0x50524906) /* "PRI\x06" */
#define PRISM_META_MAGIC		   ((uint32_t)0x50524907) /* "PRI\x07" */

static inline bool
prism_meta_layout_spilled(Dimension dim)
{
	return prism_posting_spilled(dim) || prism_global_mean_pages(dim) > 0;
}

static inline uint32_t
prism_meta_magic_for(Dimension dim)
{
	return prism_meta_layout_spilled(dim) ? PRISM_META_MAGIC
										  : PRISM_META_MAGIC_COMPAT_V6;
}

/* The magic an index must carry is the one its dimension implies; any other
 * value is an index written by some other format. */
static inline bool
prism_meta_magic_valid(uint32_t magic, Dimension dim)
{
	return magic == prism_meta_magic_for(dim);
}

/* Metadata flags */
#define PRISM_META_FLAG_FASTSCAN 0x01

typedef struct PrismMetaPage
{
	uint32_t	magic;			 /* PRISM_META_MAGIC or _COMPAT_V6 */
	Dimension	dim;			 /* vector dimension */
	uint8_t		nlevels;		 /* centroid tree depth */
	uint8_t		centroid_format; /* PrismCentroidFormat */
	BlockNumber first_centroid;	 /* root centroid page */
	BlockNumber first_posting;	 /* first posting page (one past the last
								  * centroid page); lets VACUUM skip the whole
								  * centroid region without scanning it */
	uint32_t nlist;				 /* number of leaf centroids */
	/*
	 * Centroid pages reachable from first_centroid. Maintained rather than
	 * derived: the region a build reserves is [prism_first_centroid_blkno,
	 * first_posting), but a split with no room on a level-0 page extends the
	 * relation and chains the new page past the posting region, so a block
	 * range stops measuring it. prism_rebalance folds each split's additions
	 * back in here.
	 */
	uint32_t ncentroid_pages;
	uint8_t	 metric;  /* DistanceMetric */
	uint8_t	 fan_out; /* children per tree node */
	uint8_t	 flags;	  /* PRISM_META_FLAG_* */
	uint8_t	 reserved;
	uint64_t rabitq_seed; /* seed for RaBitQ params */
						  /* Global mean vector stored inline after struct, up
						   * to PRISM_GLOBAL_MEAN_MAX_INLINE_DIM; past it the
						   * mean lives in the reference pages at
						   * PRISM_GLOBAL_MEAN_BLKNO (see index_base.h). */
} PrismMetaPage;

#define PRISM_META_FIXED_SIZE \
	(offsetof(PrismMetaPage, rabitq_seed) + sizeof(uint64_t))

static inline bool
prism_meta_mean_inline(Dimension dim)
{
	return prism_global_mean_pages(dim) == 0;
}

/* Special-area size: the struct plus the inline global mean where it fits */
#define PRISM_META_SIZE(dim)                                                 \
	(MAXALIGN(                                                               \
			PRISM_META_FIXED_SIZE + (prism_meta_mean_inline(dim)             \
											 ? (size_t)(dim) * sizeof(float) \
											 : 0)))

/*
 * The inline threshold shared with the standalone layout is exactly the
 * metapage's capacity: one more dimension would not fit the page.
 */
StaticAssertDecl(
		MAXALIGN(
				PRISM_META_FIXED_SIZE +
				(size_t)PRISM_GLOBAL_MEAN_MAX_INLINE_DIM * sizeof(float)) <=
				BLCKSZ - MAXALIGN(SizeOfPageHeaderData),
		"the inline mean must fit the metapage at the threshold");
StaticAssertDecl(
		MAXALIGN(
				PRISM_META_FIXED_SIZE +
				(size_t)(PRISM_GLOBAL_MEAN_MAX_INLINE_DIM + 1) *
						sizeof(float)) >
				BLCKSZ - MAXALIGN(SizeOfPageHeaderData),
		"the inline-mean threshold is tight");

/* Access the inline global mean vector after the struct (inline layout) */
static inline float *
prism_meta_global_mean(PrismMetaPage *meta)
{
	return (float *)((char *)meta + PRISM_META_FIXED_SIZE);
}

static inline const float *
prism_meta_global_mean_const(const PrismMetaPage *meta)
{
	return (const float *)((const char *)meta + PRISM_META_FIXED_SIZE);
}

#endif /* PRISM_META_H */
