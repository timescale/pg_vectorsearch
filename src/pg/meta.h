/*
 * meta.h - Metadata page layout for prism index
 *
 * Block 0 of every prism index stores a PrismMetaPage in the
 * page special area. It records index parameters (dimension, tree
 * depth, centroid format, distance metric, RaBitQ seed) and the
 * global mean vector used for RaBitQ query preparation.
 */

#ifndef MKT_META_H
#define MKT_META_H

#include <postgres.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <storage/bufpage.h>
#pragma GCC diagnostic pop

#include "core/types.h"

/*
 * "MKT" + a format-version byte. Bump the low byte on any incompatible
 * metapage/layout change so an index built by an older format is rejected at
 * open rather than silently misread. v2 added PrismMetaPage.first_posting,
 * which shifted the struct layout; v3 removed the unused indexed-row count,
 * which shifted it again.
 */
#define MKT_META_MAGIC ((uint32_t)0x4D4B5404) /* "MKT\x04" */

/* Metadata flags */
#define MKT_META_FLAG_FASTSCAN 0x01

typedef struct PrismMetaPage
{
	uint32_t	magic;			 /* MKT_META_MAGIC */
	Dimension	dim;			 /* vector dimension */
	uint8_t		nlevels;		 /* centroid tree depth */
	uint8_t		centroid_format; /* MktCentroidFormat */
	BlockNumber first_centroid;	 /* root centroid page */
	BlockNumber first_posting;	 /* first posting page (one past the last
								  * centroid page); lets VACUUM skip the whole
								  * centroid region without scanning it */
	uint32_t nlist;				 /* number of leaf centroids */
	/*
	 * Centroid pages reachable from first_centroid. Maintained rather than
	 * derived: the region a build reserves is [MKT_FIRST_CENTROID_BLKNO,
	 * first_posting), but a split with no room on a level-0 page extends the
	 * relation and chains the new page past the posting region, so a block
	 * range stops measuring it. prism.rebalance folds each split's additions
	 * back in here.
	 */
	uint32_t ncentroid_pages;
	uint8_t	 metric;  /* DistanceMetric */
	uint8_t	 fan_out; /* children per tree node */
	uint8_t	 flags;	  /* MKT_META_FLAG_* */
	uint8_t	 reserved;
	uint64_t rabitq_seed; /* seed for RaBitQ params */
						  /* Global mean vector stored inline after struct */
} PrismMetaPage;

/* Total special-area size including inline global mean */
#define MKT_META_SIZE(dim)                                            \
	(MAXALIGN(                                                        \
			offsetof(PrismMetaPage, rabitq_seed) + sizeof(uint64_t) + \
			(size_t)(dim) * sizeof(float)))

/* Access the inline global mean vector after the struct */
static inline float *
prism_meta_global_mean(PrismMetaPage *meta)
{
	return (float *)((char *)meta + offsetof(PrismMetaPage, rabitq_seed) +
					 sizeof(uint64_t));
}

static inline const float *
prism_meta_global_mean_const(const PrismMetaPage *meta)
{
	return (const float *)((const char *)meta +
						   offsetof(PrismMetaPage, rabitq_seed) +
						   sizeof(uint64_t));
}

#endif /* MKT_META_H */
