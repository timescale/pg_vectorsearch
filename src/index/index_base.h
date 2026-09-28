/*
 * index_base.h - Common index descriptor for search
 *
 * PrismIndexBase contains the fields needed by the shared search
 * path. Standalone embeds it in PrismIndex; PG populates it from
 * the meta page and amcache.
 */

#ifndef PRISM_INDEX_BASE_H
#define PRISM_INDEX_BASE_H

#include "core/types.h"
#include "index/centroid_page.h"
#include "index/storage.h"
#include "quant/rabitq.h"

/*
 * Block 0 holds the metapage, so the centroid region a build reserves
 * starts here and runs to first_posting.
 */
#define PRISM_FIRST_CENTROID_BLKNO 1

/*
 * Posting pages the scan cost estimate prices, and what prism_index_settings
 * reports as posting_pages.
 *
 * Block 0 is the metapage and the centroid region a build reserves starts at
 * PRISM_FIRST_CENTROID_BLKNO, so on a freshly built index the posting pages
 * are everything from first_posting onward. That range stops measuring them
 * once a split overflows a level-0 centroid page: the split extends the
 * relation and chains the new centroid page past the posting region, so the
 * page is both counted by ncentroid_pages and, to a range measurement, a
 * posting page. Taking the maintained count out of the relation's size
 * instead is layout-independent and leaves it in one term only.
 *
 * num_pages is the relation's current size, not pg_class.relpages: both
 * callers read it straight from the relation, the planner included, so the
 * count never lags a statistics update.
 *
 * An upper bound rather than an exact count: free and new pages from
 * extension slack are included, the same over-count the estimate already
 * documents for a bloated index.
 */
static inline double
prism_index_posting_pages(double num_pages, double ncentroid_pages)
{
	double n = num_pages - (double)PRISM_FIRST_CENTROID_BLKNO -
			   ncentroid_pages;

	return n < 1.0 ? 1.0 : n;
}

typedef struct PrismIndexBase
{
	/* RaBitQ (must outlive the search context) */
	RaBitQParams *params;
	float		 *pt_global_mean;
	uint64_t	  rabitq_seed;

	/* PG: both point to the same VsPgStorage (one index relation).
	 * Standalone: separate ArrayPageStorage for centroids vs postings. */
	VsStorage *centroid_storage;
	VsStorage *posting_storage;

	/* Non-NULL for inline page access (standalone postings) */
	char *page_base;

	/* Index metadata */
	Dimension dim;
	uint8_t	  nlevels;
	/* Children per tree node; floors the intermediate beam width so the
	 * top-nprobe leaves stay reachable (0 = unknown, no floor). */
	uint8_t fan_out;
	/* Leaf (posting-list) count; when nprobe covers every leaf the
	 * intermediate beam keeps whole levels so no subtree is pruned
	 * (0 = unknown, no full-coverage floor). */
	uint32_t nlist;
	/*
	 * Root of the centroid tree, which every descent starts from -- NOT the
	 * first block of the centroid region, despite the name. Where the root
	 * sits inside the region depends on the build: a serial build writes the
	 * centroid pages post-order and the root lands last, a parallel one
	 * places it first. The region itself is always
	 * [PRISM_FIRST_CENTROID_BLKNO, first_posting).
	 */
	BlockNumber first_centroid;
	/*
	 * First block of the posting-head region; leaf c's head is
	 * first_posting + c. Fixed for the life of the index, since that formula
	 * is how every head is located, so nothing can be inserted below it.
	 */
	BlockNumber first_posting;
	/*
	 * Centroid pages reachable from first_centroid. Maintained rather than
	 * derived, because a split with no room on a level-0 page extends the
	 * relation and chains the new page past first_posting, after which the
	 * reserved block range no longer measures the count.
	 */
	uint32_t			ncentroid_pages;
	DistanceMetric		metric;
	PrismCentroidFormat centroid_format;
	int					fastscan; /* 0=off, 8=uint8, 16=uint16 hacc */
	float centroid_error_scale; /* scales centroid pruning error (1=default) */
	float centroid_beam_scale;	/* intermediate beam width / nprobe
								   (0.25=default) */
	/* Build-only: exact internal-node centroids for the build descent
	 * (see PrismExactInternalCentroids in centroid_search.h). NULL — the
	 * default everywhere the base is zero-initialized — keeps the
	 * estimated scoring; the query and insert paths never set it. */
	const struct PrismExactInternalCentroids *exact_internal;
} PrismIndexBase;

static inline RaBitQParams *
prism_index_ensure_rabitq(PrismIndexBase *idx)
{
	if (idx->params == NULL)
	{
		idx->params = vs_rabitq_create(idx->dim, idx->rabitq_seed);
		if (idx->pt_global_mean != NULL)
			vs_rabitq_rotate(
					idx->params, idx->pt_global_mean, idx->pt_global_mean);
	}
	return idx->params;
}

#endif /* PRISM_INDEX_BASE_H */
