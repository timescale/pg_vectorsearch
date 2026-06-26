/*
 * index_base.h - Common index descriptor for search
 *
 * MktIndexBase contains the fields needed by the shared search
 * path. Standalone embeds it in MktIndex; PG populates it from
 * the meta page and amcache.
 */

#ifndef MKT_INDEX_BASE_H
#define MKT_INDEX_BASE_H

#include "index/centroid_compact.h"
#include "index/centroid_page.h"
#include "index/storage.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

typedef struct MktIndexBase
{
	/* RaBitQ (must outlive the search context) */
	RaBitQParams *params;
	float		 *pt_global_mean;
	uint64_t	  rabitq_seed;

	/* PG: both point to the same MktannStorage (one index relation).
	 * Standalone: separate ArrayPageStorage for centroids vs postings. */
	MktStorage *centroid_storage;
	MktStorage *posting_storage;

	/* Optional compact in-memory centroid source (FASTSCAN only). When set,
	 * the beam search reads node groups from here instead of centroid pages.
	 * NULL = read centroids via centroid_storage. */
	MktCentroidCompact *centroid_compact;

	/* Non-NULL for inline page access (standalone postings) */
	char *page_base;

	/* Index metadata */
	Dimension		  dim;
	uint8_t			  nlevels;
	BlockNumber		  first_centroid;
	DistanceMetric	  metric;
	MktCentroidFormat centroid_format;
	int				  fastscan; /* 0=off, 8=uint8, 16=uint16 hacc */
	float centroid_error_scale; /* scales centroid pruning error (1=default) */
	float centroid_beam_scale;	/* intermediate beam width / nprobe
								   (0.25=default) */
} MktIndexBase;

static inline RaBitQParams *
mkt_index_ensure_rabitq(MktIndexBase *idx)
{
	if (idx->params == NULL)
	{
		idx->params = mkt_rabitq_create(idx->dim, idx->rabitq_seed);
		if (idx->pt_global_mean != NULL)
			mkt_rabitq_rotate(
					idx->params, idx->pt_global_mean, idx->pt_global_mean);
	}
	return idx->params;
}

#endif /* MKT_INDEX_BASE_H */
