/*
 * index_base.h - Common index descriptor for search
 *
 * MktIndexBase contains the fields needed by the shared search
 * path. Standalone embeds it in MktIndex; PG populates it from
 * the meta page and amcache.
 */

#ifndef MKT_INDEX_BASE_H
#define MKT_INDEX_BASE_H

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

	/* Non-NULL for inline page access (standalone postings) */
	char *page_base;

	/* Index metadata */
	Dimension		  dim;
	uint8_t			  nlevels;
	BlockNumber		  first_centroid;
	DistanceMetric	  metric;
	MktCentroidFormat centroid_format;
	int				  fastscan; /* 0=off, 8=uint8, 16=uint16 hacc */
	bool			  route_ip; /* route by <q,c> (keep centroid magnitude) */
	int				  centroid_rerank; /* two-stage: beam shortlist factor (0=off) */
	bool			  early_terminate; /* skip clusters that can't hold a closer NN */
	float			  term_radius;	   /* global max (1-cos(v,centroid)) for the bound */

	/* Two-stage rerank pt_centroid cache (PG only; NULL in standalone).
	 * Returns the cluster's stored centroid (P^T*c, dim floats) for a posting
	 * head block, backed by a per-backend cache so the rerank avoids a buffer
	 * pin per candidate. */
	void *pt_centroid_cache;
	const float *(*pt_centroid_fn)(void *cache, BlockNumber head_blkno);
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
