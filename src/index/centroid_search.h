/*
 * centroid_search.h - Beam search over centroid tree
 *
 * Level-by-level descent with beam search, reading centroid pages
 * via MktStorage. The same function works in standalone and
 * PG mode — only the storage implementation differs.
 *
 * Algorithm:
 *   1. Read root centroid page(s), score ALL centroids
 *   2. Keep top beam_width candidates
 *   3. For each intermediate level, expand winners via child_blkno,
 *      score all children, keep top beam_width
 *   4. At leaf level, keep top nprobe candidates
 *   5. Return posting list heads + medoid TIDs for leaf winners
 */

#ifndef MKT_CENTROID_SEARCH_H
#define MKT_CENTROID_SEARCH_H

#include <stdint.h>

#include "index/centroid_page.h"
#include "index/storage.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Search result entry
 * ---------------------------------------------------------------- */
typedef struct MktCentroidResult
{
	BlockNumber		posting_head; /* head of posting list */
	ItemPointerData medoid_tid;	  /* heap TID of medoid vector */
	Distance		distance;	  /* estimated distance to query */
} MktCentroidResult;

/* ----------------------------------------------------------------
 * Search state
 * ---------------------------------------------------------------- */
typedef struct MktCentroidSearchState
{
	const RaBitQQueryState *qstate;	 /* query transformed vs global mean */
	const MktStorage	   *storage; /* page and vector I/O */
	uint32_t				beam_width;
	uint32_t				nprobe;
	Dimension				dim;
} MktCentroidSearchState;

/* ----------------------------------------------------------------
 * Beam search API
 *
 * Searches the centroid tree starting from first_centroid_blkno.
 * Returns up to nprobe leaf centroids in results[], sorted by
 * estimated distance (ascending). Returns actual count written.
 *
 * results[] must have space for at least state->nprobe entries.
 * ---------------------------------------------------------------- */
uint32_t mkt_centroid_beam_search(
		const MktCentroidSearchState *state,
		BlockNumber					  first_centroid_blkno,
		uint8_t						  nlevels,
		MktCentroidResult			 *results);

#endif /* MKT_CENTROID_SEARCH_H */
