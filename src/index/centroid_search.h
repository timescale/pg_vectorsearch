/*
 * centroid_search.h - Beam search over centroid tree
 *
 * Level-by-level descent with beam search, reading centroid pages
 * via MktStorage. The same function works in standalone and
 * PG mode — only the storage implementation differs.
 *
 * Supports all centroid page formats (RaBitQ, float32, float16).
 * RaBitQ pages use approximate distance with error bounds; float
 * and half pages use exact L2 distance (error = 0).
 *
 * Algorithm:
 *   1. Read root centroid page(s), score ALL centroids
 *   2. Keep top beam_width candidates (error-bound-aware)
 *   3. For each intermediate level, expand winners via child_blkno,
 *      score all children, keep top beam_width
 *   4. At leaf level, keep top nprobe candidates
 *   5. Return posting list heads + distances for leaf winners
 */

#ifndef MKT_CENTROID_SEARCH_H
#define MKT_CENTROID_SEARCH_H

#include <stdint.h>

#include "index/centroid_compact.h"
#include "index/centroid_page.h"
#include "index/storage.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Search result entry
 * ---------------------------------------------------------------- */
typedef struct MktCentroidResult
{
	BlockNumber posting_head; /* head of posting list */
	Distance	distance;	  /* estimated distance to query */
	Distance	error;		  /* symmetric error margin */
} MktCentroidResult;

/* ----------------------------------------------------------------
 * Per-scan scratch buffers
 *
 * Bundles the candidate-buffer pair and the score-page scratch that
 * mkt_centroid_beam_search would otherwise palloc on every call. The
 * caller (MktQueryState / MktQueryCtx) allocates this once at scan
 * setup and passes it in through MktCentroidSearchState.
 *
 * Sizing:
 *   buf_a, buf_b: cand_cap entries each. Caller picks cand_cap so it
 *     fits the worst-case max_per_page * max_beam_width.
 *   f_add..symmetric_scratch: max_per_page entries each (the largest
 *     centroid page entry count for this dim).
 * ---------------------------------------------------------------- */
struct MktCentroidScratch;
typedef struct MktCentroidScratch MktCentroidScratch;

/*
 * Allocate scratch sized for searches up to max_beam_width / nprobe
 * candidates per level. Returns NULL on failure. cleanup releases
 * the underlying buffers.
 */
MktCentroidScratch		*
mkt_centroid_scratch_create(Dimension dim, uint32_t max_beam_width);
void mkt_centroid_scratch_free(MktCentroidScratch *scratch);

/* ----------------------------------------------------------------
 * Search state
 * ---------------------------------------------------------------- */
typedef struct MktCentroidSearchState
{
	const RaBitQQueryState *qstate;	 /* query for RaBitQ pages */
	const float			   *query;	 /* raw query for float/half pages */
	MktStorage			   *storage; /* page and vector I/O */
	/* Optional compact FASTSCAN centroid source. When non-NULL, nodes are
	 * scored from this contiguous in-memory layout instead of pages. */
	const MktCentroidCompact *compact;
	uint32_t	   beam_width; /* candidates per level (>= nprobe) */
	uint32_t	   nprobe;	   /* target leaf count */
	Dimension	   dim;
	DistanceMetric metric; /* distance metric for routing */
	float error_scale;	   /* scales pruning error bound (1=default, 0=drop) */
	/* Pre-allocated scratch. Must be non-NULL and sized for at least
	 * this state's beam_width / nprobe. */
	MktCentroidScratch *scratch;
} MktCentroidSearchState;

/* ----------------------------------------------------------------
 * Search statistics (optional output)
 * ---------------------------------------------------------------- */
typedef struct MktCentroidSearchStats
{
	uint64_t dist_calcs; /* approximate distance computations */
	uint32_t pages_read; /* centroid pages read */
	/* Fine-grained wall time in ns (diagnostic; 0 when not measured). */
	uint64_t rotation_ns; /* query rotation (P^T * query) */
	uint64_t lut_ns;	  /* per-query fastscan LUT build */
	uint64_t pageread_ns; /* centroid page read/release */
	uint64_t score_ns;	  /* score_page total (incl. lut_ns) */
} MktCentroidSearchStats;

/* ----------------------------------------------------------------
 * Beam search API
 *
 * Searches the centroid tree starting from first_centroid_blkno.
 * Returns leaf centroids in results[], sorted by estimated
 * distance (ascending). Returns actual count written.
 *
 * Returns up to nprobe leaf centroids. Error-bound-aware
 * selection at intermediate levels may keep more candidates
 * than beam_width to avoid pruning uncertain results, but the
 * final output is capped at nprobe.
 *
 * results[] must have space for at least nprobe entries.
 * stats is optional (may be NULL). If provided, counters are
 * accumulated (not reset) so the caller can aggregate across
 * multiple calls.
 * ---------------------------------------------------------------- */
/*
 * centroid_vecs is an optional output buffer (may be NULL). When
 * non-NULL, the function copies the leaf centroid vector for each
 * result into centroid_vecs[i * dim .. (i+1) * dim - 1]. The
 * buffer must hold at least nprobe * dim floats. Half-
 * precision vectors are converted to float32. Used by the PG scan
 * path to obtain centroid reference vectors for float/half
 * centroid formats.
 */
uint32_t mkt_centroid_beam_search(
		const MktCentroidSearchState *state,
		BlockNumber					  first_centroid_blkno,
		uint8_t						  nlevels,
		MktCentroidResult			 *results,
		float						 *centroid_vecs,
		MktCentroidSearchStats		 *stats);

#endif /* MKT_CENTROID_SEARCH_H */
