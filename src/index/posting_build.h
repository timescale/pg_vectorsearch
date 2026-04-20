/*
 * posting_build.h - Streaming posting list builder
 *
 * Builds posting list page chains from a stream of vectors.
 * Each vector is RaBitQ-encoded and written to an in-memory page
 * buffer. Full pages are flushed to storage and linked into a
 * chain. O(1) memory per cluster regardless of cluster size.
 *
 * Follows the centroid_build.h pattern: a shared builder in
 * src/index/ that works with MktStorage, usable from both the
 * standalone benchmarker and the PG extension.
 *
 * Usage:
 *   MktPostingBuilder builder;
 *   mkt_posting_builder_init(&builder, storage, params, dim,
 *                            cluster_id, centroid);
 *   for each vector in cluster:
 *       mkt_posting_builder_add(&builder, tid, vector);
 *   BlockNumber head = mkt_posting_builder_finish(&builder);
 *   mkt_posting_builder_cleanup(&builder);
 */

#ifndef MKT_POSTING_BUILD_H
#define MKT_POSTING_BUILD_H

#include "index/posting_page.h"
#include "index/storage.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Builder state — O(1) memory per cluster via in-memory page buffer
 * ---------------------------------------------------------------- */

typedef struct MktPostingBuilder
{
	MktStorage		   *storage;
	const RaBitQParams *params;
	Dimension			dim;
	uint32_t			cluster_id;
	const float		   *centroid; /* [dim] for IVF encoding */
	const float *pt_centroid; /* [dim] P^T * centroid, written to first page */

	BlockNumber head_blkno; /* first page in chain */
	BlockNumber prev_blkno; /* previous page (for linking) */
	bool		is_first;	/* next page gets FIRST flag */

	char mem_page[BLCKSZ]
			__attribute__((aligned(8))); /* in-memory page buffer */
	bool page_dirty;					 /* has entries since last flush */

	/* Reserved contiguous block range (set via _set_reserve) */
	BlockNumber reserve_start; /* first reserved block */
	uint32_t	reserve_count; /* total reserved blocks */
	uint32_t	reserve_used;  /* blocks consumed so far */

	/* RaBitQ encode scratch buffer (~dim/8 + 12 bytes) */
	RaBitQData *enc_buf;
} MktPostingBuilder;

/*
 * Initialize builder state. Allocates a small RaBitQ encode buffer.
 * centroid is borrowed (not copied) — must remain valid until finish.
 */
void mkt_posting_builder_init(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		const float		   *centroid,
		const float		   *pt_centroid);

/*
 * Set a reserved contiguous block range for this builder.
 * Pages are flushed to reserved blocks first for sequential layout;
 * overflow falls back to new_page. Optional — without this, all
 * flushes go through new_page.
 */
void mkt_posting_builder_set_reserve(
		MktPostingBuilder *builder, BlockNumber start, uint32_t count);

/*
 * Add a vector to the posting list. Encodes with RaBitQ relative
 * to the cluster centroid. The TID identifies the source tuple —
 * in PG this is a real heap TID; in standalone, pack a vector_id
 * via mkt_posting_set_vector_id().
 */
void mkt_posting_builder_add(
		MktPostingBuilder *builder, ItemPointerData tid, const float *vector);

/*
 * Flush the last page and return the head block number.
 * Returns InvalidBlockNumber if no vectors were added.
 */
BlockNumber mkt_posting_builder_finish(MktPostingBuilder *builder);

/*
 * Free internal scratch buffers.
 */
void mkt_posting_builder_cleanup(MktPostingBuilder *builder);

/* ----------------------------------------------------------------
 * Flat builder — one buffer per cluster (standalone benchmark)
 *
 * Usage:
 *   MktFlatPostingBuilder builder;
 *   mkt_flat_posting_builder_init(&builder, params, dim,
 *                                 cluster_id, centroid, count);
 *   for each vector in cluster:
 *       mkt_flat_posting_builder_add(&builder, tid, vector);
 *   char *page = mkt_flat_posting_builder_finish(&builder);
 *   mkt_flat_posting_builder_cleanup(&builder);
 * ---------------------------------------------------------------- */

typedef struct MktFlatPostingBuilder
{
	const RaBitQParams *params;
	Dimension			dim;
	uint32_t			cluster_id;
	const float		   *centroid;

	char	*buf; /* flat page buffer */
	uint32_t max_entries;

	RaBitQData *enc_buf; /* reusable encode scratch */
} MktFlatPostingBuilder;

void mkt_flat_posting_builder_init(
		MktFlatPostingBuilder *builder,
		const RaBitQParams	  *params,
		Dimension			   dim,
		uint32_t			   cluster_id,
		const float			  *centroid,
		uint32_t			   count);

void mkt_flat_posting_builder_add(
		MktFlatPostingBuilder *builder,
		ItemPointerData		   tid,
		const float			  *vector);

/*
 * Return the flat page buffer. Owned by the memory context
 * that was current during init.
 */
char *mkt_flat_posting_builder_finish(MktFlatPostingBuilder *builder);

void mkt_flat_posting_builder_cleanup(MktFlatPostingBuilder *builder);

#endif /* MKT_POSTING_BUILD_H */
