/*
 * posting_build.h - Streaming posting list builder
 *
 * Encodes vectors with IVF residual encoding and writes posting pages
 * one entry at a time, using O(1) memory per cluster. Each builder
 * embeds an in-memory page buffer (BLCKSZ bytes) that is flushed
 * to storage when full, plus a small RaBitQ encode scratch buffer.
 *
 * Follows the centroid_build.h pattern: a shared builder in
 * src/index/ that works with MktStorage, usable from both the
 * benchmarker and the PG extension.
 *
 * Parallelism: each MktPostingBuilder is fully independent with no
 * shared mutable state. For parallel index build, each worker gets
 * its own builder instance + storage backend.
 */

#ifndef MKT_POSTING_BUILD_H
#define MKT_POSTING_BUILD_H

#include "index/posting_page.h"
#include "index/storage.h"
#include "mkt_types.h"
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
	const float		   *centroid; /* [dim] reference vector */

	BlockNumber head_blkno; /* first page in chain */
	BlockNumber prev_blkno; /* previous page (for linking) */
	bool		is_first;	/* next page gets FIRST flag */

	char mem_page[BLCKSZ]; /* in-memory page buffer */
	bool page_dirty;	   /* has entries since last flush */

	/* Reserved contiguous block range (set via _set_reserve) */
	BlockNumber reserve_start; /* first reserved block */
	uint32_t	reserve_count; /* total reserved blocks */
	uint32_t	reserve_used;  /* blocks consumed so far */

	/* RaBitQ encode scratch buffer (~dim/8 + 12 bytes) */
	RaBitQData *enc_buf;
} MktPostingBuilder;

/*
 * Initialize builder state. Allocates a small RaBitQ encode buffer.
 */
void mkt_posting_builder_init(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		const float		   *centroid);

/*
 * Set a reserved contiguous block range for this builder.
 * Pages are flushed to reserved blocks first for sequential layout;
 * overflow falls back to new_page. Optional — without this, all
 * flushes go through new_page.
 */
void mkt_posting_builder_set_reserve(
		MktPostingBuilder *builder, BlockNumber start, uint32_t count);

/*
 * Add one vector. Encodes with IVF residual from cluster centroid,
 * writes to in-memory page, flushes to storage if full. O(1) per call.
 */
void mkt_posting_builder_add(
		MktPostingBuilder *builder,
		VectorRef		   vec,
		BlockNumber		   block,
		OffsetNumber	   offset);

/*
 * Finish: flush last page to storage, return head block number.
 * Returns InvalidBlockNumber if no entries were added.
 */
BlockNumber mkt_posting_builder_finish(MktPostingBuilder *builder);

/*
 * Cleanup: free the encode buffer.
 */
void mkt_posting_builder_cleanup(MktPostingBuilder *builder);

#endif /* MKT_POSTING_BUILD_H */
