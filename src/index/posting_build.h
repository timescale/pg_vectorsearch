/*
 * posting_build.h - Streaming posting list builder
 *
 * Builds posting list page chains from a stream of vectors.
 * Supports two page formats (AoS and fastscan) via a page-ops
 * callback, and two entry modes: raw vectors (RaBitQ-encoded
 * internally) and pre-encoded data.
 *
 * Usage:
 *   MktPostingBuilder builder;
 *   mkt_posting_builder_init(&builder, storage, params, dim,
 *                            cluster_id, centroid, pt_centroid);
 *   // or: mkt_posting_builder_init_fastscan(...) for fastscan
 *   for each vector in cluster:
 *       mkt_posting_builder_add(&builder, tid, vector);
 *   BlockNumber head = mkt_posting_builder_finish(&builder);
 *   mkt_posting_builder_cleanup(&builder);
 */

#ifndef MKT_POSTING_BUILD_H
#define MKT_POSTING_BUILD_H

#include <stdatomic.h>

#include "index/posting_page.h"
#include "index/storage.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Page format ops — the only part that differs between formats
 *
 * write_entry: add one encoded entry to the in-memory page.
 *     Returns false if the page is full and needs flushing first.
 * reinit_page: re-initialize the page buffer after a flush.
 * finalize: called before the last flush (e.g., write partial
 *     fastscan group).
 * cleanup: free format-specific scratch buffers.
 * ---------------------------------------------------------------- */

struct MktPostingBuilder;

typedef struct MktPostingPageOps
{
	bool (*write_entry)(
			struct MktPostingBuilder *b,
			ItemPointerData			  tid,
			float					  f_add,
			float					  f_rescale,
			float					  f_error,
			const uint8_t			 *bits);
	void (*reinit_page)(struct MktPostingBuilder *b);
	void (*finalize)(struct MktPostingBuilder *b);
	void (*cleanup)(struct MktPostingBuilder *b);
} MktPostingPageOps;

/* Fastscan group staging buffer */
typedef struct FsGroupStage
{
	ItemPointerData tids[MKT_FASTSCAN_GROUP];
	float			f_add[MKT_FASTSCAN_GROUP];
	float			f_rescale[MKT_FASTSCAN_GROUP];
	float			f_error[MKT_FASTSCAN_GROUP];
	uint32_t		count;
} FsGroupStage;

/* ----------------------------------------------------------------
 * Unified builder state
 * ---------------------------------------------------------------- */

typedef struct MktPostingBuilder
{
	/* Common fields */
	MktStorage		   *storage;
	const RaBitQParams *params;
	Dimension			dim;
	uint32_t			cluster_id;
	const float		   *centroid;

	BlockNumber head_blkno;
	BlockNumber prev_blkno;
	bool		is_first;

	char mem_page[BLCKSZ] __attribute__((aligned(8)));
	bool page_dirty;

	BlockNumber		   reserve_start;
	uint32_t		   reserve_count;
	uint32_t		   reserve_used;
	_Atomic(uint32_t) *shared_reserve_next;
	BlockNumber		   fixed_first_blkno;

	RaBitQData *enc_buf;

	/* Page format dispatch */
	const MktPostingPageOps *page_ops;

	/* Format-specific state (only fastscan uses this) */
	struct
	{
		FsGroupStage grp;
		uint32_t	 groups_on_page;
		uint32_t	 max_groups;
		uint8_t		*bits_buf;
		uint8_t		*codes_buf;
	} fs;
} MktPostingBuilder;

/*
 * Initialize for AoS page format. Encodes vectors with RaBitQ.
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
 * Initialize for fastscan page format. When params and centroid
 * are non-NULL, use _add() to encode raw vectors. When both are
 * NULL, use _add_encoded() for pre-encoded data (AoS conversion).
 */
void mkt_posting_builder_init_fastscan(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		const float		   *centroid,
		const float		   *pt_centroid);

/*
 * Initialize continuation builder (no pt_centroid on first page).
 * Used by parallel workers that are not responsible for the first
 * page of a cluster's posting list.
 */
void mkt_posting_builder_init_continuation(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		const float		   *centroid);

void mkt_posting_builder_init_continuation_fastscan(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		const float		   *centroid);

void mkt_posting_builder_set_reserve(
		MktPostingBuilder *builder, BlockNumber start, uint32_t count);

/*
 * Shared reserve: multiple builders (from different threads) for the
 * same cluster claim page slots atomically from a shared counter.
 * Falls back to storage->new_page() if the reserved range is exhausted.
 */
void mkt_posting_builder_set_shared_reserve(
		MktPostingBuilder *builder,
		BlockNumber		   start,
		uint32_t		   count,
		_Atomic(uint32_t) *next);

/*
 * Pin the first page to a specific block number. The first flush
 * writes to this block; subsequent pages use the reserve or new_page.
 */
void mkt_posting_builder_set_first_blkno(
		MktPostingBuilder *builder, BlockNumber blkno);

/*
 * Add a raw vector. Encodes with RaBitQ relative to centroid.
 */
void mkt_posting_builder_add(
		MktPostingBuilder *builder, ItemPointerData tid, const float *vector);

/*
 * Add a pre-encoded entry. No RaBitQ encoding — data is already
 * quantized. Works with both AoS and fastscan formats.
 */
void mkt_posting_builder_add_encoded(
		MktPostingBuilder *builder,
		ItemPointerData	   tid,
		float			   f_add,
		float			   f_rescale,
		float			   f_error,
		const uint8_t	  *bits);

BlockNumber mkt_posting_builder_finish(MktPostingBuilder *builder);

/*
 * Finalize entries but don't flush the last page. The partial page
 * data remains in builder->mem_page for merging by the caller.
 * Returns the head of the flushed chain (InvalidBlockNumber if no
 * pages were flushed).
 */
BlockNumber mkt_posting_builder_finish_partial(MktPostingBuilder *builder);

void mkt_posting_builder_cleanup(MktPostingBuilder *builder);

static inline BlockNumber
mkt_posting_builder_head(const MktPostingBuilder *b)
{
	return b->head_blkno;
}

static inline BlockNumber
mkt_posting_builder_tail(const MktPostingBuilder *b)
{
	return b->prev_blkno;
}

/* ----------------------------------------------------------------
 * Flat builder — one buffer per cluster (standalone benchmark)
 * ---------------------------------------------------------------- */

typedef struct MktFlatPostingBuilder
{
	const RaBitQParams *params;
	Dimension			dim;
	uint32_t			cluster_id;
	const float		   *centroid;

	char	*buf;
	uint32_t max_entries;

	RaBitQData *enc_buf;
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

char *mkt_flat_posting_builder_finish(MktFlatPostingBuilder *builder);

void mkt_flat_posting_builder_cleanup(MktFlatPostingBuilder *builder);

#endif /* MKT_POSTING_BUILD_H */
