/*
 * posting_build.h - Posting list construction
 *
 * Vector-to-cluster assignment (tree descent, SOAR, boundary
 * replication) and streaming page builder. Used by both standalone
 * and PostgreSQL builds. No PostgreSQL dependencies.
 */

#ifndef MKT_POSTING_BUILD_H
#define MKT_POSTING_BUILD_H

#include <stdbool.h>
#include <stdint.h>

#include "algo/hkmeans.h"
#include "core/atomics.h"
#include "index/posting_page.h"
#include "index/storage.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Vector-to-cluster assignment
 * ---------------------------------------------------------------- */

#define MKT_INVALID_CLUSTER UINT32_MAX

typedef struct MktBuildParams
{
	Dimension	   dim;
	DistanceMetric metric;
	double		   soar_lambda;
	double		   boundary_epsilon;
} MktBuildParams;

typedef struct MktBuildAssignment
{
	uint32_t	 primary;
	uint32_t	 secondary;
	const float *enc_vector;
} MktBuildAssignment;

/*
 * Beam search parameters for the boundary (border-neighbor) secondary
 * search. The boundary cluster is the 2nd-nearest centroid by plain
 * distance, so a tree beam-descent finds it far cheaper than scanning
 * all leaves. SOAR's orthogonality-amplified search stays exact (its
 * optimum need not be among the nearest-by-distance leaves).
 */
#define MKT_SECONDARY_TOPK		 8
#define MKT_SECONDARY_BEAM_WIDTH 16

typedef struct MktBuildWorkerBufs
{
	float	 *norm_buf;		/* [dim] for cosine normalization */
	float	 *residual_buf; /* [dim] for SOAR residual computation */
	uint32_t *cand_leaves;	/* [MKT_SECONDARY_TOPK] beam candidates */
	Distance *cand_dists;	/* [MKT_SECONDARY_TOPK] candidate distances */
} MktBuildWorkerBufs;

MktBuildWorkerBufs mkt_build_worker_bufs_create(Dimension dim);
void			   mkt_build_worker_bufs_free(MktBuildWorkerBufs *bufs);

MktBuildAssignment mkt_build_assign_vector(
		const HKMeansResult	 *tree,
		const float			 *vec,
		const MktBuildParams *params,
		MktBuildWorkerBufs	 *bufs);

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
 * Deferred batch output
 *
 * When a builder runs in deferred mode (storage == NULL), complete
 * pages accumulate here instead of being written to storage.
 * Each page is a full BLCKSZ buffer with content laid out but
 * without assigned block numbers or chain links.
 * ---------------------------------------------------------------- */

typedef struct MktPostingBatch
{
	char	*pages;
	uint32_t count;
	uint32_t capacity;
} MktPostingBatch;

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
	mkt_atomic_uint32 *shared_reserve_next;
	BlockNumber		   fixed_first_blkno;

	RaBitQData	 *enc_buf;
	RaBitQScratch enc_scratch;

	/* Page format dispatch */
	const MktPostingPageOps *page_ops;

	/* Deferred batch output (when storage == NULL) */
	MktPostingBatch batch;

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
		mkt_atomic_uint32 *next);

/*
 * Pin the first page to a specific block number. The first flush
 * writes to this block; subsequent pages use the reserve or new_page.
 */
void mkt_posting_builder_set_first_blkno(
		MktPostingBuilder *builder, BlockNumber blkno);

/*
 * Derive RaBitQ error bound from encoding factors.
 */
float mkt_posting_derive_f_error(float f_add, float f_rescale, Dimension dim);

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

/*
 * Transfer batch ownership from builder to output struct.
 * After this call, the builder's batch is empty (zeroed) and
 * the caller owns the pages buffer.
 */
void mkt_posting_builder_take_batch(
		MktPostingBuilder *builder, MktPostingBatch *out);

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
