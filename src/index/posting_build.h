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
#include "core/memory.h"
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

/*
 * Candidate set for the SOAR secondary search. SOAR's optimum need not be
 * the nearest-by-distance leaf, so the exact search scans all leaves —
 * O(nlist) per vector, which dominates build time and grows with deeper
 * trees. Restricting the search to the K nearest leaves (a tree beam
 * descent) makes the build independent of nlist while keeping the SOAR
 * optimum in the candidate set in practice (the orthogonality term is
 * added to ||v-c||^2, so the optimum is almost always among the nearest
 * few dozen). 64 is the tree beam cap (MKT_HK_MAX_TOPK).
 */
#define MKT_SOAR_CAND_K		64
#define MKT_SOAR_BEAM_WIDTH 64

/* Candidate buffers must hold the larger of the two searches. */
#define MKT_BUILD_CAND_MAX MKT_SOAR_CAND_K

/* Vectors per batch for the GEMM secondary path (amortizes centroid
 * reads across the batch). */
#define MKT_SECONDARY_BATCH 256

/* Centroids per tile. The secondary GEMM streams centroids in tiles so
 * the working set is B*TILE (not B*nleaves), keeping brute-force viable
 * at any nlist — the fallback to tree descent is then a performance
 * choice, not a memory limit. A small tile also keeps the per-worker
 * B*TILE matrix hot in cache (the workers share L3), which measured
 * faster than one large GEMM; ~512 is past the plateau without losing
 * sgemm efficiency. */
#define MKT_SECONDARY_TILE 512

typedef struct MktBuildWorkerBufs
{
	float	 *norm_buf;		/* [dim] for cosine normalization */
	float	 *residual_buf; /* [dim] for SOAR residual computation */
	uint32_t *cand_leaves;	/* [MKT_BUILD_CAND_MAX] beam candidates */
	Distance *cand_dists;	/* [MKT_BUILD_CAND_MAX] candidate distances */
} MktBuildWorkerBufs;

MktBuildWorkerBufs mkt_build_worker_bufs_create(Dimension dim);
void			   mkt_build_worker_bufs_free(MktBuildWorkerBufs *bufs);

MktBuildAssignment mkt_build_assign_vector(
		const HKMeansResult	 *tree,
		const float			 *vec,
		const MktBuildParams *params,
		MktBuildWorkerBufs	 *bufs);

/*
 * Primary cluster via tree descent, plus the encoded vector written into
 * enc_out[dim] (normalized for cosine, copied otherwise). Used by the
 * batched secondary path, which needs the encoded vectors contiguous.
 * Returns the primary cluster; *out_dist receives its distance.
 */
uint32_t mkt_build_assign_primary(
		const HKMeansResult	 *tree,
		const float			 *vec,
		const MktBuildParams *params,
		float				 *enc_out,
		Distance			 *out_dist);

/* ----------------------------------------------------------------
 * Batched secondary (boundary + SOAR) assignment
 *
 * Primary assignment stays per-vector (tree descent); the secondary
 * search is the memory-bandwidth-bound part because every vector scans
 * all leaf centroids. Batching B vectors lets the centroid block be read
 * once and reused across the batch via two sgemm calls (V·Cᵀ for the
 * boundary/L2 distances, R·Cᵀ for SOAR), turning a memory-bound scan
 * into a compute-bound GEMM. Requires CBLAS; callers fall back to the
 * per-vector path when mkt_secondary_batch_available() is false.
 * ---------------------------------------------------------------- */

typedef struct MktSecondaryBatch
{
	uint32_t	 max_batch;
	uint32_t	 nleaves;
	Dimension	 dim;
	const float *leaf_centroids; /* [nleaves * dim], not owned */
	float		*cent_norms;	 /* [nleaves] ||c||^2 */
	float		*vc;			 /* [max_batch * TILE] <v, c> per tile */
	float		*rc;			 /* [max_batch * TILE] <r_hat, c> per tile */
	float		*residuals;		 /* [max_batch * dim] normalized residuals */
	float		*vec_norms;		 /* [max_batch] ||v||^2 */
	float		*qrv;			 /* [max_batch] r_hat . v */
	/* Per-query running reductions across centroid tiles. */
	float	 *best2;	 /* [max_batch] best boundary distance (!= primary) */
	uint32_t *c2;		 /* [max_batch] arg of best2 */
	float	 *best_oa;	 /* [max_batch] best SOAR oa distance */
	uint32_t *best_oa_c; /* [max_batch] arg of best_oa */
} MktSecondaryBatch;

/* True when CBLAS is available (the batched path needs sgemm). */
bool mkt_secondary_batch_available(void);

void mkt_secondary_batch_init(
		MktSecondaryBatch *s,
		const float		  *leaf_centroids,
		uint32_t		   nleaves,
		Dimension		   dim,
		uint32_t		   max_batch);

void mkt_secondary_batch_free(MktSecondaryBatch *s);

/*
 * Assign the secondary (replication) cluster for a batch of n <=
 * max_batch encoded vectors. primary[]/primary_dist[] come from the
 * per-vector tree descent. Writes out_secondary[n], using
 * MKT_INVALID_CLUSTER where no replication applies.
 */
void mkt_secondary_batch_assign(
		MktSecondaryBatch	 *s,
		const float			 *vecs,
		uint32_t			  n,
		const uint32_t		 *primary,
		const float			 *primary_dist,
		const MktBuildParams *params,
		uint32_t			 *out_secondary);

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
	mkt_atomic_uint32 *shared_reserve_next;
	BlockNumber		   fixed_first_blkno;

	RaBitQData	 *enc_buf;
	RaBitQScratch enc_scratch;

	/* Page format dispatch */
	const MktPostingPageOps *page_ops;

	/*
	 * Full-page sink for deferred mode (storage == NULL). A completed page is
	 * handed to this callback — the parallel build streams full pages to the
	 * leader (over shm_mq) so worker memory stays bounded to one working page
	 * per cluster. The page's cluster id and flags (first vs continuation) are
	 * carried in the page itself.
	 */
	void (*page_sink)(void *ctx, uint32_t cluster_id, const char *page);
	void *sink_ctx;

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

/*
 * Shared reserve: multiple builders (from different threads) for the
 * same cluster claim page slots atomically from a shared counter.
 * Falls back to storage->new_page() if the reserved range is exhausted.
 * Used by every build path (serial, parallel, standalone).
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

/*
 * Set a full-page sink for deferred mode (storage == NULL). When set,
 * completed pages are streamed to `sink(ctx, cluster_id, page)` instead
 * of accumulated in the batch — the PG parallel build uses this to send
 * full pages to the leader over shm_mq. NULL (default) keeps batch
 * accumulation.
 */
void mkt_posting_builder_set_page_sink(
		MktPostingBuilder *builder,
		void (*sink)(void *ctx, uint32_t cluster_id, const char *page),
		void *sink_ctx);

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
