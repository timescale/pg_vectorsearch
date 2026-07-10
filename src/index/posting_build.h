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
#include "index/query_scan.h"
#include "index/storage.h"
#include "mkt_types.h"
#include "quant/rabitq.h"

/* Opaque: the cluster-keyed sorter (defined in parallel_build.h). */
typedef struct MktSorter MktSorter;

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
 * Build-time centroid routing accuracy.
 *
 * The build assigns every vector exactly once, so it routes the centroid
 * descent for accuracy, not query speed. It must NOT inherit the
 * query-tuned mkt.centroid_beam_scale, which trades recall for QPS: a
 * small beam_scale makes the beam narrower than the MKT_SECONDARY_TOPK
 * candidates the secondary (boundary/SOAR) search requests, loosening
 * clustering and permanently lowering query recall.
 *
 * BEAM_SCALE = 1.0 makes beam_w = nprobe = MKT_SECONDARY_TOPK, i.e. the
 * descent keeps as many candidates as the secondary search returns. This
 * matches the build query-state buffers (beam_results / centroid_scratch
 * are sized for max_nprobe = MKT_SECONDARY_TOPK). The query-tuned default
 * (0.25) instead yields beam_w = 2, far narrower than the k candidates the
 * descent must produce.
 *
 * ERROR_SCALE = 0 matches the query default; with a full-width beam the
 * extra candidates a larger error bound would keep do not change the leaf
 * chosen (measured: assignment recall is unchanged). It must stay 0 here —
 * a positive error_scale in the single-candidate refine route (nprobe = 1)
 * overflows the beam-search candidate buffer and corrupts the build.
 */
#define MKT_BUILD_CENTROID_BEAM_SCALE  1.0f
#define MKT_BUILD_CENTROID_ERROR_SCALE 0.0f

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
 * Page-backed build routing (unified with the query/insert path)
 *
 * Routes each vector to its posting list exactly as a query does --
 * mkt_query_route over the centroid pages -- then encodes the RaBitQ
 * residual against the pt_centroid read from the target list's head page
 * and streams it to the cluster-keyed sorter (primary + optional SOAR /
 * boundary secondary). Shared by the serial build and the parallel posting
 * workers so build, insert, and query all route identically. The centroid
 * and head pages must already be written when this runs.
 * ---------------------------------------------------------------- */
typedef struct MktBuildRouteCtx
{
	MktQueryState	   *qs;			   /* routing state (not owned) */
	MktSorter		   *sorter;		   /* cluster-keyed output (not owned) */
	const RaBitQParams *rq_params;	   /* not owned */
	MktStorage		   *storage;	   /* head-page reads (not owned) */
	BlockNumber			first_posting; /* leaf c's head = first_posting + c */
	Dimension			dim;
	double				soar_lambda;
	double				boundary_epsilon;

	/* Owned scratch (allocated in init, freed in cleanup). */
	float	   *cand_pt; /* [MKT_SECONDARY_TOPK * dim] gathered pt_centroids */
	uint32_t   *cand_leaf; /* [MKT_SECONDARY_TOPK] leaf per candidate */
	Distance   *cand_dist; /* [MKT_SECONDARY_TOPK] candidate distances */
	float	   *pt_r;	   /* [dim] rotated residual scratch */
	RaBitQData *enc_buf;   /* RaBitQ encode output */
	RaBitQScratch enc_scratch;
	char		 *entry; /* [mkt_posting_entry_size(dim)] */

	/* Counters. */
	double indtuples;
	double soar_dupes;
} MktBuildRouteCtx;

void mkt_build_route_ctx_init(
		MktBuildRouteCtx   *ctx,
		MktQueryState	   *qs,
		MktSorter		   *sorter,
		const RaBitQParams *rq_params,
		MktStorage		   *storage,
		BlockNumber			first_posting,
		Dimension			dim,
		double				soar_lambda,
		double				boundary_epsilon);

void mkt_build_route_ctx_cleanup(MktBuildRouteCtx *ctx);

/* Route one vector, encode, and stream its entries to the sorter. Returns
 * true when a secondary (SOAR / boundary) replica was also emitted. */
bool mkt_build_route_emit(
		MktBuildRouteCtx *ctx, const float *vec, ItemPointerData tid);

/* Map a routed posting-head block back to its leaf index. Head blocks are the
 * formula first_posting + leaf, so this is a subtraction. Shared by the
 * route/encode path and the page-backed refine pass. */
static inline uint32_t
mkt_route_head_to_leaf(BlockNumber first_posting, BlockNumber head)
{
	return (uint32_t)(head - first_posting);
}

/*
 * Shared refine kernels: the serial and parallel refine passes route,
 * filter, and average identically -- only the accumulator ownership
 * (private vs striped-locked DSM) differs, so that add stays with the
 * caller.
 *
 * mkt_refine_route_row routes one row page-backed (k=1, exactly as the
 * query and insert paths do), maps the head back to its leaf, and returns
 * the vector to accumulate (the normalized copy in scratch for cosine) --
 * or NULL when the row routed nowhere or outside [tile_lo, tile_hi).
 * *out_idx is the tile-relative leaf index.
 */
const float *mkt_refine_route_row(
		struct MktQueryState *qs,
		BlockNumber			  first_posting,
		const float			 *vec,
		Dimension			  dim,
		bool				  cosine,
		float				 *scratch,
		uint32_t			  tile_lo,
		uint32_t			  tile_hi,
		uint32_t			 *out_idx);

/*
 * Divide one tile's sums by their counts and hand each refined leaf mean to
 * write_head (leaves with no routed rows keep their sample-trained head).
 * scratch is a caller-owned [dim] float buffer.
 */
typedef void (*MktLeafWriteFn)(void *ctx, uint32_t leaf, const float *vec);

/*
 * Shared leaf-head writer: writes leaf's posting-list head page (at the
 * formula-derived block first_posting + leaf) carrying pt_centroid =
 * P^T * centroid -- the encode reference the scan reads. Used as the
 * MktLeafWriteFn of both the streaming tree write and the refine pass, by
 * the serial build and the parallel leader alike. pt is caller-owned [dim]
 * scratch.
 */
typedef struct MktHeadWriteCtx
{
	MktStorage		   *storage;
	const RaBitQParams *rq_params;
	Dimension			dim;
	bool				fastscan;
	BlockNumber			first_posting; /* leaf c's head = first_posting + c */
	float			   *pt;			   /* [dim] scratch */
} MktHeadWriteCtx;

/* Stack-init/cleanup pair: allocates/frees the [dim] pt scratch. */
static inline void
mkt_head_write_ctx_init(
		MktHeadWriteCtx	   *h,
		MktStorage		   *storage,
		const RaBitQParams *rq_params,
		Dimension			dim,
		bool				fastscan,
		BlockNumber			first_posting)
{
	h->storage		 = storage;
	h->rq_params	 = rq_params;
	h->dim			 = dim;
	h->fastscan		 = fastscan;
	h->first_posting = first_posting;
	h->pt			 = mkt_alloc((size_t)dim * sizeof(float));
}

static inline void
mkt_head_write_ctx_cleanup(MktHeadWriteCtx *h)
{
	mkt_free(h->pt);
}

void mkt_write_leaf_head(void *arg, uint32_t leaf, const float *centroid);

/*
 * Build-time page-backed router base: the same MktIndexBase the query and
 * insert paths build, so the build scan routes each row identically. Both
 * back-ends must construct it the same way -- this is the one place. The
 * caller owns pt_global_mean ([dim], filled with the rotated global mean)
 * and frees it after mkt_query_state_cleanup.
 */
static inline void
mkt_build_router_base_init(
		MktIndexBase	 *base,
		RaBitQParams	 *params,
		MktStorage		 *storage,
		Dimension		  dim,
		uint8_t			  nlevels,
		BlockNumber		  first_centroid,
		DistanceMetric	  metric,
		MktCentroidFormat centroid_format,
		int				  fastscan_bits,
		double			  error_scale,
		double			  beam_scale,
		uint32_t		  fan_out,
		uint32_t		  nlist,
		uint64_t		  rabitq_seed,
		const float		 *global_mean,
		float			 *pt_global_mean)
{
	memset(base, 0, sizeof(*base));
	base->params		 = params;
	base->pt_global_mean = pt_global_mean;
	mkt_rabitq_rotate(params, global_mean, base->pt_global_mean);
	base->rabitq_seed		   = rabitq_seed;
	base->centroid_storage	   = storage;
	base->posting_storage	   = storage;
	base->page_base			   = NULL;
	base->dim				   = dim;
	base->nlevels			   = nlevels;
	base->first_centroid	   = first_centroid;
	base->metric			   = metric;
	base->centroid_format	   = centroid_format;
	base->fastscan			   = (centroid_format == MKT_CENTROID_FMT_FASTSCAN)
									   ? fastscan_bits
									   : 0;
	base->centroid_error_scale = error_scale;
	base->centroid_beam_scale  = beam_scale;
	base->fan_out = (uint8_t)(fan_out <= UINT8_MAX ? fan_out : UINT8_MAX);
	base->nlist	  = nlist;
}

void mkt_refine_write_means(
		const double   *sums,
		const uint64_t *counts,
		uint32_t		lo,
		uint32_t		hi,
		Dimension		dim,
		float		   *scratch,
		MktLeafWriteFn	write_head,
		void		   *write_head_ctx);

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

	/*
	 * Head-metadata bookkeeping. owns_head is set for head builders (those
	 * that produce the FIRST page) and drives whether finish() stamps the
	 * head's live_count / tail_blkno. n_entries counts every entry added
	 * through this builder, so a head builder that holds the whole chain
	 * (serial build, and the parallel leader's head builder when no
	 * continuations were streamed) stamps an exact live_count with no walk.
	 */
	bool	 owns_head;
	uint32_t n_entries;

	char mem_page[BLCKSZ] __attribute__((aligned(8)));
	bool page_dirty;
	/* Head page was adopted from storage (see
	 * mkt_posting_builder_adopt_head): when nothing is appended, the
	 * on-disk head is already final and is neither rewritten nor
	 * restamped. */
	bool adopted_head;

	BlockNumber fixed_first_blkno;

	RaBitQData	 *enc_buf;
	RaBitQScratch enc_scratch;

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
 * Format-dispatching init: fastscan page format when fastscan is true,
 * AoS otherwise. Same arguments as the two inits above.
 */
void mkt_posting_builder_init_fmt(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		const float		   *centroid,
		const float		   *pt_centroid,
		bool				fastscan);

/*
 * Adopt the pre-written (empty) posting-list head at head_blk and append
 * into it. The page-backed build writes each head before the posting scan
 * (the scan routes and encodes against its pt_centroid); adopting appends
 * to that same page instead of constructing a second head that must agree
 * with it, and a cluster that receives no entries keeps its on-disk head
 * untouched. Entries must arrive pre-encoded (mkt_posting_entry_add).
 */
void mkt_posting_builder_adopt_head(
		MktPostingBuilder  *builder,
		MktStorage		   *storage,
		const RaBitQParams *params,
		Dimension			dim,
		uint32_t			cluster_id,
		BlockNumber			head_blk,
		bool				fastscan);

/*
 * Pin the first page to a specific block number. The first flush
 * writes to this block; later pages are appended via new_page.
 */
void mkt_posting_builder_set_first_blkno(
		MktPostingBuilder *builder, BlockNumber blkno);

/*
 * Derive RaBitQ error bound from encoding factors.
 */

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

void mkt_posting_builder_cleanup(MktPostingBuilder *builder);

/* ----------------------------------------------------------------
 * Compact posting entry for the cluster-sorted build path.
 *
 * A fixed-size blob keyed (externally) by cluster id, carrying the RaBitQ code
 * so the sort moves ~108 bytes/vector instead of dim*4. Layout:
 *   [ItemPointerData tid][float f_add][float f_rescale][float f_error]
 *   [uint8 sign_bits[(dim+7)/8]]
 * mkt_posting_entry_encode() fills it (relative to the assigned cluster
 * centroid); mkt_posting_entry_add() replays it into a builder via
 * add_encoded. Used by both the serial (mktann_build.c) and parallel
 * (sort-seam) builds, so the format has a single definition.
 * ---------------------------------------------------------------- */
Size mkt_posting_entry_size(Dimension dim);

void mkt_posting_entry_encode(
		const RaBitQParams *params,
		const float		   *vec,
		const float		   *centroid,
		Dimension			dim,
		RaBitQData		   *enc_buf, /* scratch, MKT_RABITQ_DATA_SIZE(dim) */
		RaBitQScratch	   *scratch, /* scratch, scratch_init(dim) */
		ItemPointerData		tid,
		void			   *out_entry); /* mkt_posting_entry_size(dim) bytes */

/*
 * Page-backed twin of mkt_posting_entry_encode: encodes from a pre-computed
 * rotated residual (pt_query - pt_centroid) instead of a float vector +
 * centroid. Used by the page-backed build, where pt_centroid is read from the
 * posting head page (no in-RAM float centroids).
 */
void mkt_posting_entry_encode_from_pt(
		const RaBitQParams *params,
		const float		   *pt_residual,
		Dimension			dim,
		RaBitQData		   *enc_buf,
		RaBitQScratch	   *scratch,
		ItemPointerData		tid,
		void			   *out_entry);

void mkt_posting_entry_add(
		MktPostingBuilder *builder, const void *entry, Dimension dim);

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
