/*
 * parallel_build_worker.c - PG parallel index build, worker side
 *
 * Multi-phase parallel build:
 *   Phase 1 (sampling): cooperative heap scan, each worker collects
 *     samples into its own slot.
 *   Phase 2 (k-means): leader seeds the initial centroids, then
 *     iterative assignment + accumulation on per-worker samples,
 *     barrier-synchronized with leader reduce.
 *   Phase 3 (posting): cooperative heap scan, tree descent +
 *     RaBitQ encode + streaming to posting pages.
 */

#ifdef MKT_STANDALONE
#include "standalone/parallel_ctx.h" /* ParallelWorkerNumber */
#include "standalone/pg_compat.h"
#else
#include <postgres.h>

#include <access/parallel.h>
#include <catalog/index.h>
#include <miscadmin.h>
#include <storage/barrier.h>
#include <storage/bufmgr.h>
#include <storage/shm_toc.h>
#include <utils/rel.h>
#include <utils/wait_event.h>
#endif

#include "algo/hkmeans.h"
#include "algo/kmeans_internal.h"
#include "algo/vecops.h"
#include "core/log.h"
#include "core/memory.h"
#include "index/parallel_build.h"
#include "index/posting_build.h"
#include "index/posting_build_parallel.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Phase 1: Sampling callback
 *
 * Collects vectors into a per-worker sample slot (stride-subsampled,
 * normalized for cosine). Initial centroids are seeded later by the
 * leader from the pooled samples.
 * ---------------------------------------------------------------- */

void
mkt_sample_cb(void *state, ItemPointerData tid, const float *vec)
{
	SampleCbState *sc = (SampleCbState *)state;

	(void)tid;

	/* Stride-based subsampling */
	if (sc->stride_counter > 0)
	{
		sc->stride_counter--;
		return;
	}
	sc->stride_counter = sc->stride - 1;

	if (sc->count >= sc->max_samples)
		return;

	Dimension dim  = sc->dim;
	float	 *dest = sc->samples + (size_t)sc->count * dim;

	memcpy(dest, vec, dim * sizeof(float));

	/* Normalize for cosine */
	if (sc->metric == DISTANCE_COSINE)
	{
		float norm = mkt_l2_norm(dest, dim);
		if (norm > 0.0f)
			mkt_vector_scale(dest, 1.0f / norm, dest, dim);
	}

	sc->count++;
}

/* ----------------------------------------------------------------
 * Phase 2: K-means assignment — thin wrappers around shared kernel
 * ---------------------------------------------------------------- */

void
mkt_km_assign_and_accumulate(
		const float	  *samples,
		uint32_t	   nsamples,
		const float	  *centroids,
		const float	  *norms_c,
		uint32_t	   nlist,
		Dimension	   dim,
		DistanceMetric metric,
		float		  *out_sums,
		uint32_t	  *out_cnts,
		float		  *out_cost)
{
	memset(out_sums, 0, (size_t)nlist * dim * sizeof(float));
	memset(out_cnts, 0, nlist * sizeof(uint32_t));
	*out_cost = 0.0f;
	kmeans_assign_accumulate(
			samples,
			NULL,
			0,
			nsamples,
			centroids,
			norms_c,
			nlist,
			dim,
			metric,
			NULL,
			0,
			out_sums,
			out_cnts,
			out_cost);
}

/* ----------------------------------------------------------------
 * Phase 2c: work-partitioned subtree build (shared kernel)
 *
 * See the contract on the declaration in parallel_build.h. Each participant
 * builds the full subtree for the root children it owns and writes the
 * contiguous HKMeansResult into that child's DSM slot; the leader grafts them.
 * This generalizes the build to arbitrary depth — the subtree is however many
 * levels nlist/fan_out needs — so the parallel path is not capped at two
 * levels.
 * ---------------------------------------------------------------- */

void
mkt_subtree_build_partitioned(
		int				  participant_id,
		int				  nparticipants,
		MktDsmSamples	 *dsm_samples,
		MktDsmRootAssign *dsm_ra,
		const float		 *root_cents,
		uint32_t		  km_k,
		uint32_t		  nlist,
		uint32_t		  fan_out,
		Dimension		  dim,
		DistanceMetric	  metric,
		uint32_t		  km_max_iterations,
		char			 *subtrees_base,
		uint64_t		  slot_size)
{
	/* Leaves each subtree targets, so its depth is the global depth minus the
	 * root level (all subtrees share it, keeping the grafted tree uniform). */
	uint32_t nlist_c = (nlist + fan_out - 1) / fan_out;

	/* Per-child sample counts across all participants (one pass). */
	uint32_t *child_count = mkt_alloc0((size_t)km_k * sizeof(uint32_t));
	for (int t = 0; t < nparticipants; t++)
	{
		const uint32_t *ra = mkt_dsm_root_assignments(dsm_ra, t);
		uint32_t		n  = mkt_dsm_sample_counts(dsm_samples)[t];
		for (uint32_t i = 0; i < n; i++)
			if (ra[i] < km_k)
				child_count[ra[i]]++;
	}

	KMeansOptions opts	= MKT_KMEANS_OPTIONS_DEFAULT;
	opts.max_iterations = km_max_iterations;
	/*
	 * Per-child problems are small (~nsamples/fan_out points, k=fan_out) and
	 * run on every participant at once. The Lloyd dot-product kernel beats the
	 * CBLAS sgemm path here — sgemm's per-call overhead dominates at this
	 * size, and avoiding BLAS keeps the concurrent per-participant builds off
	 * a shared library. This is the kernel the pre-subtree child phase used.
	 */
	opts.algorithm = KMEANS_ALGO_LLOYD;

	for (uint32_t child = (uint32_t)participant_id; child < km_k;
		 child += (uint32_t)nparticipants)
	{
		uint32_t cc	  = child_count[child];
		char	*slot = mkt_dsm_child_subtree(subtrees_base, child, slot_size);

		HKMeansResult *sub = NULL;
		opts.initial_centroids =
				NULL; /* default seeding, as the serial path */
		if (cc == 0)
		{
			/* Empty root cluster (k-means reseeding makes this effectively
			 * impossible at nlevels >= 2). Seed the subtree with the root
			 * centroid as a single sample so it still has the same depth as
			 * its siblings and the graft stays uniform. */
			float *seed = mkt_alloc((size_t)dim * sizeof(float));
			memcpy(seed,
				   root_cents + (size_t)child * dim,
				   (size_t)dim * sizeof(float));
			sub = mkt_hkmeans_f32(
					seed, 1, NULL, dim, nlist_c, fan_out, metric, &opts);
			mkt_free(seed);
		}
		else
		{
			/* Index this child's samples in place instead of copying them
			 * into a contiguous [cc * dim] buffer: cc is ~total_samples /
			 * fan_out, which exceeds MaxAllocSize at high nlist. The
			 * per-participant DSM sample blocks form one flat array, so a
			 * sample's global slot is t * max_per_worker + i; hkmeans
			 * clusters via indirect access (cc uint32 indices, not cc
			 * full vectors). */
			const float *vbase = mkt_dsm_worker_samples(dsm_samples, 0);
			uint32_t	 mpw   = dsm_samples->max_per_worker;
			uint32_t	*idx   = mkt_alloc((size_t)cc * sizeof(uint32_t));
			uint32_t	 g	   = 0;
			for (int t = 0; t < nparticipants; t++)
			{
				const uint32_t *ra = mkt_dsm_root_assignments(dsm_ra, t);
				uint32_t		n  = mkt_dsm_sample_counts(dsm_samples)[t];
				for (uint32_t i = 0; i < n; i++)
					if (ra[i] == child)
						idx[g++] = (uint32_t)t * mpw + i;
			}
			sub = mkt_hkmeans_f32(
					vbase, cc, idx, dim, nlist_c, fan_out, metric, &opts);
			mkt_free(idx);
		}

		if (sub != NULL)
		{
			if ((uint64_t)sub->total_size > slot_size)
				mkt_error(
						"mktann: subtree blob %u exceeds slot (%u > %lu)",
						child,
						sub->total_size,
						(unsigned long)slot_size);
			memcpy(slot, sub, sub->total_size);
			mkt_free(sub);
		}
	}

	mkt_free(child_count);
}

/* ----------------------------------------------------------------
 * Phases 1, 2, 2b: per-participant execution, shared by leader and workers
 *
 * The leader runs as participant 0 and calls these exactly as a worker does,
 * rather than duplicating the bodies. The phase barriers live inside each
 * function, so the leader and workers stay in lockstep by construction; the
 * leader-only coordination (seeding the initial centroids, reducing the
 * per-iteration accumulators) is gated on participant_id == 0 and runs at the
 * barrier rendezvous while the other participants wait.
 * ---------------------------------------------------------------- */

void
mkt_pbuild_exec_sampling(
		int				  participant_id,
		Relation		  heap,
		Relation		  index,
		struct IndexInfo *index_info,
		MktBuildShared	 *shared,
		MktDsmSamples	 *dsm_samples,
		Barrier			 *barrier)
{
	Dimension dim = shared->dim;

	/* Stride to subsample ~max_samples_per_worker from this participant's
	 * share of the heap. */
	double est_rows = RelationGetNumberOfBlocks(heap) *
					  (BLCKSZ / (double)(dim * sizeof(float) + 32));
	double	 est_per = est_rows / shared->nparticipants;
	uint32_t stride	 = 1;
	if (est_per > shared->max_samples_per_worker)
		stride = (uint32_t)(est_per / shared->max_samples_per_worker);
	if (stride < 1)
		stride = 1;

	SampleCbState sc = {
			.samples	 = mkt_dsm_worker_samples(dsm_samples, participant_id),
			.count		 = 0,
			.max_samples = shared->max_samples_per_worker,
			.stride		 = stride,
			.stride_counter = 0,
			.dim			= dim,
			.metric			= shared->metric,
	};

	/* progress: only the leader (participant 0) drives the PG progress view.
	 */
	mkt_build_scan(
			heap,
			index,
			index_info,
			shared,
			true,
			participant_id == 0,
			mkt_sample_cb,
			&sc);

	mkt_dsm_sample_counts(dsm_samples)[participant_id] = sc.count;

	/* Barrier: all participants done sampling. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
}

uint32_t
mkt_pbuild_exec_kmeans(
		int				participant_id,
		MktBuildShared *shared,
		MktDsmSamples  *dsm_samples,
		char		   *centroids_base,
		char		   *km_workers_base,
		Barrier		   *barrier)
{
	Dimension dim			= shared->dim;
	uint32_t  km_k			= shared->km_k;
	int		  nparticipants = shared->nparticipants;
	float	 *cents			= mkt_dsm_centroids(centroids_base);
	float	 *norms_c		= mkt_dsm_norms_c(centroids_base, km_k, dim);

	/*
	 * Leader-only seed: pick km_k initial centroids from the pooled samples,
	 * spread evenly across the concatenation of every participant's slot.
	 * Centralizing it (rather than slicing by participant) keeps init
	 * independent of how many workers launched and robust to a sample-starved
	 * one. The result varies run to run (work-stealing scan order), so the
	 * parallel build is not bit-reproducible.
	 */
	if (participant_id == 0)
	{
		uint32_t total_ns = 0;
		for (int t = 0; t < nparticipants; t++)
			total_ns += mkt_dsm_sample_counts(dsm_samples)[t];

		uint32_t step = (total_ns >= km_k) ? total_ns / km_k : 1;
		for (uint32_t i = 0; i < km_k; i++)
		{
			uint32_t gidx = (total_ns > 0) ? (i * step) % total_ns : 0;

			int		 t	  = 0;
			uint32_t base = 0;
			while (t < nparticipants &&
				   base + mkt_dsm_sample_counts(dsm_samples)[t] <= gidx)
			{
				base += mkt_dsm_sample_counts(dsm_samples)[t];
				t++;
			}
			if (t < nparticipants)
				memcpy(cents + (size_t)i * dim,
					   mkt_dsm_worker_samples(dsm_samples, t) +
							   (size_t)(gidx - base) * dim,
					   dim * sizeof(float));
		}

		if (shared->metric == DISTANCE_L2)
			for (uint32_t j = 0; j < km_k; j++)
				norms_c[j] = mkt_l2_norm_squared(cents + (size_t)j * dim, dim);
	}

	/* Barrier: initial centroids + norms ready; guards the reads below against
	 * the leader's seed write. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	float	*my_samples = mkt_dsm_worker_samples(dsm_samples, participant_id);
	uint32_t my_n		= mkt_dsm_sample_counts(dsm_samples)[participant_id];
	float	*my_sums =
			mkt_dsm_km_worker_sums(km_workers_base, km_k, dim, participant_id);
	uint32_t *my_cnts =
			mkt_dsm_km_worker_cnts(km_workers_base, km_k, dim, participant_id);
	float *my_cost =
			mkt_dsm_km_worker_cost(km_workers_base, km_k, dim, participant_id);

	/* Reduce scratch is leader-only. */
	float *old_cents = participant_id == 0
							 ? mkt_alloc((size_t)km_k * dim * sizeof(float))
							 : NULL;
	Size   km_sz	 = mkt_dsm_km_workers_size(nparticipants, km_k, dim);

	uint32_t iters = 0;
	for (uint32_t iter = 0; iter < shared->km_max_iterations; iter++)
	{
		iters++;

		/* Every participant assigns + accumulates over its own samples. */
		mkt_km_assign_and_accumulate(
				my_samples,
				my_n,
				cents,
				norms_c,
				km_k,
				dim,
				shared->metric,
				my_sums,
				my_cnts,
				my_cost);

		/* Barrier: all accumulators written; the leader reduces. */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		if (participant_id == 0)
		{
			memcpy(old_cents, cents, (size_t)km_k * dim * sizeof(float));

			const float **all_sums = mkt_alloc(
					nparticipants * sizeof(float *));
			const uint32_t **all_cnts = mkt_alloc(
					nparticipants * sizeof(uint32_t *));
			float *all_costs = mkt_alloc(nparticipants * sizeof(float));
			for (int t = 0; t < nparticipants; t++)
			{
				all_sums[t] =
						mkt_dsm_km_worker_sums(km_workers_base, km_k, dim, t);
				all_cnts[t] =
						mkt_dsm_km_worker_cnts(km_workers_base, km_k, dim, t);
				all_costs[t] =
						*mkt_dsm_km_worker_cost(km_workers_base, km_k, dim, t);
			}

			float total_cost;
			float shift_sq = kmeans_merge_centroids(
					cents,
					norms_c,
					old_cents,
					all_sums,
					all_cnts,
					all_costs,
					nparticipants,
					km_k,
					dim,
					shared->metric,
					&total_cost);

			float tol_sq		 = shared->km_tolerance * shared->km_tolerance;
			shared->km_converged = (shift_sq < tol_sq);
			memset(km_workers_base, 0, km_sz);

			mkt_free(all_sums);
			mkt_free(all_cnts);
			mkt_free(all_costs);
		}

		/* Barrier: updated centroids + convergence flag visible to all. */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		if (shared->km_converged)
			break;
	}

	if (old_cents != NULL)
		mkt_free(old_cents);

	return iters;
}

void
mkt_pbuild_exec_root_assign(
		int				  participant_id,
		MktBuildShared	 *shared,
		MktDsmSamples	 *dsm_samples,
		MktDsmRootAssign *dsm_ra,
		char			 *centroids_base,
		Barrier			 *barrier)
{
	Dimension dim	  = shared->dim;
	uint32_t  km_k	  = shared->km_k;
	float	 *cents	  = mkt_dsm_centroids(centroids_base);
	float	 *norms_c = mkt_dsm_norms_c(centroids_base, km_k, dim);

	kmeans_assign(
			mkt_dsm_worker_samples(dsm_samples, participant_id),
			0,
			mkt_dsm_sample_counts(dsm_samples)[participant_id],
			cents,
			norms_c,
			km_k,
			dim,
			shared->metric,
			mkt_dsm_root_assignments(dsm_ra, participant_id));

	/* Barrier: all root assignments written; subtree gather can read them. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
}

/* ----------------------------------------------------------------
 * Phase 2.5: parallel full-table leaf-centroid refinement
 * ---------------------------------------------------------------- */

typedef struct RefineCbState
{
	MktBuildShared		*shared;
	const HKMeansResult *tree;
	float				*sums; /* shared accumulator, indexed leaf - tile_lo */
	uint64_t			*counts; /* shared accumulator */
	Dimension			 dim;
	DistanceMetric		 metric;
	uint32_t tile_lo; /* accumulate only leaves in [tile_lo, tile_hi) */
	uint32_t tile_hi;
	float	*scratch; /* per-participant normalized copy (cosine) */
} RefineCbState;

/*
 * Route a vector to its refinement leaf, exactly as both the serial and
 * parallel refine passes must: for cosine the tree is trained in normalized
 * space, so normalize into scratch first and accumulate that copy. Sets *out_v
 * to the vector to accumulate (the normalized copy for cosine, else the input)
 * and returns its leaf. Sharing this keeps the two paths' routing identical.
 */
uint32_t
mkt_refine_assign_leaf(
		const HKMeansResult *tree,
		const float			*vec,
		Dimension			 dim,
		DistanceMetric		 metric,
		float				*scratch,
		const float		   **out_v)
{
	const float *v = vec;
	if (metric == DISTANCE_COSINE)
	{
		memcpy(scratch, vec, (size_t)dim * sizeof(float));
		mkt_l2_normalize(scratch, dim);
		v = scratch;
	}
	*out_v = v;
	return mkt_hkmeans_assign(tree, v, metric, NULL);
}

static void
mkt_refine_cb(void *state, ItemPointerData tid, const float *vec)
{
	RefineCbState *rs  = (RefineCbState *)state;
	Dimension	   dim = rs->dim;

	(void)tid;

	const float *v;
	uint32_t	 leaf = mkt_refine_assign_leaf(
			rs->tree, vec, dim, rs->metric, rs->scratch, &v);
	/* Only the current tile's leaves are resident in the accumulator. */
	if (leaf < rs->tile_lo || leaf >= rs->tile_hi)
		return;
	uint32_t idx	= leaf - rs->tile_lo;
	uint32_t stripe = idx % MKT_REFINE_LOCK_STRIPES;

	mkt_pbuild_accum_lock(rs->shared, stripe);
	float *sum = rs->sums + (size_t)idx * dim;
	for (Dimension j = 0; j < dim; j++)
		sum[j] += v[j];
	rs->counts[idx]++;
	mkt_pbuild_accum_unlock(rs->shared, stripe);
}

void
mkt_pbuild_exec_refine(
		int				   participant_id,
		Relation		   heap,
		Relation		   index,
		struct IndexInfo  *index_info,
		MktBuildShared	  *shared,
		HKMeansResult	  *tree,
		MktDsmRefineAccum *accum,
		Barrier			  *barrier)
{
	Dimension dim	  = shared->dim;
	uint32_t  nleaves = tree->nleaves;
	float	 *sums	  = mkt_dsm_refine_sums(accum);
	uint64_t *counts  = mkt_dsm_refine_counts(accum);
	float	 *cents	  = hk_leaf_centroids(tree);

	/* The accumulator holds at most accum->nleaves leaves (the bounded tile
	 * capacity), so leaves are processed in tiles, re-scanning the heap per
	 * tile. nleaves <= capacity is a single tile (the common case). Leader and
	 * workers derive `tile` identically from the DSM capacity, so the barrier
	 * sequence below stays in lockstep. */
	uint32_t tile = accum->nleaves;

	RefineCbState rs = {
			.shared	 = shared,
			.tree	 = tree,
			.sums	 = sums,
			.counts	 = counts,
			.dim	 = dim,
			.metric	 = shared->metric,
			.scratch = mkt_alloc((size_t)dim * sizeof(float)),
	};

	for (uint32_t it = 0; it < shared->refine_iters; it++)
	{
		for (uint32_t lo = 0; lo < nleaves; lo += tile)
		{
			uint32_t hi = (lo + tile < nleaves) ? lo + tile : nleaves;
			rs.tile_lo	= lo;
			rs.tile_hi	= hi;

			/* Leader zeroes the (tile-sized) accumulator and resets the scan.
			 */
			if (participant_id == 0)
			{
				memset(sums, 0, (size_t)(hi - lo) * dim * sizeof(float));
				memset(counts, 0, (size_t)(hi - lo) * sizeof(uint64_t));
				mkt_pbuild_rescan(heap, shared);
			}
			/* Barrier: accumulator cleared + scan reset before anyone scans.
			 */
			BarrierArriveAndWait(
					barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

			/* All participants cooperatively scan the heap and accumulate. */
			mkt_build_scan(
					heap,
					index,
					index_info,
					shared,
					true,
					participant_id == 0,
					mkt_refine_cb,
					&rs);

			/* Barrier: every row accumulated before the leader divides. */
			BarrierArriveAndWait(
					barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

			/* Leader recomputes this tile's leaf centroids = per-leaf means.
			 */
			if (participant_id == 0)
			{
				for (uint32_t l = lo; l < hi; l++)
				{
					if (counts[l - lo] == 0)
						continue; /* keep subsample centroid for empty leaf */
					float *sum = sums + (size_t)(l - lo) * dim;
					float *c   = cents + (size_t)l * dim;
					double inv = 1.0 / (double)counts[l - lo];
					for (Dimension j = 0; j < dim; j++)
						c[j] = (float)(sum[j] * inv);
				}
			}
			/* Barrier: refined centroids visible before the next tile/pass. */
			BarrierArriveAndWait(
					barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
		}
	}

	mkt_free(rs.scratch);
}

/* ----------------------------------------------------------------
 * Phase 3: posting scan -> cluster-keyed sort (sort-seam path)
 *
 * Each worker assigns + RaBitQ-encodes every vector and feeds the compact
 * entry into the shared cluster-keyed sorter (primary + optional SOAR
 * secondary). The leader merges and builds the pages. Memory is bounded by
 * maintenance_work_mem inside the sorter.
 * ---------------------------------------------------------------- */
typedef struct PostingSortCbState
{
	const HKMeansResult *tree;
	MktBuildParams		 bp;
	MktBuildWorkerBufs	 bufs;
	const RaBitQParams	*rq_params;
	RaBitQData			*enc_buf;
	RaBitQScratch		*enc_scratch;
	const float			*leaf_cents;
	MktSorter			*sorter;
	Dimension			 dim;
	char				*entry; /* scratch, mkt_posting_entry_size(dim) */
	double				 indtuples;
	double				 soar_dupes;
} PostingSortCbState;

/*
 * Emit a vector's posting entries into the cluster-keyed sorter: the primary,
 * plus the secondary (SOAR / boundary replica) when the assignment has one.
 * Each entry is RaBitQ-encoded relative to its own cluster centroid; the
 * scratch buffers (enc_buf, enc_scratch, entry) are reused across both. Shared
 * by the serial build callback and the parallel posting worker. Returns true
 * when a secondary entry was written (the caller counts replicas).
 */
bool
mkt_posting_emit_assignment(
		MktSorter				 *sorter,
		const MktBuildAssignment *asgn,
		const RaBitQParams		 *params,
		const float				 *leaf_centroids,
		Dimension				  dim,
		ItemPointerData			  tid,
		RaBitQData				 *enc_buf,
		RaBitQScratch			 *enc_scratch,
		void					 *entry)
{
	mkt_posting_entry_encode(
			params,
			asgn->enc_vector,
			leaf_centroids + (size_t)asgn->primary * dim,
			dim,
			enc_buf,
			enc_scratch,
			tid,
			entry);
	mkt_pbuild_sort_put(sorter, asgn->primary, entry);

	if (asgn->secondary == MKT_INVALID_CLUSTER)
		return false;

	mkt_posting_entry_encode(
			params,
			asgn->enc_vector,
			leaf_centroids + (size_t)asgn->secondary * dim,
			dim,
			enc_buf,
			enc_scratch,
			tid,
			entry);
	mkt_pbuild_sort_put(sorter, asgn->secondary, entry);
	return true;
}

static void
posting_sort_cb(void *state, ItemPointerData tid, const float *vec)
{
	PostingSortCbState *cbs = (PostingSortCbState *)state;

	MktBuildAssignment asgn =
			mkt_build_assign_vector(cbs->tree, vec, &cbs->bp, &cbs->bufs);

	if (mkt_posting_emit_assignment(
				cbs->sorter,
				&asgn,
				cbs->rq_params,
				cbs->leaf_cents,
				cbs->dim,
				tid,
				cbs->enc_buf,
				cbs->enc_scratch,
				cbs->entry))
		cbs->soar_dupes++;
	cbs->indtuples++;
}

/* ----------------------------------------------------------------
 * Worker entry point — multi-phase build
 * ---------------------------------------------------------------- */

void
mkt_parallel_build_main(dsm_segment *seg, shm_toc *toc)
{
	MktPBuildWorker w;
	mkt_pbuild_worker_attach(toc, &w);

	MktBuildShared *shared	  = w.shared;
	Barrier		   *barrier	  = w.barrier;
	Relation		heapRel	  = w.heapRel;
	Relation		indexRel  = w.indexRel;
	int				worker_id = w.worker_id;
	Dimension		dim		  = w.dim;

	/* ---- Phases 1, 2, 2b: the shared per-participant bodies (the leader runs
	 * the very same code as participant 0). Each call includes its phase
	 * barrier(s). ---- */
	MktDsmSamples *dsm_samples =
			shm_toc_lookup(toc, MKT_DSM_KEY_SAMPLES, false);
	char  *centroids_base = shm_toc_lookup(toc, MKT_DSM_KEY_CENTROIDS, false);
	float *cents		  = mkt_dsm_centroids(centroids_base);
	char *km_workers_base = shm_toc_lookup(toc, MKT_DSM_KEY_KM_WORKERS, false);
	MktDsmRootAssign *dsm_ra =
			shm_toc_lookup(toc, MKT_DSM_KEY_ROOT_ASSIGN, false);
	uint32_t   km_k		 = shared->km_k;
	IndexInfo *indexInfo = BuildIndexInfo(indexRel);
#ifndef MKT_STANDALONE
	/* The leader marked the build concurrent and scans with an MVCC snapshot;
	 * the worker's freshly built IndexInfo defaults to non-concurrent, so it
	 * must be aligned or heapam's snapshot/OldestXmin assert trips in the scan
	 * (a valid OldestXmin paired with an MVCC snapshot). Matches nbtsort. */
	indexInfo->ii_Concurrent = shared->concurrent;
#endif

	mkt_pbuild_exec_sampling(
			worker_id,
			heapRel,
			indexRel,
			indexInfo,
			shared,
			dsm_samples,
			barrier);
	mkt_pbuild_exec_kmeans(
			worker_id,
			shared,
			dsm_samples,
			centroids_base,
			km_workers_base,
			barrier);
	mkt_pbuild_exec_root_assign(
			worker_id, shared, dsm_samples, dsm_ra, centroids_base, barrier);

	/* ---- Phase 2c: Child subtrees (work-partitioned, barrier-free) ----
	 *
	 * For a hierarchical tree (>= 2 levels) each participant builds the full
	 * subtree for the root children it owns. A flat (1-level) build has no
	 * children — the leader clusters it serially and we just meet the barrier.
	 * The participant computes nlevels itself so it agrees with the leader. */
	if (mkt_compute_nlevels(shared->nlist, shared->fan_out) >= 2)
	{
		char *subtrees_base =
				shm_toc_lookup(toc, MKT_DSM_KEY_CHILD_SUBTREES, false);
		mkt_subtree_build_partitioned(
				worker_id,
				shared->nparticipants,
				dsm_samples,
				dsm_ra,
				cents,
				km_k,
				shared->nlist,
				shared->fan_out,
				dim,
				shared->metric,
				shared->km_max_iterations,
				subtrees_base,
				shared->subtree_slot_size);
	}

	/* Barrier: all participants done building subtrees; the leader grafts the
	 * tree next. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 2.5: parallel leaf refinement (maintenance_work_mem-bounded
	 * builds only). Gated on refine_iters so the leader and workers run the
	 * identical barrier sequence. The leader publishes the grafted tree to DSM
	 * before the first barrier here. ---- */
	if (shared->refine_iters > 0)
	{
		/* Barrier: leader has grafted + published the tree to DSM. */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		HKMeansResult *rtree = shm_toc_lookup(toc, MKT_DSM_KEY_TREE, false);
		MktDsmRefineAccum *accum =
				shm_toc_lookup(toc, MKT_DSM_KEY_REFINE_ACCUM, false);
		mkt_pbuild_exec_refine(
				worker_id,
				heapRel,
				indexRel,
				indexInfo,
				shared,
				rtree,
				accum,
				barrier);
	}

	/* Barrier: leader built + published the tree and initialized the shared
	 * sorter. Stay attached — the phase-3 barrier below syncs all worker sorts
	 * before the leader merges. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 3: posting scan -> cluster-keyed sort ---- */
	HKMeansResult *tree = shm_toc_lookup(toc, MKT_DSM_KEY_TREE, false);
	void *sortshared	= shm_toc_lookup(toc, MKT_DSM_KEY_SORTSHARED, false);

	uint32_t	  entry_size = (uint32_t)mkt_posting_entry_size(dim);
	RaBitQParams *rq_params	 = mkt_rabitq_create(dim, shared->rabitq_seed);
	const float	 *leaf_cents = hk_leaf_centroids(tree);
	RaBitQData	 *enc_buf	 = mkt_alloc(MKT_RABITQ_DATA_SIZE(dim));
	RaBitQScratch enc_scratch;
	mkt_rabitq_scratch_init(&enc_scratch, dim);

	/* worker_id is 1..N for launched workers; the sorter's 0-based worker
	 * index is worker_id - 1. Worker sorts run concurrently, so each gets a
	 * share of the budget (mwm / participants) to bound peak memory; the
	 * leader merge runs alone afterward and uses the full budget. */
	int worker_wm = shared->work_mem_kb /
					(shared->nparticipants > 0 ? shared->nparticipants : 1);
	if (worker_wm < 64)
		worker_wm = 64;
	MktSorter *sorter = mkt_pbuild_sort_begin(
			sortshared, seg, worker_id - 1, 0, false, entry_size, worker_wm);

	MktBuildWorkerBufs bufs = mkt_build_worker_bufs_create(dim);
	PostingSortCbState cbs	= {
			 .tree		  = tree,
			 .bp		  = {.dim			   = dim,
							 .metric		   = shared->metric,
							 .soar_lambda	   = shared->soar_lambda,
							 .boundary_epsilon = shared->boundary_epsilon},
			 .bufs		  = bufs,
			 .rq_params	  = rq_params,
			 .enc_buf	  = enc_buf,
			 .enc_scratch = &enc_scratch,
			 .leaf_cents  = leaf_cents,
			 .sorter	  = sorter,
			 .dim		  = dim,
			 .entry		  = mkt_alloc(entry_size),
			 .indtuples	  = 0,
			 .soar_dupes  = 0,
	 };

	mkt_build_scan(
			heapRel,
			indexRel,
			indexInfo,
			shared,
			true,
			false,
			posting_sort_cb,
			&cbs);

	mkt_pbuild_sort_performsort(sorter);

	/* Barrier: every worker has finished sorting; the leader merges next. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	BarrierDetach(barrier);

	mkt_pbuild_worker_add_counts(shared, cbs.indtuples, cbs.soar_dupes);

	mkt_pbuild_sort_end(sorter);
	mkt_free(cbs.entry);
	mkt_build_worker_bufs_free(&bufs);
	mkt_rabitq_scratch_cleanup(&enc_scratch);
	mkt_free(enc_buf);

	mkt_pbuild_worker_detach(toc, &w);
}
