/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * parallel_build_worker.c - Parallel index build, per-participant bodies
 *
 * The leader runs these as participant 0 alongside the workers:
 *   Sampling: cooperative heap scan, each participant fills its own
 *     sample slot.
 *   K-means: leader seeds the initial centroids, then iterative
 *     assignment + accumulation over the per-participant samples,
 *     barrier-synchronized with a leader reduce.
 *   Subtrees: batched largest-first schedule; each participant clusters
 *     its batch's child into a ring slot the leader consumes.
 *   Refine (when subsampled): route the full table page-backed and
 *     accumulate per-leaf means into the shared tiled accumulator.
 *   Posting: cooperative heap scan, page-backed routing + RaBitQ encode
 *     into the cluster-keyed shared sort.
 */

#ifdef VS_STANDALONE
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

#include <inttypes.h>
#include <math.h>

#include "algo/hkmeans.h"
#include "algo/kmeans_internal.h"
#include "algo/vecops.h"
#include "core/log.h"
#include "core/memory.h"
#include "index/build_progress.h"
#include "index/index_build.h"
#include "index/parallel_build.h"
#include "index/posting_build.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Phase 1: Sampling callback
 *
 * Collects vectors into a per-worker sample slot (stride-subsampled,
 * normalized for cosine). Initial centroids are seeded later by the
 * leader from the pooled samples.
 * ---------------------------------------------------------------- */

void
prism_sample_cb(void *state, ItemPointerData tid, const float *vec)
{
	SampleCbState *sc = (SampleCbState *)state;

	(void)tid;

	sc->seen++;
	/* The sampling pass walks the whole heap; surface that in the progress
	 * view in coarse batches (the leader publishes exact counts at phase
	 * boundaries). */
	if ((sc->seen & 1023) == 0)
		prism_build_progress_incr_tuples(1024);

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

	/* Normalize for cosine. Zero-norm vectors are dropped from the
	 * sample: they carry no direction, so under cosine they sit at the
	 * origin -- equidistant from every real point -- and k-means drags
	 * centroids toward them, warping the encode references (and so the
	 * distance estimates) for every vector in the affected subtree.
	 * They stay in the posting lists like any other row; they just
	 * don't get a vote on the clustering. The check runs on the source
	 * so a dropped vector costs no copy, and the scale writes straight
	 * into the sample slot, normalizing and copying in one pass. */
	if (sc->metric == DISTANCE_COSINE)
	{
		float norm_sq = vs_l2_norm_squared(vec, dim);

		if (norm_sq == 0.0f)
			return;
		vec32_scale(vec, 1.0f / sqrtf(norm_sq), dest, dim);
	}
	else
		memcpy(dest, vec, dim * sizeof(float));

	sc->count++;
}

/* ----------------------------------------------------------------
 * Phase 2: K-means assignment — thin wrappers around shared kernel
 * ---------------------------------------------------------------- */

void
prism_km_assign_and_accumulate(
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

/*
 * Build ONE root-child's subtree into `slot` (a DSM ring slot indexed by
 * participant, not by child, so the batched streaming build keeps only
 * nparticipants subtrees resident). The subtree is clustered from this child's
 * root-assigned samples, indexed in place (no contiguous per-child copy).
 */
void
prism_build_child_subtree(
		uint32_t			child,
		uint32_t			child_count,
		int					nparticipants,
		PrismDsmSamples	   *dsm_samples,
		PrismDsmRootAssign *dsm_ra,
		const float		   *root_cents,
		uint32_t			nlist,
		uint32_t			fan_out,
		Dimension			dim,
		DistanceMetric		metric,
		uint32_t			km_max_iterations,
		char			   *slot,
		uint64_t			slot_size)
{
	uint32_t nlist_c = (nlist + fan_out - 1) / fan_out;
	/* The caller's histogram already counted this child's samples (one pass
	 * over the assignments instead of one per child). */
	uint32_t cc = child_count;

	KMeansOptions opts	   = VS_KMEANS_OPTIONS_DEFAULT;
	opts.max_iterations	   = km_max_iterations;
	opts.algorithm		   = KMEANS_ALGO_LLOYD;
	opts.initial_centroids = NULL;

	const size_t vec_nbytes = (size_t)dim * sizeof(float);

	HKMeansResult *sub = NULL;
	if (cc == 0)
	{
		float *seed = vs_alloc(vec_nbytes);
		memcpy(seed, root_cents + (size_t)child * dim, vec_nbytes);
		sub = vs_hkmeans_f32(
				seed, 1, NULL, dim, nlist_c, fan_out, metric, &opts);
		vs_free(seed);
	}
	else
	{
		const float *vbase = prism_dsm_worker_samples(dsm_samples, 0);
		uint32_t	 mpw   = dsm_samples->max_per_worker;
		uint32_t	*idx   = vs_alloc((size_t)cc * sizeof(uint32_t));
		uint32_t	 g	   = 0;
		for (int t = 0; t < nparticipants; t++)
		{
			const uint32_t *ra = prism_dsm_root_assignments(dsm_ra, t);
			uint32_t		n  = prism_dsm_sample_counts(dsm_samples)[t];
			for (uint32_t i = 0; i < n; i++)
				if (ra[i] == child)
					idx[g++] = (uint32_t)t * mpw + i;
		}
		sub = vs_hkmeans_f32(
				vbase, cc, idx, dim, nlist_c, fan_out, metric, &opts);
		vs_free(idx);
	}

	if (sub != NULL)
	{
		if ((uint64_t)sub->total_size > slot_size)
			vs_error(
					"prism: subtree blob %u exceeds slot "
					"(%u > %" PRIu64 ")",
					child,
					sub->total_size,
					slot_size);
		memcpy(slot, sub, sub->total_size);
		vs_free(sub);
	}
}

void
prism_pbuild_count_children(
		PrismDsmSamples	   *dsm_samples,
		PrismDsmRootAssign *dsm_ra,
		int					nparticipants,
		uint32_t			km_k,
		uint32_t		   *out_counts)
{
	memset(out_counts, 0, (size_t)km_k * sizeof(uint32_t));
	for (int t = 0; t < nparticipants; t++)
	{
		const uint32_t *ra = prism_dsm_root_assignments(dsm_ra, t);
		uint32_t		n  = prism_dsm_sample_counts(dsm_samples)[t];
		for (uint32_t i = 0; i < n; i++)
			out_counts[ra[i]]++;
	}
}

void
prism_pbuild_stream_subtrees(
		int					participant_id,
		int					nparticipants,
		PrismDsmSamples	   *dsm_samples,
		PrismDsmRootAssign *dsm_ra,
		const float		   *root_cents,
		uint32_t			km_k,
		uint32_t			nlist,
		uint32_t			fan_out,
		Dimension			dim,
		DistanceMetric		metric,
		uint32_t			km_max_iterations,
		char			   *subtrees_base,
		uint64_t			slot_size,
		Barrier			   *barrier,
		PrismBatchCb		batch_cb,
		void			   *cb_arg,
		uint32_t		   *out_child_order)
{
	uint32_t np		  = (uint32_t)nparticipants;
	uint32_t nbatches = (km_k + np - 1) / np;
	char	*slot =
			prism_dsm_child_subtree(subtrees_base, participant_id, slot_size);

	/* One histogram pass over the root assignments feeds every child's
	 * sample count (prism_build_child_subtree needs it, and counting per
	 * child would re-scan the assignments km_k times). */
	uint32_t *child_count = vs_alloc0((size_t)km_k * sizeof(uint32_t));
	prism_pbuild_count_children(
			dsm_samples, dsm_ra, nparticipants, km_k, child_count);

	/* Schedule children largest-first (LPT): every batch waits for its
	 * slowest subtree at two barriers, so the skewed children must land in
	 * the full batches, leaving the tail batch the small ones. Counts are
	 * identical for every participant (same shared assignments) and ties
	 * break on the child id, so all participants and the leader's blob
	 * replay derive the same order with no coordination. */
	uint32_t *order = vs_alloc((size_t)km_k * sizeof(uint32_t));
	for (uint32_t i = 0; i < km_k; i++)
		order[i] = i;
	for (uint32_t i = 1; i < km_k; i++)
	{
		uint32_t id = order[i];
		uint32_t j	= i;
		while (j > 0 && (child_count[order[j - 1]] < child_count[id] ||
						 (child_count[order[j - 1]] == child_count[id] &&
						  order[j - 1] > id)))
		{
			order[j] = order[j - 1];
			j--;
		}
		order[j] = id;
	}
	if (out_child_order != NULL)
		memcpy(out_child_order, order, (size_t)km_k * sizeof(uint32_t));

	for (uint32_t b = 0; b < nbatches; b++)
	{
		uint32_t my_idx = b * np + (uint32_t)participant_id;
		if (my_idx < km_k)
		{
			uint32_t my_child = order[my_idx];
			prism_build_child_subtree(
					my_child,
					child_count[my_child],
					nparticipants,
					dsm_samples,
					dsm_ra,
					root_cents,
					nlist,
					fan_out,
					dim,
					metric,
					km_max_iterations,
					slot,
					slot_size);
		}

		/* All participants have built this batch's subtrees into their slots.
		 */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		/* Leader consumes the batch before the slots are reused. */
		if (participant_id == 0 && batch_cb != NULL)
		{
			uint32_t base = b * np;
			uint32_t bs	  = km_k - base;
			if (bs > np)
				bs = np;
			batch_cb(cb_arg, order + base, bs, subtrees_base, slot_size);
		}

		/* Leader done with the batch; slots free for the next batch. */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	}

	vs_free(order);
	vs_free(child_count);
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
prism_pbuild_exec_sampling(
		int				  participant_id,
		Relation		  heap,
		Relation		  index,
		struct IndexInfo *index_info,
		PrismBuildShared *shared,
		PrismDsmSamples	 *dsm_samples,
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
			.samples = prism_dsm_worker_samples(dsm_samples, participant_id),
			.count	 = 0,
			.seen	 = 0,
			.max_samples	= shared->max_samples_per_worker,
			.stride			= stride,
			.stride_counter = 0,
			.dim			= dim,
			.metric			= shared->metric,
	};

	/* progress: only the leader (participant 0) drives the PG progress view.
	 * The scan count is discarded: the posting pass is the one full scan that
	 * feeds shared->reltuples, so the extra passes must not double-count. */
	(void)prism_build_scan(
			heap,
			index,
			index_info,
			shared,
			true,
			participant_id == 0,
			prism_sample_cb,
			&sc);

	prism_dsm_sample_counts(dsm_samples)[participant_id] = sc.count;
	prism_dsm_sample_seen(dsm_samples)[participant_id]	 = sc.seen;

	/* Flush the sub-batch remainder so the participant's contribution to
	 * the progress view is exact (and so every participant makes at least
	 * one flush per scan, whatever share the work-stealing gave it). */
	prism_build_progress_incr_tuples((int64_t)(sc.seen & 1023));

	/* Barrier: all participants done sampling. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
}

uint32_t
prism_pbuild_exec_kmeans(
		int				  participant_id,
		PrismBuildShared *shared,
		PrismDsmSamples	 *dsm_samples,
		char			 *centroids_base,
		char			 *km_workers_base,
		Barrier			 *barrier)
{
	Dimension dim			= shared->dim;
	uint32_t  km_k			= shared->km_k;
	int		  nparticipants = shared->nparticipants;
	float	 *cents			= prism_dsm_centroids(centroids_base);
	float	 *norms_c		= prism_dsm_norms_c(centroids_base, km_k, dim);
	/* One seed vector, and the whole km_k-centroid array. */
	const size_t vec_nbytes	  = (size_t)dim * sizeof(float);
	const size_t cents_nbytes = (size_t)km_k * vec_nbytes;

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
			total_ns += prism_dsm_sample_counts(dsm_samples)[t];

		uint32_t step = (total_ns >= km_k) ? total_ns / km_k : 1;
		for (uint32_t i = 0; i < km_k; i++)
		{
			uint32_t gidx = (total_ns > 0) ? (i * step) % total_ns : 0;

			int		 t	  = 0;
			uint32_t base = 0;
			while (t < nparticipants &&
				   base + prism_dsm_sample_counts(dsm_samples)[t] <= gidx)
			{
				base += prism_dsm_sample_counts(dsm_samples)[t];
				t++;
			}
			if (t < nparticipants)
				memcpy(cents + (size_t)i * dim,
					   prism_dsm_worker_samples(dsm_samples, t) +
							   (size_t)(gidx - base) * dim,
					   vec_nbytes);
		}

		if (shared->metric == DISTANCE_L2)
			for (uint32_t j = 0; j < km_k; j++)
				norms_c[j] = vs_l2_norm_squared(cents + (size_t)j * dim, dim);
	}

	/* Barrier: initial centroids + norms ready; guards the reads below against
	 * the leader's seed write. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	float *my_samples = prism_dsm_worker_samples(dsm_samples, participant_id);
	uint32_t my_n	  = prism_dsm_sample_counts(dsm_samples)[participant_id];
	float	*my_sums  = prism_dsm_km_worker_sums(
			   km_workers_base, km_k, dim, participant_id);
	uint32_t *my_cnts = prism_dsm_km_worker_cnts(
			km_workers_base, km_k, dim, participant_id);
	float *my_cost = prism_dsm_km_worker_cost(
			km_workers_base, km_k, dim, participant_id);

	/* Reduce scratch is leader-only. */
	float *old_cents = participant_id == 0 ? vs_alloc(cents_nbytes) : NULL;
	Size   km_sz	 = prism_dsm_km_workers_size(nparticipants, km_k, dim);

	uint32_t iters = 0;
	for (uint32_t iter = 0; iter < shared->km_max_iterations; iter++)
	{
		iters++;

		/* Every participant assigns + accumulates over its own samples. */
		prism_km_assign_and_accumulate(
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
			memcpy(old_cents, cents, cents_nbytes);

			const float **all_sums = vs_alloc(nparticipants * sizeof(float *));
			const uint32_t **all_cnts = vs_alloc(
					nparticipants * sizeof(uint32_t *));
			float *all_costs = vs_alloc(nparticipants * sizeof(float));
			for (int t = 0; t < nparticipants; t++)
			{
				all_sums[t] = prism_dsm_km_worker_sums(
						km_workers_base, km_k, dim, t);
				all_cnts[t] = prism_dsm_km_worker_cnts(
						km_workers_base, km_k, dim, t);
				all_costs[t] = *prism_dsm_km_worker_cost(
						km_workers_base, km_k, dim, t);
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

			vs_free(all_sums);
			vs_free(all_cnts);
			vs_free(all_costs);
		}

		/* Barrier: updated centroids + convergence flag visible to all. */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		if (shared->km_converged)
			break;
	}

	if (old_cents != NULL)
		vs_free(old_cents);

	return iters;
}

void
prism_pbuild_exec_root_assign(
		int					participant_id,
		PrismBuildShared   *shared,
		PrismDsmSamples	   *dsm_samples,
		PrismDsmRootAssign *dsm_ra,
		char			   *centroids_base,
		Barrier			   *barrier)
{
	Dimension dim	  = shared->dim;
	uint32_t  km_k	  = shared->km_k;
	float	 *cents	  = prism_dsm_centroids(centroids_base);
	float	 *norms_c = prism_dsm_norms_c(centroids_base, km_k, dim);

	kmeans_assign(
			prism_dsm_worker_samples(dsm_samples, participant_id),
			0,
			prism_dsm_sample_counts(dsm_samples)[participant_id],
			cents,
			norms_c,
			km_k,
			dim,
			shared->metric,
			prism_dsm_root_assignments(dsm_ra, participant_id));

	/* Barrier: all root assignments written; subtree gather can read them. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
}

/* ----------------------------------------------------------------
 * Phase 2.5: parallel full-table leaf-encode-reference refinement
 *
 * Page-backed mirror of the serial serial_refine_heads: route every row
 * exactly as the query/insert do (prism_query_route k=1 over the centroid
 * pages, then head -> leaf), accumulate per-leaf means into the tiled DSM
 * accumulator, and rewrite each leaf's head-page pt_centroid to the full-table
 * mean. No in-RAM tree.
 * ---------------------------------------------------------------- */

typedef struct RefineCbState
{
	PrismBuildShared *shared;
	PrismQueryState	 *qs;	   /* page-backed router (workers) */
	BlockNumber first_posting; /* head -> leaf: leaf = head - first_posting */
	double	   *sums;		   /* shared accumulator, indexed leaf - tile_lo */
	uint64_t   *counts;		   /* shared accumulator */
	Dimension	dim;
	bool		cosine;
	uint32_t	tile_lo; /* accumulate only leaves in [tile_lo, tile_hi) */
	uint32_t	tile_hi;
	float	   *scratch; /* per-participant normalized copy (cosine) */
} RefineCbState;

static void
prism_refine_cb(void *state, ItemPointerData tid, const float *vec)
{
	RefineCbState *rs  = (RefineCbState *)state;
	Dimension	   dim = rs->dim;

	(void)tid;

	uint32_t	 idx;
	const float *v = prism_refine_route_row(
			rs->qs,
			rs->first_posting,
			vec,
			dim,
			rs->cosine,
			rs->scratch,
			rs->tile_lo,
			rs->tile_hi,
			&idx);
	if (v == NULL)
		return;

	uint32_t stripe = idx % PRISM_REFINE_LOCK_STRIPES;

	prism_pbuild_accum_lock(rs->shared, stripe);
	double *sum = rs->sums + (size_t)idx * dim;
	for (Dimension j = 0; j < dim; j++)
		sum[j] += v[j];
	rs->counts[idx]++;
	prism_pbuild_accum_unlock(rs->shared, stripe);
}

void
prism_pbuild_exec_refine_paged(
		int						participant_id,
		Relation				heap,
		Relation				index,
		struct IndexInfo	   *index_info,
		PrismBuildShared	   *shared,
		struct PrismQueryState *qs,
		BlockNumber				first_posting,
		PrismDsmRefineAccum	   *accum,
		Barrier				   *barrier,
		PrismLeafWriteFn		write_head,
		void				   *write_head_ctx)
{
	Dimension dim	  = shared->dim;
	uint32_t  nleaves = shared->nlist; /* actual leaf count (published) */
	double	 *sums	  = prism_dsm_refine_sums(accum);
	uint64_t *counts  = prism_dsm_refine_counts(accum);

	/* The accumulator holds at most accum->nleaves leaves (the bounded tile
	 * capacity), so leaves are processed in tiles, re-scanning the heap per
	 * tile. nleaves <= capacity is a single tile (the common case). Leader and
	 * workers derive `tile` identically from the DSM capacity, so the barrier
	 * sequence below stays in lockstep. */
	uint32_t tile = accum->nleaves;

	RefineCbState rs = {
			.shared		   = shared,
			.qs			   = qs,
			.first_posting = first_posting,
			.sums		   = sums,
			.counts		   = counts,
			.dim		   = dim,
			.cosine		   = (shared->metric == DISTANCE_COSINE),
			.scratch	   = vs_alloc((size_t)dim * sizeof(float)),
	};

	/* Single pass: routing reads only the centroid pages, which refine
	 * never rewrites, so the assignment is a fixed point and one pass
	 * reaches it. */
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
				memset(sums, 0, (size_t)(hi - lo) * dim * sizeof(double));
				memset(counts, 0, (size_t)(hi - lo) * sizeof(uint64_t));
				prism_pbuild_rescan(heap, shared);
			}
			/* Barrier: accumulator cleared + scan reset before anyone scans.
			 */
			BarrierArriveAndWait(
					barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

			/* Workers cooperatively scan the heap and accumulate page-backed;
			 * the leader does not route (it has no qs), it only
			 * clears/divides, mirroring the phase-3 division of labor. */
			if (participant_id != 0)
				(void)prism_build_scan(
						heap,
						index,
						index_info,
						shared,
						true,
						false,
						prism_refine_cb,
						&rs);

			/* Barrier: every row accumulated before the leader divides. */
			BarrierArriveAndWait(
					barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

			/* Leader rewrites this tile's leaf head pages = per-leaf means. */
			if (participant_id == 0)
				prism_refine_write_means(
						sums,
						counts,
						lo,
						hi,
						dim,
						rs.scratch,
						write_head,
						write_head_ctx);
			/* Barrier: refined heads written before the next tile/pass. */
			BarrierArriveAndWait(
					barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
		}
	}

	vs_free(rs.scratch);
}

/* ----------------------------------------------------------------
 * Phase 3: posting scan -> cluster-keyed sort (sort-seam path)
 *
 * Each worker routes every vector page-backed (the same prism_query_route the
 * query and insert paths use), RaBitQ-encodes against the target list's head
 * pt_centroid, and feeds the compact entry into the shared cluster-keyed
 * sorter (primary + optional SOAR / boundary secondary), via the shared
 * PrismBuildRouteCtx helper. The leader merges and builds the pages. Memory is
 * bounded by maintenance_work_mem inside the sorter.
 * ---------------------------------------------------------------- */
static void
route_scan_cb(void *state, ItemPointerData tid, const float *vec)
{
	PrismBuildRouteCtx *ctx = (PrismBuildRouteCtx *)state;

	prism_build_route_emit(ctx, vec, tid);
	if (((uint64_t)ctx->indtuples & 1023) == 0)
		prism_build_progress_incr_tuples(1024);
}

/* ----------------------------------------------------------------
 * Worker entry point — multi-phase build
 * ---------------------------------------------------------------- */

void
prism_parallel_build_main(dsm_segment *seg, shm_toc *toc)
{
	PrismPBuildWorker w;
	prism_pbuild_worker_attach(toc, &w);

	PrismBuildShared *shared	= w.shared;
	Barrier			 *barrier	= w.barrier;
	Relation		  heapRel	= w.heapRel;
	Relation		  indexRel	= w.indexRel;
	int				  worker_id = w.worker_id;
	Dimension		  dim		= w.dim;

	/* ---- Phases 1, 2, 2b: the shared per-participant bodies (the leader runs
	 * the very same code as participant 0). Each call includes its phase
	 * barrier(s). ---- */
	void			*sample_seg = NULL;
	PrismDsmSamples *dsm_samples =
			prism_pbuild_samples_attach(toc, shared, &sample_seg);
	char *centroids_base = shm_toc_lookup(toc, PRISM_DSM_KEY_CENTROIDS, false);
	float *cents		 = prism_dsm_centroids(centroids_base);
	char  *km_workers_base =
			shm_toc_lookup(toc, PRISM_DSM_KEY_KM_WORKERS, false);
	PrismDsmRootAssign *dsm_ra =
			shm_toc_lookup(toc, PRISM_DSM_KEY_ROOT_ASSIGN, false);
	IndexInfo *indexInfo = BuildIndexInfo(indexRel);
#ifndef VS_STANDALONE
	/* The leader marked the build concurrent and scans with an MVCC snapshot;
	 * the worker's freshly built IndexInfo defaults to non-concurrent, so it
	 * must be aligned or heapam's snapshot/OldestXmin assert trips in the scan
	 * (a valid OldestXmin paired with an MVCC snapshot). Matches nbtsort. */
	indexInfo->ii_Concurrent = shared->concurrent;
#endif

	prism_pbuild_exec_sampling(
			worker_id,
			heapRel,
			indexRel,
			indexInfo,
			shared,
			dsm_samples,
			barrier);
	prism_pbuild_exec_kmeans(
			worker_id,
			shared,
			dsm_samples,
			centroids_base,
			km_workers_base,
			barrier);
	prism_pbuild_exec_root_assign(
			worker_id, shared, dsm_samples, dsm_ra, centroids_base, barrier);

	/* ---- Phase 2c: batched streaming subtree build (page-backed) ----
	 *
	 * For a hierarchical tree (>= 2 levels) the workers build per-root-child
	 * subtrees into a bounded ring of slots (subtree DSM is nparticipants
	 * slots, independent of the partition count); the leader consumes each
	 * batch, recording layout counts and keeping the blob in a spillable
	 * store, and later streams every subtree's pages by itself from that
	 * store. Workers pass no callback (leader-only consumes each batch). The
	 * barrier sequence is inside prism_pbuild_stream_subtrees, identical for
	 * leader and workers. A flat (1-level) build has no subtrees — the leader
	 * writes the single level directly, and neither side runs the subtree
	 * barriers. */
	if (vs_hkmeans_nlevels(shared->nlist, shared->fan_out) >= 2)
	{
		/* Ring barrier: the leader creates the subtree ring (sized from the
		 * per-child sample counts) and publishes its handle + slot size
		 * before arriving; attach only after. */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
		void *ring_seg = NULL;
		char *subtrees_base =
				prism_pbuild_subtree_ring_attach(shared, &ring_seg);
		prism_pbuild_stream_subtrees(
				worker_id,
				shared->nparticipants,
				dsm_samples,
				dsm_ra,
				cents,
				shared->km_k,
				shared->nlist,
				shared->fan_out,
				dim,
				shared->metric,
				shared->km_max_iterations,
				subtrees_base,
				shared->subtree_slot_size,
				barrier,
				NULL,
				NULL,
				NULL);
		prism_pbuild_subtree_ring_release(ring_seg);
	}

	/* Barrier: leader finished streaming the centroid tree + published the
	 * routing state + initialized the sorter; workers build their page-backed
	 * router next (from the just-published shared state). */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 3 setup: page-backed router shared by refine + posting scan
	 * --- */
	void *sortshared = shm_toc_lookup(toc, PRISM_DSM_KEY_SORTSHARED, false);
	BlockNumber	 first_posting = shared->first_posting;
	const float *global_mean =
			shm_toc_lookup(toc, PRISM_DSM_KEY_GLOBAL_MEAN, false);

	/* Exact internal-node centroids, published by the leader before the
	 * tree-ready barrier above: the build descent scores the internal
	 * tree levels exactly (an empty collection leaves the hook inert). */
	void					   *exact_seg = NULL;
	PrismExactInternalCentroids exact_centroids;
	prism_exact_centroid_collection_view(
			prism_pbuild_exact_centroids_attach(shared, &exact_seg),
			&exact_centroids);

	uint32_t	  entry_size = (uint32_t)prism_posting_entry_size(dim);
	RaBitQParams *rq_params	 = vs_rabitq_create(dim, shared->rabitq_seed);

	/* Per-worker storage over the index for page-backed head/centroid reads
	 * (PG opens one on the worker's indexRel; standalone shares the leader's).
	 */
	VsStorage *storage = prism_pbuild_worker_storage(&w);

	/* Routing base — the same PrismIndexBase the query/insert build, so the
	 * worker routes each row identically. nlevels + first_centroid (the
	 * streamed tree's root block) come from the shared state the leader
	 * published; the scales + global mean + fastscan bits also from shared. */
	PrismIndexBase base;
	prism_build_router_base_init(
			&base,
			rq_params,
			storage,
			dim,
			shared->nlevels,
			shared->first_centroid,
			shared->metric,
			shared->centroid_format,
			shared->fastscan_bits,
			shared->centroid_error_scale,
			shared->centroid_beam_scale,
			shared->fan_out,
			shared->nlist,
			shared->rabitq_seed,
			global_mean,
			vs_alloc((size_t)dim * sizeof(float)));
	/* Build-only accuracy hook: exact scoring of the internal tree levels
	 * (the query and insert paths never set this). */
	base.exact_internal = &exact_centroids;

	PrismQueryState qs;
	prism_query_state_init(&qs, &base, 1, PRISM_SECONDARY_TOPK);

	/* ---- Phase 2.5: page-backed full-table refine (only when subsampled)
	 * ---- Workers route + accumulate; the leader clears/divides and rewrites
	 * heads. Gated on shared->refine (identical on both sides) so the
	 * barrier sequence stays in lockstep. */
	if (shared->refine)
	{
		/* The accumulator overlays the sample region (dead since the subtree
		 * phase); the leader initialized its header before the tree-ready
		 * barrier above. */
		PrismDsmRefineAccum *accum = prism_pbuild_refine_overlay(dsm_samples);
		prism_pbuild_exec_refine_paged(
				worker_id,
				heapRel,
				indexRel,
				indexInfo,
				shared,
				&qs,
				first_posting,
				accum,
				barrier,
				NULL,
				NULL);
	}

	/* The samples (and the refine overlay riding in them) are dead; hand the
	 * segment back before the posting sort claims its own memory budget. */
	prism_pbuild_samples_release(dsm_samples, sample_seg);
	dsm_samples = NULL;

	/* ---- Phase 3: posting scan -> cluster-keyed sort (page-backed) ---- */

	/* worker_id is 1..N for launched workers; the sorter's 0-based worker
	 * index is worker_id - 1. Only the workers sort concurrently -- the
	 * leader's merge-only sorter opens after every worker sort finishes and
	 * uses the full budget -- so the budget splits across the workers
	 * alone; splitting across all participants would strand one share for
	 * the whole sort phase. */
	int nsorters  = shared->nparticipants > 1 ? shared->nparticipants - 1 : 1;
	int worker_wm = shared->work_mem_kb / nsorters;
	if (worker_wm < 64)
		worker_wm = 64;
	PrismSorter *sorter = prism_pbuild_sort_begin(
			sortshared, seg, worker_id - 1, 0, false, entry_size, worker_wm);

	PrismBuildRouteCtx route;
	prism_build_route_ctx_init(
			&route,
			&qs,
			sorter,
			rq_params,
			storage,
			first_posting,
			dim,
			shared->soar_lambda,
			shared->boundary_epsilon);

	/* Barrier: the leader reset the scan for the posting phase (after refine
	 * consumed it); workers may now scan. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	double heap_tuples = prism_build_scan(
			heapRel,
			indexRel,
			indexInfo,
			shared,
			true,
			false,
			route_scan_cb,
			&route);

	/* Sub-batch progress remainder, as in the sampling scan. */
	prism_build_progress_incr_tuples(
			(int64_t)((uint64_t)route.indtuples & 1023));

	prism_pbuild_sort_performsort(sorter);

	/* Barrier: every worker has finished sorting; the leader merges next. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	BarrierDetach(barrier);

	prism_pbuild_worker_add_counts(
			shared, route.indtuples, route.soar_dupes, heap_tuples);

	prism_pbuild_sort_end(sorter);
	prism_build_route_ctx_cleanup(&route);
	prism_query_state_cleanup(&qs);
	vs_free(base.pt_global_mean);
	prism_pbuild_exact_centroids_release(exact_seg);
	prism_pbuild_worker_storage_release(storage);

	prism_pbuild_worker_detach(toc, &w);
}
