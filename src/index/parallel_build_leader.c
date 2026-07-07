/*
 * parallel_build_leader.c - PG parallel index build, leader side
 *
 * The leader sets up the DSM, launches workers, participates in the
 * sampling + k-means phases as worker_id 0, streams the centroid tree to
 * pages, then (phase 3) merges the workers' cluster-keyed sorted runs and
 * builds each posting list. Head blocks are formula-derived (cluster c's head
 * is first_posting + c), so no per-partition head array is materialized.
 * Returns false if parallelism could not start, so the caller falls back to a
 * serial build.
 */

#ifdef MKT_STANDALONE
#include "standalone/instr_time.h"
#include "standalone/parallel_ctx.h" /* ParallelContext + lifecycle */
#include "standalone/pg_compat.h"
#else
#include <postgres.h>

#include <access/parallel.h>
#include <access/table.h>
#include <access/tableam.h>
#include <access/xloginsert.h>
#include <catalog/index.h>
#include <commands/progress.h>
#include <common/pg_prng.h>
#include <miscadmin.h>
#include <pgstat.h>
#include <portability/instr_time.h>
#include <tcop/tcopprot.h>
#include <utils/backend_progress.h>
#include <utils/memutils.h>
#include <utils/rel.h>
#include <utils/sampling.h>
#endif

#include <math.h>

#include "algo/distance.h"
#include "algo/hkmeans.h"
#include "algo/kmeans.h"
#include "algo/kmeans_internal.h"
#include "algo/vecops.h"
#include "core/log.h"
#include "core/memory.h"
#include "index/build_progress.h"
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/index_build.h"
#include "index/parallel_build.h"
#include "index/posting_build.h"
#include "index/posting_page.h"
#include "mkt_halfvec.h"
#include "mkt_vector.h"
#include "quant/fastscan.h"

#ifndef MKT_STANDALONE
#include "mkt_pg.h"
#include "mktann_build.h"
#include "mktann_meta.h"
#include "mktann_storage.h"
#endif

/*
 * Build every cluster's posting list from a populated (not yet performsorted)
 * cluster-keyed sorter. Shared by the serial build and the parallel leader:
 * performsort, then read entries grouped by cluster and write each list with a
 * single resident page builder (head at first_posting + c, continuations
 * appended at the relation's end and chained). Ends the sorter.
 *
 * Why iterate clusters 0..nlist rather than just draining the sorter until it
 * is empty: EVERY cluster needs a posting-list head, including clusters that
 * received no vectors. The centroid tree's leaf entries reference first_posting
 * + c for every c in [0, nlist) (see mkt_write_centroid_tree), so an empty
 * cluster still needs a valid (empty) head block to point at. A loop over the
 * cluster index emits a head for every cluster uniformly — an empty cluster's
 * inner while simply does not run and finish() returns an empty head. A
 * drain-until-empty loop would only produce heads for clusters present in the
 * stream and would have to separately backfill empty heads for gap clusters and
 * for all trailing clusters past the last one seen.
 *
 * This REQUIRES (and assumes) the sorter returns entries in ascending cluster
 * order, matching the 0..nlist iteration order: the sorter is keyed on the
 * cluster id (PG tuplesort on the int4 key; standalone qsort by cluster), so
 * all entries for a cluster are contiguous and clusters appear in increasing
 * order. Thus while walking c upward, any remaining entry has cur_cluster >= c
 * (it equals c for a non-empty cluster, or is greater when c is empty). The
 * asserts below make that contract explicit: an out-of-order key would pair
 * entries with the wrong cluster, and an out-of-range id (>= nlist) would
 * never match any c and be silently dropped (leaving the sorter non-empty at
 * the end).
 */
void
mkt_posting_build_lists(
		MktSorter		   *sorter,
		MktStorage		   *storage,
		uint32_t			nlist,
		Dimension			dim,
		bool				fastscan,
		const RaBitQParams *rq_params,
		const float		   *ref_vecs,
		BlockNumber			first_posting)
{
	mkt_pbuild_sort_performsort(sorter);

	/*
	 * Each list's RaBitQ reference is P^T * its centroid. Two sources:
	 *   ref_vecs != NULL: rotate the in-RAM float centroid on the fly into
	 * this scratch (in-RAM tree still resident). ref_vecs == NULL: page-backed
	 * — the head page was pre-written with its pt_centroid during the centroid
	 * build, so read it from the head (which is then re-created, full, below).
	 * No in-RAM float centroids needed, so the tree can be freed before this
	 * runs. Either way it is one dim-vector, not a pt_centroids[nlist*dim]
	 * array.
	 */
	float *pt_centroid = mkt_alloc((size_t)dim * sizeof(float));

	uint32_t	cur_cluster = 0;
	const void *entry		= NULL;
	bool		have = mkt_pbuild_sort_getnext(sorter, &cur_cluster, &entry);
	for (uint32_t c = 0; c < nlist; c++)
	{
		/* Head block of cluster c is the formula first_posting + c; the head
		 * region [first_posting, first_posting + nlist) was pre-extended and its
		 * pt_centroid written during the centroid streaming. Continuation pages
		 * are appended at the relation's end and chained (no per-cluster
		 * reserve), so no O(nlist) reserve arrays are needed. */
		BlockNumber	 head_blk = first_posting + c;
		const float *cvec	  = (ref_vecs != NULL) ? ref_vecs + (size_t)c * dim
											   : NULL;
		if (ref_vecs != NULL)
			mkt_rabitq_rotate(rq_params, cvec, pt_centroid);
		else
		{
			Page hp = mkt_storage_read_page(storage, head_blk);
			memcpy(pt_centroid,
				   mkt_posting_pt_centroid(hp),
				   (size_t)dim * sizeof(float));
			mkt_storage_release_page(storage, head_blk);
		}

		MktPostingBuilder hb;
		if (fastscan)
			mkt_posting_builder_init_fastscan(
					&hb, storage, rq_params, dim, c, cvec, pt_centroid);
		else
			mkt_posting_builder_init(
					&hb, storage, rq_params, dim, c, cvec, pt_centroid);
		mkt_posting_builder_set_first_blkno(&hb, head_blk);

		/* Ascending, grouped order: a pending entry is never for a cluster we
		 * already finished. If it were < c we would have skipped its head. */
		Assert(!have || cur_cluster >= c);

		while (have && cur_cluster == c)
		{
			mkt_posting_entry_add(&hb, entry, dim);
			have = mkt_pbuild_sort_getnext(sorter, &cur_cluster, &entry);
		}

		(void)mkt_posting_builder_finish(&hb);
		mkt_posting_builder_cleanup(&hb);
	}

	/* Every entry must have landed in some cluster's list. A leftover entry
	 * means an id >= nlist (out of range) that matched no c and would
	 * otherwise be silently dropped. */
	Assert(!have);

	mkt_free(pt_centroid);
	mkt_pbuild_sort_end(sorter);
}

/*
 * Shared pre-posting centroid setup: normalize the leaf centroids for cosine
 * (in place — the tree is trained in normalized space), reserve block 0 (the
 * metadata page, written later by the shared finalize) plus the centroid
 * pages, and return the posting-area start block. The page layout is
 * deterministic from the tree, so the shared finalize recomputes it for the
 * centroid-tree write; only first_posting is needed here (for the posting
 * reserve, which differs between serial and parallel). The rotated
 * P^T*centroid each posting list needs is computed on the fly, per cluster, in
 * mkt_posting_build_lists. Shared by the serial build and the parallel leader.
 */
BlockNumber
mkt_build_setup_centroid_layout(
		MktStorage		 *storage,
		HKMeansResult	 *tree,
		Dimension		  dim,
		uint32_t		  nlist,
		DistanceMetric	  metric,
		MktCentroidFormat centroid_format)
{
	float *ref_vecs = hk_leaf_centroids(tree);
	if (metric == DISTANCE_COSINE)
		for (uint32_t c = 0; c < nlist; c++)
			mkt_l2_normalize(ref_vecs + (size_t)c * dim, dim);

	/* Block 0 = metadata page; extend so it exists (contents written later).
	 */
	mkt_storage_extend(storage, 1);

	uint32_t	max_ent = mkt_centroid_max_entries_fmt(dim, centroid_format);
	BlockNumber first_centroid = 1;
	/*
	 * mkt_compute_centroid_layout both returns where the posting area starts
	 * and fills a per-node first-block array. Here we only need the former (to
	 * size the posting reserve), so the array is throwaway scratch we free at
	 * once. The other callers — the centroid-tree writers in the build
	 * finalize — pass a long-lived array and keep it to place each node's
	 * centroid pages.
	 */
	BlockNumber *nfb = mkt_alloc((size_t)tree->nnodes * sizeof(BlockNumber));
	BlockNumber	 first_posting =
			mkt_compute_centroid_layout(tree, max_ent, first_centroid, nfb);
	mkt_free(nfb);
	mkt_storage_extend(storage, first_posting - first_centroid);
	return first_posting;
}

/* ----------------------------------------------------------------
 * Batched streaming tree build — leader-side callbacks
 * ---------------------------------------------------------------- */

/* Per-leaf head-page writer (mirrors the serial serial_write_head): writes the
 * cluster's posting-list head carrying pt_centroid = P^T*centroid at its
 * reserved head block. Passed to mkt_write_subtree_streaming. */
typedef struct LeaderHeadCtx
{
	MktStorage		   *storage;
	const RaBitQParams *rq_params;
	Dimension			dim;
	bool				fastscan;
	BlockNumber			first_posting; /* leaf c's head = first_posting + c */
	float			   *pt;			   /* [dim] scratch */
	float			   *pt_cache; /* optional [nlist * dim] rotated-centroid
								   * cache for the exact batched secondary
								   * (NULL = disabled) */
} LeaderHeadCtx;

static void
leader_write_head(void *arg, uint32_t leaf, const float *centroid)
{
	LeaderHeadCtx *h = (LeaderHeadCtx *)arg;
	mkt_rabitq_rotate(h->rq_params, centroid, h->pt);
	if (h->pt_cache != NULL)
		memcpy(h->pt_cache + (size_t)leaf * h->dim,
			   h->pt,
			   (size_t)h->dim * sizeof(float));

	MktPostingBuilder hb;
	if (h->fastscan)
		mkt_posting_builder_init_fastscan(
				&hb, h->storage, h->rq_params, h->dim, leaf, centroid, h->pt);
	else
		mkt_posting_builder_init(
				&hb, h->storage, h->rq_params, h->dim, leaf, centroid, h->pt);
	mkt_posting_builder_set_first_blkno(&hb, h->first_posting + leaf);
	mkt_posting_builder_finish(&hb);
	mkt_posting_builder_cleanup(&hb);
}

/* PLAN-pass batch callback: record each subtree's leaf count + centroid-page
 * count (no writes) so the leader can size the reserve + block layout, and
 * keep the blob in the spillable store for the streaming pass to read back
 * in the same (child) order. */
typedef struct PlanCbArg
{
	uint32_t *nleaves_arr;	 /* [km_k] */
	uint32_t *pages_arr;	 /* [km_k] */
	uint32_t  max_ent;
	uint32_t  subtree_nlevels; /* actual depth of the (uniform) subtrees */
	Dimension dim;
	double	 *leaf_sum; /* [dim] leaf-centroid sum across all subtrees, for
						 * the leaf_mean the leader uses as global_mean */
	MktBlobStore *store; /* subtree blobs, in child order */
} PlanCbArg;

static void
plan_batch_cb(
		void *arg, uint32_t base_child, uint32_t bs, char *base,
		uint64_t slot_size)
{
	PlanCbArg *a = (PlanCbArg *)arg;
	for (uint32_t s = 0; s < bs; s++)
	{
		const HKMeansResult *sub =
				(const HKMeansResult *)mkt_dsm_child_subtree(base, s, slot_size);
		uint32_t	 child = base_child + s;
		BlockNumber *nfb =
				mkt_alloc((size_t)sub->nnodes * sizeof(BlockNumber));
		/* Subtrees are built to a uniform depth, so any non-empty one gives the
		 * streamed tree's subtree depth (full depth = this + 1 root level). */
		if (sub->nleaves > 0)
			a->subtree_nlevels = sub->nlevels;
		a->nleaves_arr[child] = sub->nleaves;
		a->pages_arr[child] =
				(uint32_t)mkt_compute_centroid_layout(sub, a->max_ent, 0, nfb);
		mkt_free(nfb);

		if (a->leaf_sum != NULL)
		{
			const float *lc = hk_leaf_centroids(sub);
			for (uint32_t l = 0; l < sub->nleaves; l++)
				for (Dimension d = 0; d < a->dim; d++)
					a->leaf_sum[d] += lc[(size_t)l * a->dim + d];
		}

		mkt_pbuild_blobstore_put(a->store, sub, sub->total_size);
	}
}

bool
do_parallel_build(
		Relation				 heap,
		Relation				 index,
		struct IndexInfo		*index_info,
		const MktBuildConfig	*config,
		MktStorage				*storage,
		struct MktBuildProgress *prog,
		HKMeansResult		   **out_tree,
		double					*out_heap_tuples,
		double					*out_indtuples,
		double					*out_soar_dupes,
		float				   **out_global_mean,
		bool					*out_centroids_written,
		BlockNumber				*out_first_posting)
{
	int nworkers = index_info->ii_ParallelWorkers;

	if (out_centroids_written)
		*out_centroids_written = false;
	if (out_first_posting)
		*out_first_posting = InvalidBlockNumber;
	if (out_global_mean)
		*out_global_mean = NULL;

	MktPBuildLeader lead;
	if (!mkt_pbuild_setup_shared(&lead, heap, index, config, nworkers))
		return false;

	ParallelContext	 *pcxt			  = lead.pcxt;
	MktBuildShared	 *shared		  = lead.shared;
	Barrier			 *barrier		  = lead.barrier;
	MktDsmSamples	 *dsm_samples	  = lead.dsm_samples;
	char			 *centroids_base  = lead.centroids_base;
	float			 *cents			  = lead.cents;
	char			 *km_workers_base = lead.km_workers_base;
	MktDsmRootAssign *dsm_ra		  = lead.dsm_ra;
	WalUsage		 *walusage		  = lead.walusage;
	BufferUsage		 *bufferusage	  = lead.bufferusage;
	int				  nparticipants	  = lead.nparticipants;
	uint32_t		  km_k			  = lead.km_k;
	Dimension		  dim			  = lead.dim;
	uint32_t		  nlist			  = lead.nlist;
	uint64_t		  rabitq_seed	  = lead.rabitq_seed;
	uint32_t		  fan_out		  = lead.fan_out;

	/*
	 * Introspection: name the dataset-scaling allocations up front (always
	 * logged, regardless of the GUC) so an OOM in any of them is
	 * pre-explained, and record the committed DSM size so the per-phase memory
	 * lines can add it. nlist here is the worst-case upper bound (the tree is
	 * not built yet).
	 */
	mkt_build_report_dsm_bytes(prog, (uint64_t)lead.dsm_total);
	mkt_build_report_planned_alloc(
			prog,
			(uint64_t)mkt_dsm_samples_size(
					nparticipants, lead.max_per_worker, dim),
			/* The tree is streamed to pages from a bounded ring of subtree
			 * slots (part of dsm_total); it is never held whole in memory. */
			0,
			/* Per-cluster encode references are rotated one at a time in
			 * mkt_posting_build_lists; no bulk nlist*dim array exists. */
			0,
			(uint64_t)lead.dsm_total);

	instr_time t_launch_start;
	INSTR_TIME_SET_CURRENT(t_launch_start);

	/*
	 * The leader participates in every phase, so it joins the dynamic barrier
	 * as a party (workers attach the same way as they start). Attaching before
	 * launching workers guarantees the leader is counted before any worker can
	 * arrive, so the barrier never advances a phase without it.
	 */
	BarrierAttach(barrier);

	/* ---- Launch workers + wait until they've all attached ---- */
	if (!mkt_pbuild_launch(pcxt, barrier))
		return false;
	/* ==== Leader participates in all phases as worker_id=0 ==== */

	/* ---- Phase 1: sampling (leader runs the worker body as participant 0)
	 * ---- */
	mkt_build_report_phase(prog, MKT_BUILD_PHASE_SAMPLE);
	mkt_pbuild_exec_sampling(
			0, heap, index, index_info, shared, dsm_samples, barrier);

	instr_time t_sample_end;
	INSTR_TIME_SET_CURRENT(t_sample_end);
	INSTR_TIME_SUBTRACT(t_sample_end, t_launch_start);
	mkt_debug(
			"mktann: phase 1 (sampling) %.1fms",
			INSTR_TIME_GET_MILLISEC(t_sample_end));

	instr_time t_km_start;
	INSTR_TIME_SET_CURRENT(t_km_start);

	/* ---- Phase 2: root k-means. The leader runs the worker body as
	 * participant 0; inside it additionally seeds the initial centroids and
	 * reduces the per-iteration accumulators (gated on participant 0). ---- */
	mkt_build_report_phase(prog, MKT_BUILD_PHASE_KMEANS);
	uint32_t km_iters = mkt_pbuild_exec_kmeans(
			0, shared, dsm_samples, centroids_base, km_workers_base, barrier);

	{
		instr_time t_km_elapsed;
		INSTR_TIME_SET_CURRENT(t_km_elapsed);
		INSTR_TIME_SUBTRACT(t_km_elapsed, t_km_start);
		mkt_debug(
				"mktann: root kmeans %.1fms (%u iters, k=%u)",
				INSTR_TIME_GET_MILLISEC(t_km_elapsed),
				km_iters,
				km_k);
	}

	/*
	 * Compute nlevels to decide flat vs hierarchical assembly. For
	 * nlevels == 1 (flat) the root k-means already produced every leaf, so
	 * the tree is built from the centroids directly. For nlevels >= 2 each
	 * participant builds the full subtree — to whatever depth nlist/fan_out
	 * needs — for the root children it owns, and the leader streams each to
	 * centroid pages a batch at a time (no in-RAM whole-tree assembly).
	 */
	uint32_t nlevels = 1;
	{
		uint32_t n = nlist;
		while (n > fan_out)
		{
			n = (n + fan_out - 1) / fan_out;
			nlevels++;
		}
	}

	HKMeansResult *tree = NULL;

	/* ---- Phase 2b: root assignment (leader as participant 0). ---- */
	mkt_pbuild_exec_root_assign(
			0, shared, dsm_samples, dsm_ra, centroids_base, barrier);

	/* ---- Batched streaming tree build --------------------------------------
	 * Workers build per-root-child subtrees into a bounded ring of slots; the
	 * leader records each batch's layout counts, keeps the blobs in a
	 * spillable store, and streams them to centroid pages. Peak subtree DSM is
	 * nparticipants slots, independent of nlist. ------------------------------ */
	mkt_build_report_phase(prog, MKT_BUILD_PHASE_SETUP);
	RaBitQParams	 *rq_params = mkt_rabitq_create(dim, rabitq_seed);
	MktCentroidFormat fmt		= shared->centroid_format;
	uint32_t		  max_ent	= mkt_centroid_max_entries_fmt(dim, fmt);
	char			 *subtrees_base =
			shm_toc_lookup(pcxt->toc, MKT_DSM_KEY_CHILD_SUBTREES, false);
	uint64_t slot_size = shared->subtree_slot_size;

	/*
	 * global_mean = mean of the LEAF centroids (the encoder centering) -- an
	 * unweighted per-cluster mean, not the per-vector sample mean. The
	 * quantization quality of every centroid and posting code depends on this
	 * anchor. Pages are written only after the PLAN pass has built every
	 * subtree, so
	 * the leaf-centroid sum is accumulated there (plan_batch_cb) and the mean
	 * is ready before any page is encoded. Filled per branch below.
	 */
	float *global_mean = mkt_alloc((size_t)dim * sizeof(float));

	BlockNumber first_centroid = 1; /* block 0 = metadata */
	BlockNumber first_posting  = 0;
	BlockNumber root_blk	   = InvalidBlockNumber;
	uint8_t		out_nlevels	   = (uint8_t)nlevels;

	if (nlevels >= 2)
	{
		uint32_t *nleaves_arr = mkt_alloc0((size_t)km_k * sizeof(uint32_t));
		uint32_t *pages_arr	  = mkt_alloc0((size_t)km_k * sizeof(uint32_t));

		/* PLAN pass: build subtrees, discover leaf + page counts (and the
		 * leaf-centroid sum for global_mean). */
		mkt_build_report_phase(prog, MKT_BUILD_PHASE_SUBTREES);
		PlanCbArg planarg = {
				.nleaves_arr	 = nleaves_arr,
				.pages_arr		 = pages_arr,
				.max_ent		 = max_ent,
				.subtree_nlevels = 1,
				.dim			 = dim,
				.leaf_sum		 = mkt_alloc0((size_t)dim * sizeof(double)),
				.store			 = mkt_pbuild_blobstore_begin(),
		};
		mkt_pbuild_stream_subtrees(
				0, nparticipants, dsm_samples, dsm_ra, cents, km_k, nlist,
				fan_out, dim, shared->metric, shared->km_max_iterations,
				subtrees_base, slot_size, barrier, plan_batch_cb, &planarg);

		/* The streamed tree's depth is the (uniform) subtree depth plus the root
		 * level. This is the actual written depth — k-means can produce subtrees
		 * deeper than the uniform target — and it is what the page-backed query
		 * must descend, so publish it (not the pre-k-means target nlevels). */
		out_nlevels = (uint8_t)(planarg.subtree_nlevels + 1);

		/* Root page(s) occupy the reserved block(s) at first_centroid; subtrees
		 * follow, so meta.first_centroid stays 1 (root written last, in place). */
		uint32_t  root_pages = (km_k + max_ent - 1) / max_ent;
		uint32_t *leaf_off	 = mkt_alloc((size_t)km_k * sizeof(uint32_t));
		uint32_t *block_off	 = mkt_alloc((size_t)km_k * sizeof(uint32_t));
		uint32_t	lo		 = 0;
		BlockNumber bo		 = 0;
		for (uint32_t c = 0; c < km_k; c++)
		{
			leaf_off[c]	 = lo;
			block_off[c] = (uint32_t)bo;
			lo += nleaves_arr[c];
			bo += pages_arr[c];
		}
		uint32_t	actual_nlist = lo;
		BlockNumber subtree_base = first_centroid + root_pages;
		first_posting			 = subtree_base + bo;

		/* The leaf-pt cache was sized for the REQUESTED nlist, but k-means
		 * can produce more leaves (a leaf-parent may split into up to fan_out
		 * leaves). Disable the cache -- workers fall back to the beam
		 * secondary -- rather than overrun the DSM region. */
		if ((uint64_t)actual_nlist * dim * sizeof(float) >
			shared->leaf_pt_bytes)
			shared->leaf_pt_bytes = 0;

		/* Leaf-centroid mean from the PLAN pass -> the encoder centering. */
		for (Dimension d = 0; d < dim; d++)
			global_mean[d] = actual_nlist > 0 ? (float)(planarg.leaf_sum[d] /
														(double)actual_nlist)
											  : 0.0f;
		if (shared->metric == DISTANCE_COSINE)
			mkt_l2_normalize(global_mean, dim);
		mkt_free(planarg.leaf_sum);
		planarg.leaf_sum = NULL;

		/* Head blocks are formula-derived: leaf c's head is first_posting + c, a
		 * contiguous head region of actual_nlist pages. Pre-extend the relation
		 * to cover the centroid pages + the head region so the write pass can
		 * write both at reserved blocks; continuation pages are appended past it
		 * during mkt_posting_build_lists. No O(nlist) reserve arrays. */
		mkt_storage_extend(storage, first_posting + actual_nlist);

		/* Streaming pass: read each subtree blob back from the store (child
		 * order matches the append order) and stream its centroid + head
		 * pages to the reserved block range, then the root page. Leader-only:
		 * the PLAN pass already produced every subtree, so the workers have
		 * nothing to contribute here and run no barriers for this phase. */
		mkt_build_report_phase(prog, MKT_BUILD_PHASE_CENTROID);
		BlockNumber *subtree_root_blk =
				mkt_alloc((size_t)km_k * sizeof(BlockNumber));
		LeaderHeadCtx head = {
				.storage	   = storage,
				.rq_params	   = rq_params,
				.dim		   = dim,
				.fastscan	   = shared->fastscan,
				.first_posting = first_posting,
				.pt			   = mkt_alloc((size_t)dim * sizeof(float)),
				.pt_cache	   = shared->leaf_pt_bytes > 0
									   ? shm_toc_lookup(
											 pcxt->toc,
											 MKT_DSM_KEY_LEAF_PT,
											 false)
									   : NULL,
		};
		HKMeansResult *blob = mkt_alloc(slot_size);
		mkt_pbuild_blobstore_rewind(planarg.store);
		for (uint32_t c = 0; c < km_k; c++)
		{
			(void)mkt_pbuild_blobstore_get(planarg.store, blob, slot_size);
			BlockNumber base_blk = subtree_base + block_off[c];
			uint32_t	pages;
			subtree_root_blk[c] = mkt_write_subtree_streaming(
					storage, blob, dim, fan_out, fmt, rq_params, global_mean,
					first_posting, leaf_off[c], base_blk, leader_write_head,
					&head, &pages);
		}
		mkt_free(blob);
		mkt_pbuild_blobstore_end(planarg.store);
		planarg.store = NULL;

		/* Root centroid page at the reserved first_centroid (children = subtree
		 * roots). Written last, but in place, so first_centroid stays 1. */
		root_blk = first_centroid;
		if (fmt == MKT_CENTROID_FMT_FASTSCAN)
			mkt_centroid_write_fastscan_pages(
					storage, dim, km_k, 0, 0, rq_params, cents, global_mean,
					subtree_root_blk, root_blk);
		else
		{
			CentroidEncoderState est;
			CentroidEncoder		*enc = centroid_encoder_init(
					&est, fmt, cents, dim, rq_params, global_mean);
			mkt_centroid_write_pages(
					storage, dim, km_k, fmt, 0, 0, (uint16_t)fan_out, enc,
					subtree_root_blk, NULL, root_blk);
		}

		mkt_free(head.pt);
		mkt_free(subtree_root_blk);
		mkt_free(leaf_off);
		mkt_free(block_off);
		mkt_free(nleaves_arr);
		mkt_free(pages_arr);
		nlist = actual_nlist;
	}
	else
	{
		/* Flat (nlevels == 1): cents already holds every leaf centroid; stream
		 * the one-level tree directly (root = leaf-parent at first_centroid). */
		if (shared->metric == DISTANCE_COSINE)
			for (uint32_t c = 0; c < km_k; c++)
				mkt_l2_normalize(cents + (size_t)c * dim, dim);

		HKMeansResult *flat = mkt_hkmeans_build_flat(cents, km_k, fan_out, dim);
		if (flat == NULL)
		{
			mkt_free(global_mean);
			WaitForParallelWorkersToFinish(pcxt);
			mkt_pbuild_teardown(pcxt);
			return false;
		}

		uint32_t	 actual_nlist = flat->nleaves;
		BlockNumber *nfb =
				mkt_alloc((size_t)flat->nnodes * sizeof(BlockNumber));
		uint32_t centroid_pages =
				(uint32_t)mkt_compute_centroid_layout(flat, max_ent, 0, nfb);
		mkt_free(nfb);
		first_posting = first_centroid + centroid_pages;

		/* Cache sized for the worst-case leaf count; disable rather than
		 * overrun if k-means still exceeded it. */
		if ((uint64_t)actual_nlist * dim * sizeof(float) >
			shared->leaf_pt_bytes)
			shared->leaf_pt_bytes = 0;

		/* Leaf-centroid mean -> the encoder centering (flat tree in hand). */
		mkt_vector_mean(
				hk_leaf_centroids(flat), actual_nlist, dim, global_mean);
		if (shared->metric == DISTANCE_COSINE)
			mkt_l2_normalize(global_mean, dim);

		/* Head region: actual_nlist pages at first_posting (leaf c -> head
		 * first_posting + c). Pre-extend to cover centroid + head region;
		 * continuations append past it. No O(nlist) reserve. */
		mkt_storage_extend(storage, first_posting + actual_nlist);

		mkt_build_report_phase(prog, MKT_BUILD_PHASE_CENTROID);
		LeaderHeadCtx head = {
				.storage	   = storage,
				.rq_params	   = rq_params,
				.dim		   = dim,
				.fastscan	   = shared->fastscan,
				.first_posting = first_posting,
				.pt			   = mkt_alloc((size_t)dim * sizeof(float)),
				.pt_cache	   = shared->leaf_pt_bytes > 0
									   ? shm_toc_lookup(
											 pcxt->toc,
											 MKT_DSM_KEY_LEAF_PT,
											 false)
									   : NULL,
		};
		uint32_t pages;
		root_blk = mkt_write_subtree_streaming(
				storage, flat, dim, fan_out, fmt, rq_params, global_mean,
				first_posting, 0, first_centroid, leader_write_head, &head,
				&pages);
		mkt_free(head.pt);
		out_nlevels = (uint8_t)flat->nlevels;
		mkt_free(flat);
		nlist = actual_nlist;
	}

	/* Publish the routing state the workers read in phase 3: nlist, the tree
	 * root block + depth, the posting-head base (leaf c's head = first_posting +
	 * c), the global mean, and the page store. */
	shared->nlist		   = nlist;
	shared->first_centroid = root_blk;
	shared->nlevels		   = out_nlevels;
	shared->first_posting  = first_posting;
	if (out_first_posting)
		*out_first_posting = first_posting;
	{
		float *dsm_gmean =
				shm_toc_lookup(pcxt->toc, MKT_DSM_KEY_GLOBAL_MEAN, false);
		memcpy(dsm_gmean, global_mean, (size_t)dim * sizeof(float));
	}
	mkt_pbuild_publish_storage(shared, storage);

	if (out_global_mean)
		*out_global_mean = global_mean; /* caller owns it (metadata write) */
	else
		mkt_free(global_mean);
	if (out_centroids_written)
		*out_centroids_written = true;

	/* Lightweight metadata carrier; no in-RAM tree exists to hand back. */
	tree		  = mkt_alloc0(sizeof(HKMeansResult));
	tree->nleaves = nlist;
	tree->nlevels = out_nlevels;
	tree->dim	  = dim;

	/* Phase 3: workers scan + route page-backed + encode + sort; the leader
	 * merges. Report before releasing workers so progress reflects the whole
	 * (multi-hour at scale) scan. The seam fires the "mktann-build-load" hook. */
	mkt_build_report_phase(prog, MKT_BUILD_PHASE_SCAN_PARALLEL);

	/* Initialize the shared cluster sorter for the launched-worker count BEFORE
	 * the ready barrier, so it is ready when workers attach in phase 3. The
	 * leader merges only (it does not sort a share). */
	void *sortshared =
			shm_toc_lookup(pcxt->toc, MKT_DSM_KEY_SORTSHARED, false);
	mkt_pbuild_sort_shared_init(
			sortshared, pcxt->nworkers_launched, pcxt->seg);

	/* Barrier: centroid/head pages written + routing state published + sorter
	 * ready; workers build their page-backed router next. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 2.5: page-backed full-table refine (only when subsampled) ----
	 * The workers route + accumulate; the leader (participant 0) clears the tiled
	 * accumulator, resets the scan per tile, and rewrites each leaf's head-page
	 * pt_centroid to the full-table mean. Gated on shared->refine_iters, matching
	 * the workers, so the internal barriers stay in lockstep. */
	if (shared->refine_iters > 0)
	{
		mkt_build_report_phase(prog, MKT_BUILD_PHASE_REFINE);
		MktDsmRefineAccum *accum =
				shm_toc_lookup(pcxt->toc, MKT_DSM_KEY_REFINE_ACCUM, false);
		LeaderHeadCtx rhead = {
				.storage	   = storage,
				.rq_params	   = rq_params,
				.dim		   = dim,
				.fastscan	   = shared->fastscan,
				.first_posting = first_posting,
				.pt			   = mkt_alloc((size_t)dim * sizeof(float)),
				.pt_cache	   = shared->leaf_pt_bytes > 0
									   ? shm_toc_lookup(
											 pcxt->toc,
											 MKT_DSM_KEY_LEAF_PT,
											 false)
									   : NULL,
		};
		mkt_pbuild_exec_refine_paged(
				0, heap, index, index_info, shared, NULL, first_posting, accum,
				barrier, leader_write_head, &rhead);
		mkt_free(rhead.pt);
		mkt_build_report_phase(prog, MKT_BUILD_PHASE_SCAN_PARALLEL);
	}

	/* Re-init the scan for the posting phase (the refine passes above consumed
	 * it). Guarded by the barrier below so no worker scans before the reset. */
	mkt_pbuild_rescan(heap, shared);

	/* Barrier: scan reset for the posting phase; workers start the page-backed
	 * posting scan+sort. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	instr_time t_scan_start;
	INSTR_TIME_SET_CURRENT(t_scan_start);

	/* Barrier: wait until every worker has finished sorting its run, then merge
	 * and build. The leader does not scan; the workers cover the heap. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	BarrierDetach(barrier);

	WaitForParallelWorkersToFinish(pcxt);

	instr_time t_scan_end;
	INSTR_TIME_SET_CURRENT(t_scan_end);
	INSTR_TIME_SUBTRACT(t_scan_end, t_scan_start);

	for (int i = 0; i < pcxt->nworkers_launched; i++)
		InstrAccumParallelQuery(&bufferusage[i], &walusage[i]);

	*out_heap_tuples = shared->reltuples;
	*out_indtuples	 = shared->indtuples;
	*out_soar_dupes	 = shared->soar_dupes;
	*out_tree		 = tree;

	/* The workers cover the heap cooperatively while the leader blocks on the
	 * scan barrier above, so there is no leader-side loop to advance the % mid
	 * scan; publish the final scanned count now that the workers have
	 * finished.
	 */
	mkt_build_report_progress(prog, shared->indtuples);

	instr_time t_merge_start;
	INSTR_TIME_SET_CURRENT(t_merge_start);

	/*
	 * Merge the workers' sorted runs and build each cluster's posting list
	 * with a single resident page builder, in cluster order (same loop as the
	 * serial path). Entries arrive grouped by cluster, so there is no
	 * partial-page fold and no chain to splice.
	 */
	mkt_build_report_phase(prog, MKT_BUILD_PHASE_POSTING);
	MktSorter *sorter = mkt_pbuild_sort_begin(
			sortshared,
			pcxt->seg,
			0,
			pcxt->nworkers_launched,
			true,
			(uint32_t)mkt_posting_entry_size(dim),
			shared->work_mem_kb);
	mkt_posting_build_lists(
			sorter,
			storage,
			nlist,
			dim,
			shared->fastscan,
			rq_params,
			/* ref_vecs = NULL: read each list's pt_centroid from the head page
			 * pre-written above (page-backed), not from the in-RAM tree. */
			NULL,
			first_posting);

	uint32_t total_pages = RelationGetNumberOfBlocks(index) - first_posting;

	instr_time t_merge_end;
	INSTR_TIME_SET_CURRENT(t_merge_end);
	INSTR_TIME_SUBTRACT(t_merge_end, t_merge_start);

	mkt_debug(
			"mktann: parallel streaming build with %d workers, "
			"%u clusters, %u pages, "
			"scan+drain %.1fms, finalize %.1fms",
			pcxt->nworkers_launched,
			nlist,
			total_pages,
			INSTR_TIME_GET_MILLISEC(t_scan_end),
			INSTR_TIME_GET_MILLISEC(t_merge_end));

	mkt_pbuild_teardown(pcxt);

	return true;
}
