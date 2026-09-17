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

#include <inttypes.h>
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
#include "quant/fastscan.h"
#include "types/halfvec.h"
#include "types/vector.h"

#ifndef MKT_STANDALONE
#include "build.h"
#include "meta.h"
#include "storage.h"
#include "support_pg.h"
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
 * received no vectors. The centroid tree's leaf entries reference
 * first_posting
 * + c for every c in [0, nlist) (see mkt_write_centroid_tree), so an empty
 * cluster still needs a valid (empty) head block to point at. A loop over the
 * cluster index emits a head for every cluster uniformly — an empty cluster's
 * inner while simply does not run and finish() returns an empty head. A
 * drain-until-empty loop would only produce heads for clusters present in the
 * stream and would have to separately backfill empty heads for gap clusters
 * and for all trailing clusters past the last one seen.
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
		BlockNumber			first_posting)
{
	mkt_pbuild_sort_performsort(sorter);

	/*
	 * Each list's head was pre-written with its pt_centroid (P^T * centroid,
	 * the RaBitQ encode reference) during the centroid build: adopt it and
	 * append the sorted entries in place. No second head construction, no
	 * in-RAM float centroids — the tree can be freed before this runs — and
	 * a cluster with no entries keeps its on-disk head untouched.
	 */
	uint32_t	cur_cluster = 0;
	const void *entry		= NULL;
	bool		have = mkt_pbuild_sort_getnext(sorter, &cur_cluster, &entry);
	for (uint32_t c = 0; c < nlist; c++)
	{
		/* Head block of cluster c is the formula first_posting + c; the head
		 * region [first_posting, first_posting + nlist) was pre-extended and
		 * its pt_centroid written during the centroid streaming. Continuation
		 * pages are appended at the relation's end and chained (no per-cluster
		 * reserve), so no O(nlist) reserve arrays are needed. Adopt the
		 * pre-written head and append in place: no second head construction,
		 * and a cluster with no entries keeps its on-disk head untouched. */
		BlockNumber		  head_blk = first_posting + c;
		MktPostingBuilder hb;
		mkt_posting_builder_adopt_head(
				&hb, storage, rq_params, dim, c, head_blk, fastscan);

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
	mkt_pbuild_sort_end(sorter);
}

/* ----------------------------------------------------------------
 * Batched streaming tree build — leader-side callbacks
 * ---------------------------------------------------------------- */

/* PLAN-pass batch callback: record each subtree's leaf count + centroid-page
 * count (no writes) so the leader can size the reserve + block layout, and
 * keep the blob in the spillable store for the streaming pass to read back
 * in the same (child) order. */
typedef struct PlanCbArg
{
	uint32_t *nleaves_arr; /* [km_k] */
	uint32_t *pages_arr;   /* [km_k] */
	uint32_t  max_ent;
	uint32_t  subtree_nlevels; /* actual depth of the (uniform) subtrees */
	Dimension dim;
	double	 *leaf_sum;	 /* [dim] leaf-centroid sum across all subtrees, for
						  * the leaf_mean the leader uses as global_mean */
	MktBlobStore *store; /* subtree blobs, in child order */
} PlanCbArg;

static void
plan_batch_cb(
		void		   *arg,
		const uint32_t *children,
		uint32_t		bs,
		char		   *base,
		uint64_t		slot_size)
{
	PlanCbArg *a = (PlanCbArg *)arg;
	for (uint32_t s = 0; s < bs; s++)
	{
		const HKMeansResult *sub = (const HKMeansResult *)
				mkt_dsm_child_subtree(base, s, slot_size);
		uint32_t	 child = children[s];
		BlockNumber *nfb   = mkt_alloc(
				  (size_t)sub->nnodes * sizeof(BlockNumber));
		/* Subtrees are built to a uniform depth, so any non-empty one gives
		 * the streamed tree's subtree depth (full depth = this + 1 root
		 * level). */
		if (sub->nleaves > 0)
			a->subtree_nlevels = sub->nlevels;
		a->nleaves_arr[child] = sub->nleaves;
		a->pages_arr[child]	  = (uint32_t)
				mkt_compute_centroid_layout(sub, a->max_ent, 0, nfb);
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

/* What both routing-tree shapes hand back to do_parallel_build: the layout
 * facts the leader publishes to the workers before phase 3. */
typedef struct TreeLayout
{
	BlockNumber first_posting; /* leaf c's head = first_posting + c */
	BlockNumber root_blk;	   /* tree root page */
	uint8_t		nlevels;	   /* written depth */
	uint32_t	nlist;		   /* actual leaf count */
} TreeLayout;

/*
 * Assemble and write the routing tree for the hierarchical shape
 * (nlevels >= 2): each participant builds the subtree of every root child
 * it owns; the leader records each batch's layout counts (PLAN), computes
 * the block layout and the leaf-centroid mean, pre-extends the relation,
 * replays the spilled subtree blobs to pages, and writes the root page
 * above them. Fills global_mean and the layout the caller publishes.
 */
static bool
build_routing_tree_batched(
		MktBuildShared			  *shared,
		MktStorage				  *storage,
		MktBuildProgress		  *prog,
		MktDsmSamples			  *dsm_samples,
		MktDsmRootAssign		  *dsm_ra,
		float					  *cents,
		uint32_t				   km_k,
		uint32_t				   nlist,
		uint32_t				   fan_out,
		Barrier					  *barrier,
		RaBitQParams			  *rq_params,
		uint32_t				   max_ent,
		BlockNumber				   first_centroid,
		float					  *global_mean,
		MktExactCentroidCollector *collector,
		TreeLayout				  *out)
{
	Dimension		  dim			= shared->dim;
	int				  nparticipants = shared->nparticipants;
	MktCentroidFormat fmt			= shared->centroid_format;

	/* Size the ring slot from the actual per-child sample counts: a subtree
	 * can never hold more leaves than the samples routed to its root child,
	 * which caps the slot far below the analytic worst case (nlist here is
	 * the worst-case sizing bound, up to fan_out x the requested count). */
	uint32_t max_cc = 1;
	{
		uint32_t *child_count = mkt_alloc0((size_t)km_k * sizeof(uint32_t));
		mkt_pbuild_count_children(
				dsm_samples, dsm_ra, nparticipants, km_k, child_count);
		for (uint32_t c = 0; c < km_k; c++)
			if (child_count[c] > max_cc)
				max_cc = child_count[c];
		mkt_free(child_count);
	}
	uint32_t nlist_c = (nlist + fan_out - 1) / fan_out;
	uint64_t slot_size =
			mkt_hkmeans_max_blob_size_capped(nlist_c, fan_out, dim, max_cc);

	/* The slot must fit one allocation (the leader reads each spilled blob
	 * back through a palloc'd buffer), which also keeps the blob format's
	 * 32-bit interior offsets valid. */
	if (slot_size > (uint64_t)MaxAllocSize)
		mkt_error(
				"mktann: subtree slot %llu MB exceeds the allocation limit "
				"(nlist %u, fan_out %u); increase fan_out or decrease nlist",
				(unsigned long long)(slot_size >> 20),
				nlist,
				fan_out);
	/* The ring is the build's only region outside the sample budget; hold
	 * it to the same standard. work_mem_kb == 0 = unbudgeted back-end. */
	if (shared->work_mem_kb > 0 &&
		(uint64_t)nparticipants * slot_size >
				(uint64_t)shared->work_mem_kb * 1024)
		mkt_error(
				"mktann: subtree ring %llu MB exceeds maintenance_work_mem "
				"(nlist %u, fan_out %u); increase maintenance_work_mem or "
				"fan_out",
				(unsigned long long)(((uint64_t)nparticipants * slot_size) >>
									 20),
				nlist,
				fan_out);

	/* Ring barrier: create + publish (handle, slot size) before arriving;
	 * the workers attach after. */
	void *ring_seg		= NULL;
	char *subtrees_base = mkt_pbuild_subtree_ring_create(
			shared, nparticipants, slot_size, &ring_seg);
	mkt_debug(
			"mktann: subtree ring %d x %llu KB (largest child %u samples)",
			nparticipants,
			(unsigned long long)(slot_size >> 10),
			max_cc);
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

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
	/* The batch schedule is largest-first; the blob store receives the
	 * subtrees in that order, so the replay below needs the same order
	 * to place each blob at its child's reserved block range. */
	uint32_t *child_order = mkt_alloc((size_t)km_k * sizeof(uint32_t));
	mkt_pbuild_stream_subtrees(
			0,
			nparticipants,
			dsm_samples,
			dsm_ra,
			cents,
			km_k,
			nlist,
			fan_out,
			dim,
			shared->metric,
			shared->km_max_iterations,
			subtrees_base,
			slot_size,
			barrier,
			plan_batch_cb,
			&planarg,
			child_order);

	/* Every batch is consumed into the blob store; the ring is dead before
	 * the write pass starts. */
	mkt_pbuild_subtree_ring_release(ring_seg);

	/* Root page(s) occupy the reserved block(s) at first_centroid;
	 * subtrees follow, so meta.first_centroid stays 1 (root written last,
	 * in place). */
	uint32_t	root_pages = (km_k + max_ent - 1) / max_ent;
	uint32_t   *leaf_off   = mkt_alloc((size_t)km_k * sizeof(uint32_t));
	uint32_t   *block_off  = mkt_alloc((size_t)km_k * sizeof(uint32_t));
	uint32_t	lo		   = 0;
	BlockNumber bo		   = 0;
	for (uint32_t c = 0; c < km_k; c++)
	{
		leaf_off[c]	 = lo;
		block_off[c] = (uint32_t)bo;
		lo += nleaves_arr[c];
		bo += pages_arr[c];
	}
	uint32_t	actual_nlist  = lo;
	BlockNumber subtree_base  = first_centroid + root_pages;
	BlockNumber first_posting = subtree_base + bo;

	/* The centroid-page region [first_centroid, first_posting) is now
	 * sized; the streaming pass below collects every internal node's
	 * exact centroids into it (the root included). */
	if (collector != NULL)
		mkt_exact_centroid_collector_init(
				collector,
				dim,
				fmt,
				first_centroid,
				(uint32_t)(first_posting - first_centroid),
				mkt_exact_centroid_budget(shared->work_mem_kb),
				mkt_exact_centroid_expected_slots(actual_nlist, fan_out));

	/* Leaf-centroid mean from the PLAN pass -> the encoder centering. */
	for (Dimension d = 0; d < dim; d++)
		global_mean[d] = actual_nlist > 0 ? (float)(planarg.leaf_sum[d] /
													(double)actual_nlist)
										  : 0.0f;
	if (shared->metric == DISTANCE_COSINE)
		mkt_l2_normalize(global_mean, dim);
	mkt_free(planarg.leaf_sum);
	planarg.leaf_sum = NULL;

	/* Head blocks are formula-derived: leaf c's head is first_posting + c,
	 * a contiguous head region of actual_nlist pages. Pre-extend the
	 * relation to cover the centroid pages + the head region so the write
	 * pass can write both at reserved blocks; continuation pages are
	 * appended past it during mkt_posting_build_lists. No O(nlist) reserve
	 * arrays. */
	mkt_build_reserve_layout(storage, first_posting + actual_nlist);

	/* Streaming pass: read each subtree blob back from the store (child
	 * order matches the append order) and stream its centroid + head
	 * pages to the reserved block range, then the root page. Leader-only:
	 * the PLAN pass already produced every subtree, so the workers have
	 * nothing to contribute here and run no barriers for this phase. */
	mkt_build_report_phase(prog, MKT_BUILD_PHASE_CENTROID);
	BlockNumber *subtree_root_blk = mkt_alloc(
			(size_t)km_k * sizeof(BlockNumber));
	MktHeadWriteCtx head;
	mkt_head_write_ctx_init(
			&head, storage, rq_params, dim, shared->fastscan, first_posting);
	HKMeansResult *blob = mkt_alloc(slot_size);
	mkt_pbuild_blobstore_rewind(planarg.store);
	for (uint32_t i = 0; i < km_k; i++)
	{
		/* Blobs arrive in the largest-first batch order; c is the child
		 * whose reserved block range this blob belongs to. */
		uint32_t c = child_order[i];
		(void)mkt_pbuild_blobstore_get(planarg.store, blob, slot_size);
		BlockNumber base_blk = subtree_base + block_off[c];
		subtree_root_blk[c]	 = mkt_routing_subtree_write(
				 storage,
				 blob,
				 dim,
				 shared->metric,
				 fan_out,
				 1, /* subtrees hang off the level-0 root */
				 fmt,
				 rq_params,
				 global_mean,
				 first_posting,
				 leaf_off[c],
				 base_blk,
				 mkt_write_leaf_head,
				 &head,
				 collector);
	}
	mkt_free(blob);
	mkt_free(child_order);
	mkt_pbuild_blobstore_end(planarg.store);
	planarg.store = NULL;

	/* Root centroid page at the reserved first_centroid (children =
	 * subtree roots). Written last, but in place, so first_centroid
	 * stays 1. */
	BlockNumber root_blk = first_centroid;
	mkt_centroid_write_node(
			storage,
			dim,
			cents,
			km_k,
			fmt,
			0,
			0,
			(uint16_t)fan_out,
			rq_params,
			global_mean,
			subtree_root_blk,
			NULL,
			root_blk,
			collector);

	mkt_head_write_ctx_cleanup(&head);
	mkt_free(subtree_root_blk);
	mkt_free(leaf_off);
	mkt_free(block_off);
	mkt_free(nleaves_arr);
	mkt_free(pages_arr);

	out->first_posting = first_posting;
	out->root_blk	   = root_blk;
	out->nlevels	   = (uint8_t)(planarg.subtree_nlevels + 1);
	out->nlist		   = actual_nlist;
	return true;
}

/*
 * Assemble and write the routing tree for the flat shape (nlevels == 1):
 * the root k-means centroids already ARE the leaves, so stream the
 * one-level tree directly (root = leaf-parent page at first_centroid).
 * Fills global_mean and the layout the caller publishes; false when the
 * flat carrier cannot be built.
 */
static bool
build_routing_tree_flat(
		MktBuildShared	 *shared,
		MktStorage		 *storage,
		MktBuildProgress *prog,
		float			 *cents,
		uint32_t		  km_k,
		uint32_t		  fan_out,
		RaBitQParams	 *rq_params,
		uint32_t		  max_ent,
		BlockNumber		  first_centroid,
		float			 *global_mean,
		TreeLayout		 *out)
{
	Dimension		  dim = shared->dim;
	MktCentroidFormat fmt = shared->centroid_format;

	/* Flat (nlevels == 1): cents already holds every leaf centroid; stream
	 * the one-level tree directly (root = leaf-parent at first_centroid).
	 */
	if (shared->metric == DISTANCE_COSINE)
		for (uint32_t c = 0; c < km_k; c++)
			mkt_l2_normalize(cents + (size_t)c * dim, dim);

	HKMeansResult *flat = mkt_hkmeans_build_flat(cents, km_k, fan_out, dim);
	if (flat == NULL)
		return false;

	uint32_t	 actual_nlist = flat->nleaves;
	BlockNumber *nfb = mkt_alloc((size_t)flat->nnodes * sizeof(BlockNumber));
	uint32_t	 centroid_pages = (uint32_t)
			mkt_compute_centroid_layout(flat, max_ent, 0, nfb);
	mkt_free(nfb);
	BlockNumber first_posting = first_centroid + centroid_pages;

	/* Leaf-centroid mean -> the encoder centering (flat tree in hand). */
	mkt_vector_mean(hk_leaf_centroids(flat), actual_nlist, dim, global_mean);
	if (shared->metric == DISTANCE_COSINE)
		mkt_l2_normalize(global_mean, dim);

	/* Head region: actual_nlist pages at first_posting (leaf c -> head
	 * first_posting + c). Pre-extend to cover centroid + head region;
	 * continuations append past it. No O(nlist) reserve. */
	mkt_build_reserve_layout(storage, first_posting + actual_nlist);

	mkt_build_report_phase(prog, MKT_BUILD_PHASE_CENTROID);
	MktHeadWriteCtx head;
	mkt_head_write_ctx_init(
			&head, storage, rq_params, dim, shared->fastscan, first_posting);
	BlockNumber root_blk = mkt_routing_subtree_write(
			storage,
			flat,
			dim,
			shared->metric,
			fan_out,
			0, /* the flat tree IS the root level */
			fmt,
			rq_params,
			global_mean,
			first_posting,
			0,
			first_centroid,
			mkt_write_leaf_head,
			&head,
			/* the flat tree's only level is the leaf level — nothing
			 * internal to collect */
			NULL);
	mkt_head_write_ctx_cleanup(&head);

	out->first_posting = first_posting;
	out->root_blk	   = root_blk;
	out->nlevels	   = (uint8_t)flat->nlevels;
	out->nlist		   = actual_nlist;
	mkt_free(flat);
	return true;
}

bool
do_parallel_build(
		Relation				 heap,
		Relation				 index,
		struct IndexInfo		*index_info,
		const MktBuildConfig	*config,
		MktStorage				*storage,
		struct MktBuildProgress *prog,
		uint32_t				*out_nlist,
		uint8_t					*out_tree_nlevels,
		double					*out_heap_tuples,
		double					*out_indtuples,
		double					*out_soar_dupes,
		float				   **out_global_mean,
		BlockNumber				*out_first_posting)
{
	int nworkers = index_info->ii_ParallelWorkers;

	*out_nlist		  = 0;
	*out_tree_nlevels = 0;
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
	if (!mkt_pbuild_launch(pcxt, barrier, shared))
	{
		/* Teardown already ran; hand the sample segment back too so the
		 * serial fallback starts from a clean budget. */
		mkt_pbuild_samples_release(dsm_samples, lead.sample_seg);
		return false;
	}

	/* The launch may have narrowed the participant count to the party that
	 * actually attached; partition the phases below over that count. */
	nparticipants = shared->nparticipants;

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
	uint32_t nlevels = mkt_hkmeans_nlevels(nlist, fan_out);

	/* ---- Phase 2b: root assignment (leader as participant 0). ---- */
	mkt_pbuild_exec_root_assign(
			0, shared, dsm_samples, dsm_ra, centroids_base, barrier);

	/* ---- Batched streaming tree build --------------------------------------
	 * Workers build per-root-child subtrees into a bounded ring of slots; the
	 * leader records each batch's layout counts, keeps the blobs in a
	 * spillable store, and streams them to centroid pages. Peak subtree DSM is
	 * nparticipants slots, independent of nlist.
	 * ------------------------------ */
	mkt_build_report_phase(prog, MKT_BUILD_PHASE_SETUP);
	RaBitQParams	 *rq_params = mkt_rabitq_create(dim, rabitq_seed);
	MktCentroidFormat fmt		= shared->centroid_format;
	uint32_t		  max_ent	= mkt_centroid_max_entries_fmt(dim, fmt);

	/*
	 * global_mean = mean of the LEAF centroids (the encoder centering) -- an
	 * unweighted per-cluster mean, not the per-vector sample mean. The
	 * quantization quality of every centroid and posting code depends on this
	 * anchor. Pages are written only after the PLAN pass has built every
	 * subtree, so
	 * the leaf-centroid sum is accumulated there (plan_batch_cb) and the mean
	 * is ready before any page is encoded. Filled per branch below.
	 */
	const size_t vec_nbytes	 = (size_t)dim * sizeof(float);
	float		*global_mean = mkt_alloc(vec_nbytes);

	BlockNumber first_centroid = MKT_FIRST_CENTROID_BLKNO;
	BlockNumber first_posting  = 0;
	BlockNumber root_blk	   = InvalidBlockNumber;
	uint8_t		out_nlevels	   = (uint8_t)nlevels;

	/* Exact internal-centroid collection for the phase-2.5/3 build
	 * descent (see MktExactCentroidCollector in index_build.h). */
	MktExactCentroidCollector  exact_centroids = {0};
	MktExactCentroidCollector *collector =
			mkt_exact_centroid_enabled(nlevels, fmt) ? &exact_centroids : NULL;

	TreeLayout layout;
	bool	   tree_ok;
	if (nlevels >= 2)
		tree_ok = build_routing_tree_batched(
				shared,
				storage,
				prog,
				dsm_samples,
				dsm_ra,
				cents,
				km_k,
				nlist,
				fan_out,
				barrier,
				rq_params,
				max_ent,
				first_centroid,
				global_mean,
				collector,
				&layout);
	else
		tree_ok = build_routing_tree_flat(
				shared,
				storage,
				prog,
				cents,
				km_k,
				fan_out,
				rq_params,
				max_ent,
				first_centroid,
				global_mean,
				&layout);
	if (!tree_ok)
	{
		if (collector != NULL)
			mkt_exact_centroid_collector_cleanup(collector);
		mkt_free(global_mean);
		mkt_pbuild_samples_release(dsm_samples, lead.sample_seg);
		WaitForParallelWorkersToFinish(pcxt);
		mkt_pbuild_teardown(pcxt);
		return false;
	}
	nlist		  = layout.nlist;
	first_posting = layout.first_posting;
	root_blk	  = layout.root_blk;
	out_nlevels	  = layout.nlevels;

	/* Publish the exact internal-node centroids the tree write collected,
	 * for the workers' phase-2.5/3 build descent (exact-centroid seam; must
	 * precede the tree-ready barrier). Without a collector the collection
	 * is an empty header and the workers' scoring hook stays inert. */
	void *exact_seg	  = NULL;
	char *exact_cents = mkt_pbuild_exact_centroids_create(
			shared, mkt_exact_centroid_collection_size(collector), &exact_seg);
	mkt_exact_centroid_collection_write(collector, exact_cents);
	if (collector != NULL)
		mkt_exact_centroid_collector_cleanup(collector);

	/* Publish the routing state the workers read in phase 3: nlist, the tree
	 * root block + depth, the posting-head base (leaf c's head = first_posting
	 * + c), the global mean, and the page store. */
	shared->nlist		   = nlist;
	shared->first_centroid = root_blk;
	shared->nlevels		   = out_nlevels;
	shared->first_posting  = first_posting;
	if (out_first_posting)
		*out_first_posting = first_posting;
	{
		float *dsm_gmean =
				shm_toc_lookup(pcxt->toc, MKT_DSM_KEY_GLOBAL_MEAN, false);
		memcpy(dsm_gmean, global_mean, vec_nbytes);
	}
	mkt_pbuild_publish_storage(shared, storage);

	if (out_global_mean)
		*out_global_mean = global_mean; /* caller owns it (metadata write) */
	else
		mkt_free(global_mean);

	/* No in-RAM tree exists; the caller's metadata write needs only the
	 * shape. */
	*out_nlist		  = nlist;
	*out_tree_nlevels = out_nlevels;

	/* The refine decision, now that the actual leaf count is known: refine
	 * only when the sample was bounded below the table AND the leaves are
	 * sample-thin -- a leaf's encode reference is a sample mean whose error
	 * shrinks with its sample count, so past the threshold the full-table
	 * scan recomputes what the sample already got right. Published before
	 * the ready barrier; workers gate the refine phase (and its barriers)
	 * on it after that barrier. */
	{
		uint64_t collected = 0;
		uint64_t seen	   = 0;
		for (int t = 0; t < nparticipants; t++)
		{
			collected += mkt_dsm_sample_counts(dsm_samples)[t];
			seen += mkt_dsm_sample_seen(dsm_samples)[t];
		}
		/* kept < seen means the sampling scans skipped real rows, so the
		 * sample is a strict subset of the table (kept == seen means the
		 * sample IS the table -- nothing to refine from). */
		shared->refine = shared->refine_threshold > 0 && collected < seen &&
						 shared->refine_tile_cap > 0 && nlist > 0 &&
						 collected / nlist <
								 (uint64_t)shared->refine_threshold;
		mkt_debug(
				"mktann: refine gate: kept=%" PRIu64 " seen=%" PRIu64
				" nlist=%u threshold=%u -> %s",
				collected,
				seen,
				nlist,
				shared->refine_threshold,
				shared->refine ? "refine" : "skip");
	}

	/* The samples are dead (their last readers were the subtree builders);
	 * lay the refine accumulator over them before the ready barrier so the
	 * workers -- who pass that barrier ahead of the refine phase -- see an
	 * initialized header. */
	MktDsmRefineAccum *refine_accum = NULL;
	if (shared->refine)
	{
		refine_accum		  = mkt_pbuild_refine_overlay(dsm_samples);
		refine_accum->nleaves = shared->refine_tile_cap;
		refine_accum->dim	  = dim;
	}

	/* Initialize the shared cluster sorter for the launched-worker count
	 * BEFORE the ready barrier, so it is ready when workers attach in phase 3.
	 * The leader merges only (it does not sort a share). */
	void *sortshared =
			shm_toc_lookup(pcxt->toc, MKT_DSM_KEY_SORTSHARED, false);
	mkt_pbuild_sort_shared_init(
			sortshared, pcxt->nworkers_launched, pcxt->seg);

	/* Barrier: centroid/head pages written + routing state published + sorter
	 * ready; workers build their page-backed router next. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	/* ---- Phase 2.5: page-backed full-table refine (only when subsampled)
	 * ---- The workers route + accumulate; the leader (participant 0) clears
	 * the tiled accumulator, resets the scan per tile, and rewrites each
	 * leaf's head-page pt_centroid to the full-table mean. Gated on
	 * shared->refine, matching the workers, so the internal barriers stay
	 * in lockstep. */
	if (shared->refine)
	{
		mkt_build_report_phase(prog, MKT_BUILD_PHASE_REFINE);
		MktDsmRefineAccum *accum = refine_accum;
		MktHeadWriteCtx	   rhead;
		mkt_head_write_ctx_init(
				&rhead,
				storage,
				rq_params,
				dim,
				shared->fastscan,
				first_posting);
		mkt_pbuild_exec_refine_paged(
				0,
				heap,
				index,
				index_info,
				shared,
				NULL,
				first_posting,
				accum,
				barrier,
				mkt_write_leaf_head,
				&rhead);
		mkt_head_write_ctx_cleanup(&rhead);
	}

	/* Phase 3: workers scan + route page-backed + encode + sort; the leader
	 * merges. Reported exactly once -- the seam fires the "mktann-build-load"
	 * test hook, and a 'wait' attached there must pause the build a single
	 * time -- and before the scan-reset barrier below releases the workers,
	 * so progress reflects the whole (multi-hour at scale) scan. */
	mkt_build_report_phase(prog, MKT_BUILD_PHASE_SCAN_PARALLEL);

	/* The samples (and the refine overlay riding in them) are dead; hand the
	 * segment back before the posting sort claims its own memory budget. */
	mkt_pbuild_samples_release(dsm_samples, lead.sample_seg);
	dsm_samples = NULL;

	/* Re-init the scan for the posting phase (the refine passes above consumed
	 * it). Guarded by the barrier below so no worker scans before the reset.
	 */
	mkt_pbuild_rescan(heap, shared);

	/* Barrier: scan reset for the posting phase; workers start the page-backed
	 * posting scan+sort. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	instr_time t_scan_start;
	INSTR_TIME_SET_CURRENT(t_scan_start);

	/* Barrier: wait until every worker has finished sorting its run, then
	 * merge and build. The leader does not scan; the workers cover the heap.
	 */
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

	/* Workers detached from the exact-centroid collection when routing ended;
	 * the leader's release is the last and frees it. */
	mkt_pbuild_exact_centroids_release(exact_seg);

	mkt_pbuild_teardown(pcxt);

	return true;
}
