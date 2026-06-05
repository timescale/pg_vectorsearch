/*
 * parallel_build_leader.c - PG parallel index build, leader side
 *
 * The leader sets up the DSM, launches workers, participates in the
 * sampling + k-means phases as worker_id 0, then (phase 3) reserves the
 * posting layout, drains the workers' streamed pages over their shm_mq
 * queues, and finalizes each list (head + trailing-partial merge + chain
 * stitch). do_parallel_build returns the tree + posting heads to
 * mktann_build, which writes the centroid tree and metadata. Returns false
 * if parallelism could not start, so the caller falls back to a serial build.
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
#include "index/centroid_build.h"
#include "index/centroid_page.h"
#include "index/index_build.h"
#include "index/parallel_build.h"
#include "index/posting_build.h"
#include "index/posting_build_parallel.h"
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

bool
do_parallel_build(
		Relation			  heap,
		Relation			  index,
		struct IndexInfo	 *index_info,
		const MktBuildConfig *config,
		MktStorage			 *storage,
		HKMeansResult		**out_tree,
		BlockNumber			 *posting_heads,
		double				 *out_heap_tuples,
		double				 *out_indtuples,
		double				 *out_soar_dupes)
{
	int nworkers = index_info->ii_ParallelWorkers;

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
	void			 *dsm_tree		  = lead.dsm_tree;
	char			 *queues_base	  = lead.queues_base;
	char			 *dsm_partials	  = lead.dsm_partials;
	WalUsage		 *walusage		  = lead.walusage;
	BufferUsage		 *bufferusage	  = lead.bufferusage;
	int				  nparticipants	  = lead.nparticipants;
	uint32_t		  km_k			  = lead.km_k;
	Dimension		  dim			  = lead.dim;
	uint32_t		  nlist			  = lead.nlist;
	uint64_t		  rabitq_seed	  = lead.rabitq_seed;
	uint32_t		  fan_out		  = lead.fan_out;
	Size			  max_tree_sz	  = lead.max_tree_sz;

	instr_time t_launch_start;
	INSTR_TIME_SET_CURRENT(t_launch_start);

	/* ---- Launch workers + wait until they've all attached ---- */
	if (!mkt_pbuild_launch(pcxt, barrier))
		return false;
	/* ==== Leader participates in all phases as worker_id=0 ==== */

	/* ---- Phase 1: sampling (leader runs the worker body as participant 0)
	 * ---- */
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
	 * needs — for the root children it owns, and the leader grafts them.
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

	instr_time t_child_start;
	INSTR_TIME_SET_CURRENT(t_child_start);

	/* ---- Phase 2b: root assignment (leader as participant 0). ---- */
	mkt_pbuild_exec_root_assign(
			0, shared, dsm_samples, dsm_ra, centroids_base, barrier);

	if (nlevels >= 2)
	{
		/*
		 * Phase 2c: the leader builds the subtrees it owns (participant 0), to
		 * whatever depth nlist/fan_out requires — no two-level cap.
		 */
		char *subtrees_base = lead.child_subtrees_base;
		mkt_subtree_build_partitioned(
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
				shared->subtree_slot_size);

		/* Barrier: all participants done building subtrees. */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		/* Graft the fan_out subtrees under a fresh root (the root centroids).
		 */
		const HKMeansResult **subs = mkt_alloc(
				(size_t)km_k * sizeof(HKMeansResult *));
		for (uint32_t c = 0; c < km_k; c++)
			subs[c] = (const HKMeansResult *)mkt_dsm_child_subtree(
					subtrees_base, c, shared->subtree_slot_size);
		tree = mkt_hkmeans_graft(cents, km_k, subs, dim);
		mkt_free(subs);

		nlist = tree ? tree->nleaves : 0;
	}
	else
	{
		/*
		 * Flat (nlevels == 1, nlist <= fan_out): root k-means ran with
		 * km_k == nlist, so cents already holds every leaf centroid. The
		 * workers skip the subtree build; match their child-done barrier, then
		 * build the one-level tree straight from cents.
		 */
		BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
		tree = mkt_hkmeans_build_flat(cents, km_k, fan_out, dim);

		nlist = tree ? tree->nleaves : 0;
	}

	{
		instr_time t_child_elapsed;
		INSTR_TIME_SET_CURRENT(t_child_elapsed);
		INSTR_TIME_SUBTRACT(t_child_elapsed, t_child_start);
		mkt_debug(
				"mktann: child kmeans %.1fms (nlevels=%u, "
				"%u children, %u leaves)",
				INSTR_TIME_GET_MILLISEC(t_child_elapsed),
				nlevels,
				km_k,
				tree ? tree->nleaves : 0);
	}

	if (tree == NULL)
	{
		WaitForParallelWorkersToFinish(pcxt);
		mkt_pbuild_teardown(pcxt);
		return false;
	}

	/* Update nlist in shared state (workers read it for phase 3). */
	shared->nlist = nlist;

	/* Copy tree into pre-allocated DSM slot */
	if (tree->total_size > max_tree_sz)
		mkt_error(
				"mktann: tree too large for DSM (%u > %zu)",
				tree->total_size,
				max_tree_sz);
	memcpy(dsm_tree, tree, tree->total_size);

	/* Normalize leaf centroids for cosine */
	float *ref_vecs = hk_leaf_centroids(tree);
	if (shared->metric == DISTANCE_COSINE)
		for (uint32_t c = 0; c < nlist; c++)
			mkt_l2_normalize(ref_vecs + (size_t)c * dim, dim);

	/* Compute P^T * centroids */
	RaBitQParams *rq_params = mkt_rabitq_create(dim, rabitq_seed);
	float *pt_centroids		= mkt_alloc((size_t)nlist * dim * sizeof(float));
	for (uint32_t c = 0; c < nlist; c++)
		mkt_rabitq_rotate(
				rq_params,
				ref_vecs + (size_t)c * dim,
				pt_centroids + (size_t)c * dim);

	/* Block 0 = metadata page. Extend 1 page so block 0 exists. */
	mkt_storage_extend(storage, 1);

	/* Compute exact centroid page layout from the real tree,
	 * then extend for centroid + posting pages. Same layout
	 * logic as the serial path. */
	uint32_t cent_max_ent =
			mkt_centroid_max_entries_fmt(dim, shared->centroid_format);
	BlockNumber *node_first_blkno = mkt_alloc(
			tree->nnodes * sizeof(BlockNumber));
	BlockNumber first_centroid = 1;
	BlockNumber first_posting  = mkt_compute_centroid_layout(
			 tree, cent_max_ent, first_centroid, node_first_blkno);
	uint32_t n_centroid_pages = first_posting - first_centroid;
	mkt_storage_extend(storage, n_centroid_pages);

	instr_time t_km_end;
	INSTR_TIME_SET_CURRENT(t_km_end);
	INSTR_TIME_SUBTRACT(t_km_end, t_km_start);
	mkt_debug(
			"mktann: tree+setup %.1fms, %u clusters",
			INSTR_TIME_GET_MILLISEC(t_km_end),
			nlist);

	/* Re-init the scan for the posting phase (back-end seam). */
	mkt_pbuild_rescan(heap, shared);

	/* Barrier: tree ready, workers can start posting scan. This is the last
	 * barrier; phase 3 (drain) uses the shm_mq queues, not the barrier, so the
	 * leader detaches once released. */
	BarrierArriveAndWait(barrier, WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	BarrierDetach(barrier);

	instr_time t_scan_start;
	INSTR_TIME_SET_CURRENT(t_scan_start);

	/*
	 * ---- Phase 3: leader pre-reserves layout, then drains worker queues ----
	 *
	 * Bounded streaming build: the launched workers (worker_id 1..N) scan,
	 * assign + RaBitQ-encode, and stream completed full pages over their
	 * shm_mq; the leader does NOT scan here. It estimates each list's size,
	 * reserves a contiguous block range per list, pre-extends the relation,
	 * then drains every queue — placing each page in its list's range and
	 * linking it into the cluster's chain — until all queues detach. The
	 * finalize below writes each list's head (with centroid metadata) and
	 * splices the chain. For both AoS and fastscan, each worker holds its
	 * trailing partial page per cluster and the finalize folds those into the
	 * head, so the head is populated and there is no per-worker under-full
	 * page left in the chain.
	 */
	HKMeansResult *tree_r = (HKMeansResult *)dsm_tree;

	/* Per-cluster page estimate from the sample assignment, extrapolated
	 * to the full table (handles skew; slight over-estimate for headroom). */
	uint32_t *cluster_counts = mkt_alloc0((size_t)nlist * sizeof(uint32_t));
	uint32_t  n_est_samples	 = 0;
	for (int w = 0; w < nparticipants; w++)
	{
		float	*sw = mkt_dsm_worker_samples(dsm_samples, w);
		uint32_t nw = mkt_dsm_sample_counts(dsm_samples)[w];
		n_est_samples += nw;
		for (uint32_t i = 0; i < nw; i++)
		{
			Distance d;
			cluster_counts[mkt_hkmeans_assign(
					tree_r, sw + (size_t)i * dim, shared->metric, &d)]++;
		}
	}
	/* Extrapolate per-cluster sample counts to the full table; the reserve
	 * estimator applies the format + replication headroom. */
	bool replicate = shared->soar_lambda > 0.0 ||
					 shared->boundary_epsilon > 0.0;
	{
		double est_rows = RelationGetNumberOfBlocks(heap) *
						  (BLCKSZ / (double)(dim * sizeof(float) + 32));
		double scale = n_est_samples > 0 ? est_rows / n_est_samples : 1.0;
		for (uint32_t c = 0; c < nlist; c++)
			cluster_counts[c] = (uint32_t)((double)cluster_counts[c] * scale);
	}

	/* Leader-local reserve (0-based ranges; first_posting added on write).
	 * first_posting is the centroid-layout posting start computed above. */
	MktPostingReserve reserve;
	mkt_posting_reserve_init(
			&reserve,
			cluster_counts,
			nlist,
			nparticipants,
			dim,
			shared->fastscan,
			replicate);
	mkt_free(cluster_counts);

	/* Pre-extend the relation to cover the reserved ranges. mkt_storage_extend
	 * handles any backend-specific batching internally (the PG storage chunks
	 * around the ExtendBufferedRelBy pin limit). A cluster that outgrows its
	 * (over-)reservation overflows via on-demand new_page during the drain, so
	 * there is no separate spill region to pre-extend. */
	mkt_storage_extend(storage, reserve.total);

	/* Attach as receiver to each launched worker's queue. */
	int				nq = pcxt->nworkers_launched;
	shm_mq_handle **rh = mkt_alloc0(
			(size_t)nparticipants * sizeof(shm_mq_handle *));
	for (int wi = 0; wi < nq; wi++)
	{
		shm_mq *mq = (shm_mq *)mkt_dsm_posting_queue(queues_base, wi + 1);
		rh[wi + 1] = shm_mq_attach(mq, pcxt->seg, NULL);
	}

	/*
	 * Drain: place each continuation page at the next offset within its
	 * list's reserved range (offset 0 is the head, written in finalize); a
	 * list that outgrows its (over-)reservation extends the relation on demand
	 * via new_page. cl_used[c] counts placed continuations and drives the
	 * placement; cont_first/cont_last record the actual block numbers so the
	 * chain is linked from real placements rather than assuming the
	 * continuations are contiguous (they aren't, once a list overflows).
	 */
	uint32_t	*cl_used	= mkt_alloc0((size_t)nlist * sizeof(uint32_t));
	BlockNumber *cont_first = mkt_alloc((size_t)nlist * sizeof(BlockNumber));
	BlockNumber *cont_last	= mkt_alloc((size_t)nlist * sizeof(BlockNumber));
	for (uint32_t c = 0; c < nlist; c++)
	{
		cont_first[c] = InvalidBlockNumber;
		cont_last[c]  = InvalidBlockNumber;
	}
	bool *qdone = mkt_alloc0((size_t)(nq > 0 ? nq : 1) * sizeof(bool));
	int	  ndone = 0;
	while (ndone < nq)
	{
		bool progressed = false;
		for (int wi = 0; wi < nq; wi++)
		{
			Size		  len;
			void		 *data;
			shm_mq_result res;

			if (qdone[wi])
				continue;
			res = shm_mq_receive(rh[wi + 1], &len, &data, true);
			if (res == SHM_MQ_SUCCESS)
			{
				Page		src = (Page)data;
				uint32_t	c	= mkt_posting_opaque(src)->cluster_id;
				uint32_t	off = ++cl_used[c]; /* 1.. ; 0 = head */
				BlockNumber blk;
				Page		dst;
				if (off < reserve.counts[c])
				{
					/* Within the cluster's reserved (over-estimated) range. */
					blk = first_posting + reserve.starts[c] + off;
					dst = mkt_storage_write_page(storage, blk);
				}
				else
				{
					/* Cluster outgrew its reservation: extend on demand. The
					 * page lands at the end of the relation (non-sequential
					 * for this list, but rare) and is linked in by block
					 * number below. */
					dst = mkt_storage_new_page(storage, &blk);
				}
				memcpy(dst, src, BLCKSZ);
				mkt_storage_commit_page(storage, blk);

				/* Link into this cluster's continuation chain using actual
				 * block numbers. The previous page's next_blkno is fixed up
				 * once its successor's block is known; the final page's link
				 * is set in finalize. */
				if (cont_last[c] == InvalidBlockNumber)
				{
					cont_first[c] = blk;
				}
				else
				{
					Page prev = mkt_storage_write_page(storage, cont_last[c]);
					mkt_posting_opaque(prev)->next_blkno = blk;
					mkt_storage_commit_page(storage, cont_last[c]);
				}
				cont_last[c] = blk;
				progressed	 = true;
			}
			else if (res == SHM_MQ_DETACHED)
			{
				qdone[wi] = true;
				ndone++;
				progressed = true;
			}
		}
		if (!progressed)
		{
			WaitLatch(
					MyLatch,
					WL_LATCH_SET | WL_EXIT_ON_PM_DEATH,
					-1L,
					WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
			ResetLatch(MyLatch);
			CHECK_FOR_INTERRUPTS();
		}
	}
	for (int wi = 0; wi < nq; wi++)
		shm_mq_detach(rh[wi + 1]);
	mkt_free(rh);
	mkt_free(qdone);

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

	instr_time t_merge_start;
	INSTR_TIME_SET_CURRENT(t_merge_start);

	/*
	 * Finalize each list: write its head page (the list's reserved offset 0)
	 * with centroid metadata, fold every worker's trailing partial page into
	 * the head (re-packed optimally, overflowing past the continuations / into
	 * spill), then splice the chain head -> continuations -> overflow.
	 */
	uint32_t packed_bytes = (dim + 7) / 8;
	uint8_t *unpack_buf	  = shared->fastscan
								  ? mkt_alloc(MKT_FASTSCAN_GROUP * packed_bytes)
								  : NULL;
	for (uint32_t c = 0; c < nlist; c++)
	{
		BlockNumber head_blk = first_posting + reserve.starts[c];

		/* Overflow from the head builder is claimed after the
		 * continuations (offsets 1..cl_used are already written). */
		mkt_atomic_init_u32(&reserve.nexts[c], cl_used[c] + 1);

		MktPostingBuilder hb;
		if (shared->fastscan)
			mkt_posting_builder_init_fastscan(
					&hb,
					storage,
					rq_params,
					dim,
					c,
					ref_vecs + (size_t)c * dim,
					pt_centroids + (size_t)c * dim);
		else
			mkt_posting_builder_init(
					&hb,
					storage,
					rq_params,
					dim,
					c,
					ref_vecs + (size_t)c * dim,
					pt_centroids + (size_t)c * dim);
		mkt_posting_builder_set_shared_reserve(
				&hb,
				first_posting + reserve.starts[c],
				reserve.counts[c],
				&reserve.nexts[c]);
		mkt_posting_builder_set_first_blkno(&hb, head_blk);

		/*
		 * Fold every worker's trailing partial page for this cluster into the
		 * head builder, which re-packs it optimally. Worker pages are always
		 * continuations (only the leader makes heads), so content lives at the
		 * continuation offset. For fastscan we unpack each group's codes back
		 * to per-vector 1-bit form so add_encoded can re-pack them.
		 */
		for (int w = 0; w < nparticipants; w++)
		{
			Page pg = mkt_dsm_worker_partials(dsm_partials, nlist, w) +
					  (size_t)c * BLCKSZ;
			MktPostingPageOpaque *op = mkt_posting_opaque(pg);
			if (op->entry_count == 0)
				continue;
			char	*ct	 = mkt_posting_content(pg);
			uint32_t cnt = op->entry_count;

			if (shared->fastscan)
			{
				uint32_t ngroups = (cnt + MKT_FASTSCAN_GROUP - 1) /
								   MKT_FASTSCAN_GROUP;
				for (uint32_t g = 0; g < ngroups; g++)
				{
					uint32_t g_count = cnt - g * MKT_FASTSCAN_GROUP;
					if (g_count > MKT_FASTSCAN_GROUP)
						g_count = MKT_FASTSCAN_GROUP;
					mkt_fastscan_unpack_codes(
							mkt_fastscan_group_codes(ct, g, dim),
							g_count,
							dim,
							unpack_buf);
					ItemPointerData *tids =
							mkt_fastscan_group_tids(ct, g, dim);
					float *fa = mkt_fastscan_group_f_add(ct, g, dim);
					float *fr = mkt_fastscan_group_f_rescale(ct, g, dim);
					float *fe = mkt_fastscan_group_f_error(ct, g, dim);
					for (uint32_t v = 0; v < g_count; v++)
						mkt_posting_builder_add_encoded(
								&hb,
								tids[v],
								fa[v],
								fr[v],
								fe[v],
								unpack_buf + (size_t)v * packed_bytes);
				}
			}
			else
			{
				for (uint32_t e = 0; e < cnt; e++)
				{
					MktPostingEntryHeader *hdr =
							mkt_posting_entry_at(ct, e, dim);
					mkt_posting_builder_add_encoded(
							&hb,
							hdr->meta.tid,
							hdr->f_add,
							hdr->f_rescale,
							hdr->f_error,
							hdr->bits);
				}
			}
		}

		mkt_posting_builder_finish(&hb);
		mkt_posting_builder_cleanup(&hb);

		/* Splice the worker continuation chain (already linked internally
		 * during the drain, in real block order) between the head and the head
		 * builder's own overflow chain: head -> cont_first .. cont_last ->
		 * ov1, where ov1 is whatever the head builder linked to (its overflow,
		 * or InvalidBlockNumber when the head didn't overflow). */
		if (cont_first[c] != InvalidBlockNumber)
		{
			Page		hp	= mkt_storage_write_page(storage, head_blk);
			BlockNumber ov1 = mkt_posting_opaque(hp)->next_blkno;
			mkt_posting_opaque(hp)->next_blkno = cont_first[c];
			mkt_storage_commit_page(storage, head_blk);

			Page lp = mkt_storage_write_page(storage, cont_last[c]);
			mkt_posting_opaque(lp)->next_blkno = ov1;
			mkt_storage_commit_page(storage, cont_last[c]);
		}

		posting_heads[c] = head_blk;
	}

	uint32_t total_pages = RelationGetNumberOfBlocks(index) - first_posting;

	mkt_posting_reserve_free(&reserve);
	mkt_free(cl_used);
	mkt_free(cont_first);
	mkt_free(cont_last);
	if (unpack_buf != NULL)
		mkt_free(unpack_buf);
	mkt_free(pt_centroids);

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
