/*
 * posting_split.c - Incremental posting-list split (see header)
 */

#include "algo/kmeans.h"
#include "algo/vecops.h"
#include "core/memory.h"
#include "index/centroid_page.h"
#include "index/posting_insert.h"
#include "index/posting_page.h"
#include "index/posting_split.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Entry collection
 * ---------------------------------------------------------------- */

typedef struct SplitEntries
{
	float			*vecs; /* [count * dim] full-precision vectors */
	ItemPointerData *tids; /* [count] */
	uint32_t		 count;
	uint32_t		 cap;
	uint32_t		 cluster_id; /* the old list's cluster id */
} SplitEntries;

static void
split_entries_grow(SplitEntries *e, Dimension dim)
{
	uint32_t		 new_cap = e->cap ? e->cap * 2 : 64;
	float			*nv = mkt_alloc((size_t)new_cap * dim * sizeof(float));
	ItemPointerData *nt = mkt_alloc((size_t)new_cap * sizeof(ItemPointerData));
	if (e->count > 0)
	{
		memcpy(nv, e->vecs, (size_t)e->count * dim * sizeof(float));
		memcpy(nt, e->tids, (size_t)e->count * sizeof(ItemPointerData));
	}
	if (e->vecs != NULL)
		mkt_free(e->vecs);
	if (e->tids != NULL)
		mkt_free(e->tids);
	e->vecs = nv;
	e->tids = nt;
	e->cap	= new_cap;
}

/* Fetch one entry's vector and, on success, record its tid. */
static void
collect_one(
		SplitEntries	  *out,
		const MktSplitEnv *env,
		Dimension		   dim,
		ItemPointerData	   tid)
{
	if (out->count == out->cap)
		split_entries_grow(out, dim);

	if (env->fetch_vector(
				env->ctx, tid, out->vecs + (size_t)out->count * dim, dim))
	{
		out->tids[out->count] = tid;
		out->count++;
	}
}

/*
 * Walk the posting chain from `head`, fetching each live entry's
 * full-precision vector via env. Fills `out`. Returns 0 on success.
 */
static int
collect_entries(
		MktStorage		  *storage,
		Dimension		   dim,
		BlockNumber		   head,
		const MktSplitEnv *env,
		SplitEntries	  *out)
{
	memset(out, 0, sizeof(*out));

	BlockNumber blk	  = head;
	bool		first = true;
	while (blk != InvalidBlockNumber)
	{
		Page						page = mkt_storage_read_page(storage, blk);
		const MktPostingPageOpaque *op	 = mkt_posting_opaque(page);
		BlockNumber					next = op->next_blkno;
		uint16_t					cnt	 = op->entry_count;
		bool is_first  = (op->flags & MKT_POSTING_PAGE_FIRST) != 0;
		bool tombstone = (op->flags & MKT_POSTING_PAGE_TOMBSTONED) != 0;

		if (first)
		{
			out->cluster_id = op->cluster_id;
			first			= false;
		}

		if (!tombstone)
		{
			char *content	  = is_first ? mkt_posting_content_first(page, dim)
										 : mkt_posting_content(page);
			bool  is_fastscan = (op->flags & MKT_POSTING_PAGE_FASTSCAN) != 0;

			if (is_fastscan)
			{
				/* SoA: tids live in fixed 32-entry group sections. The last
				 * group may be partial; entry_count bounds the valid slots.
				 * Live pages have no per-entry delete flag (deletes tombstone
				 * the whole page, handled above). */
				uint32_t ngroups = ((uint32_t)cnt + MKT_FASTSCAN_GROUP - 1) /
								   MKT_FASTSCAN_GROUP;
				for (uint32_t g = 0; g < ngroups; g++)
				{
					ItemPointerData *tids =
							mkt_fastscan_group_tids(content, g, dim);
					uint32_t base_i = g * MKT_FASTSCAN_GROUP;
					uint32_t valid	= (cnt - base_i) < MKT_FASTSCAN_GROUP
											? (cnt - base_i)
											: MKT_FASTSCAN_GROUP;
					for (uint32_t v = 0; v < valid; v++)
						collect_one(out, env, dim, tids[v]);
				}
			}
			else
			{
				for (uint16_t i = 0; i < cnt; i++)
				{
					MktPostingEntryHeader *hdr =
							mkt_posting_entry_at(content, i, dim);
					if (hdr->meta.flags & MKT_POSTING_FLAG_DELETED)
						continue;
					collect_one(out, env, dim, hdr->meta.tid);
				}
			}
		}

		mkt_storage_release_page(storage, blk);
		blk = next;
	}
	return 0;
}

static void
split_entries_free(SplitEntries *e)
{
	if (e->vecs != NULL)
		mkt_free(e->vecs);
	if (e->tids != NULL)
		mkt_free(e->tids);
	e->vecs	 = NULL;
	e->tids	 = NULL;
	e->count = 0;
	e->cap	 = 0;
}

/* ----------------------------------------------------------------
 * Posting-head creation
 * ---------------------------------------------------------------- */

/*
 * Create a fresh posting-list head (first page) storing pt_centroid, with an
 * empty chain (tail = self, live_count = 0). Returns the new block number.
 */
static BlockNumber
new_posting_head(
		MktStorage	*storage,
		Dimension	 dim,
		uint32_t	 cluster_id,
		const float *pt_centroid)
{
	BlockNumber blkno;
	Page		page = mkt_storage_new_page(storage, &blkno);

	mkt_posting_page_init(page, cluster_id, dim, MKT_POSTING_PAGE_FIRST);
	memcpy(mkt_posting_pt_centroid_mut(page),
		   pt_centroid,
		   (size_t)dim * sizeof(float));

	MktPostingPageOpaque *op = mkt_posting_opaque(page);
	op->tail_blkno			 = blkno;
	op->live_count			 = 0;

	mkt_storage_commit_page(storage, blkno);
	return blkno;
}

/* ----------------------------------------------------------------
 * Centroid-tree flip
 * ---------------------------------------------------------------- */

/*
 * In the flat (nlevels == 1) tree, find the leaf entry whose child_blkno ==
 * head, and the last page of the level-0 chain (append target). Returns 0 on
 * success, -1 if the leaf entry was not found.
 */
static int
find_leaf_and_tail(
		MktStorage	*storage,
		BlockNumber	 first_centroid,
		BlockNumber	 head,
		BlockNumber *found_page,
		uint32_t	*found_idx,
		BlockNumber *tail_page,
		uint8_t		*level)
{
	*found_page = InvalidBlockNumber;
	*tail_page	= InvalidBlockNumber;

	BlockNumber blk = first_centroid;
	while (blk != InvalidBlockNumber)
	{
		Page page						 = mkt_storage_read_page(storage, blk);
		const MktCentroidPageOpaque *op	 = MKT_CENTROID_OPAQUE(page);
		uint16_t					 cnt = op->entry_count;
		BlockNumber					 next = op->next_blkno;
		*level							  = op->level;

		for (uint16_t i = 0; i < cnt; i++)
		{
			if (mkt_centroid_meta(page, i)->child_blkno == head)
			{
				*found_page = blk;
				*found_idx	= i;
			}
		}
		if (next == InvalidBlockNumber)
			*tail_page = blk;

		mkt_storage_release_page(storage, blk);
		blk = next;
	}

	return (*found_page == InvalidBlockNumber) ? -1 : 0;
}

/*
 * Retire a now-unreachable chain: the env defers reclaim behind an XID gate
 * (PG) or the core tombstones it immediately (standalone / NULL seam).
 */
static void
retire_or_tombstone_chain(
		MktIndexBase *base, const MktSplitEnv *env, BlockNumber head)
{
	if (env->retire_chain != NULL)
	{
		env->retire_chain(env->ctx, base->posting_storage, head);
		return;
	}
	BlockNumber blk = head;
	while (blk != InvalidBlockNumber)
	{
		Page page = mkt_storage_write_page(base->posting_storage, blk);
		MktPostingPageOpaque *op   = mkt_posting_opaque(page);
		BlockNumber			  next = op->next_blkno;
		op->flags |= MKT_POSTING_PAGE_TOMBSTONED;
		mkt_storage_commit_page(base->posting_storage, blk);
		blk = next;
	}
}

/* ----------------------------------------------------------------
 * LIRE boundary reassignment
 * ---------------------------------------------------------------- */

/* Read a posting head's stored P^T*centroid into out[dim]. */
static void
read_head_pt_centroid(
		MktStorage *st, BlockNumber head, Dimension dim, float *out)
{
	Page page = mkt_storage_read_page(st, head);
	memcpy(out, mkt_posting_pt_centroid(page), (size_t)dim * sizeof(float));
	mkt_storage_release_page(st, head);
}

/*
 * After a split, restore the nearest-partition invariant across the new
 * boundary: for the k_neighbors leaves nearest the new centroids, pull in any
 * entry now closer to c0/c1 than to its own centroid. Everything is done in
 * rotated (P^T) space using the heads' stored pt_centroids, which are exact.
 *
 * An affected neighbor is rewritten in place: a fresh head under the same
 * centroid receives the entries that stay, the movers go to h0/h1, the leaf is
 * repointed at the new head, and the old chain is retired. This reuses the
 * split's machinery and needs no per-entry deletion (so it is agnostic to AoS
 * vs FASTSCAN posting layout). Returns the number of entries moved.
 */
static uint32_t
reassign_neighbors(
		MktIndexBase	  *base,
		const MktSplitEnv *env,
		RaBitQParams	  *params,
		RaBitQScratch	  *scratch,
		BlockNumber		   h0,
		BlockNumber		   h1,
		const float		  *pt_c0,
		const float		  *pt_c1,
		uint32_t		   k_neighbors)
{
	Dimension dim = base->dim;
	if (k_neighbors == 0)
		return 0;

	/* 1. Enumerate leaf head blocks from the flat centroid chain (excluding
	 * the two new heads). */
	uint32_t	 cap	= base->nlist + 8;
	BlockNumber *heads	= mkt_alloc((size_t)cap * sizeof(BlockNumber));
	uint32_t	 nheads = 0;
	BlockNumber	 cblk	= base->first_centroid;
	while (cblk != InvalidBlockNumber)
	{
		Page page = mkt_storage_read_page(base->centroid_storage, cblk);
		const MktCentroidPageOpaque *op	  = MKT_CENTROID_OPAQUE(page);
		uint16_t					 cnt  = op->entry_count;
		BlockNumber					 next = op->next_blkno;
		for (uint16_t i = 0; i < cnt && nheads < cap; i++)
		{
			BlockNumber ch = mkt_centroid_meta(page, i)->child_blkno;
			if (ch != h0 && ch != h1 && ch != InvalidBlockNumber)
				heads[nheads++] = ch;
		}
		mkt_storage_release_page(base->centroid_storage, cblk);
		cblk = next;
	}

	/* 2. Keep the k_neighbors heads whose centroid is nearest either new
	 * centroid (simple partial selection over the small leaf set). */
	float *pt_cL = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	float *dists = mkt_alloc((size_t)(nheads ? nheads : 1) * sizeof(float));
	for (uint32_t i = 0; i < nheads; i++)
	{
		read_head_pt_centroid(base->posting_storage, heads[i], dim, pt_cL);
		float d0 = mkt_l2_distance_squared(pt_cL, pt_c0, dim);
		float d1 = mkt_l2_distance_squared(pt_cL, pt_c1, dim);
		dists[i] = d0 < d1 ? d0 : d1;
	}
	uint32_t nsel = k_neighbors < nheads ? k_neighbors : nheads;
	for (uint32_t s = 0; s < nsel; s++)
	{
		uint32_t best = s;
		for (uint32_t j = s + 1; j < nheads; j++)
			if (dists[j] < dists[best])
				best = j;
		float tmpd		 = dists[s];
		dists[s]		 = dists[best];
		dists[best]		 = tmpd;
		BlockNumber tmph = heads[s];
		heads[s]		 = heads[best];
		heads[best]		 = tmph;
	}

	/* 3. Rewrite each selected neighbor, moving entries closer to c0/c1. */
	float	*pt_e  = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	uint32_t moved = 0;
	for (uint32_t s = 0; s < nsel; s++)
	{
		BlockNumber L = heads[s];
		read_head_pt_centroid(base->posting_storage, L, dim, pt_cL);

		SplitEntries ne;
		collect_entries(base->posting_storage, dim, L, env, &ne);
		if (ne.count == 0)
		{
			split_entries_free(&ne);
			continue;
		}

		/* Decide per entry: 0 = stay in L, 1 = move to h0, 2 = move to h1. */
		uint8_t *tgt	   = mkt_alloc((size_t)ne.count);
		uint32_t local_mov = 0;
		for (uint32_t i = 0; i < ne.count; i++)
		{
			mkt_rabitq_rotate(params, ne.vecs + (size_t)i * dim, pt_e);
			float dL = mkt_l2_distance_squared(pt_e, pt_cL, dim);
			float d0 = mkt_l2_distance_squared(pt_e, pt_c0, dim);
			float d1 = mkt_l2_distance_squared(pt_e, pt_c1, dim);
			if (d0 < dL && d0 <= d1)
				tgt[i] = 1;
			else if (d1 < dL)
				tgt[i] = 2;
			else
				tgt[i] = 0;
			if (tgt[i] != 0)
				local_mov++;
		}

		if (local_mov == 0)
		{
			mkt_free(tgt);
			split_entries_free(&ne);
			continue;
		}

		/* Rewrite L in place under the same centroid; movers go to h0/h1. */
		BlockNumber Lnew = new_posting_head(
				base->posting_storage, dim, ne.cluster_id, pt_cL);
		for (uint32_t i = 0; i < ne.count; i++)
		{
			mkt_rabitq_rotate(params, ne.vecs + (size_t)i * dim, pt_e);
			BlockNumber dst = tgt[i] == 0 ? Lnew : (tgt[i] == 1 ? h0 : h1);
			mkt_posting_insert_one(
					base->posting_storage,
					params,
					dim,
					dst,
					ne.tids[i],
					pt_e,
					scratch,
					false,
					0);
		}

		BlockNumber found_page, tail_page;
		uint32_t	found_idx;
		uint8_t		level;
		if (find_leaf_and_tail(
					base->centroid_storage,
					base->first_centroid,
					L,
					&found_page,
					&found_idx,
					&tail_page,
					&level) == 0)
		{
			Page cp =
					mkt_storage_write_page(base->centroid_storage, found_page);
			mkt_centroid_page_set_child(cp, found_idx, Lnew);
			mkt_storage_commit_page(base->centroid_storage, found_page);
		}
		retire_or_tombstone_chain(base, env, L);

		moved += local_mov;
		mkt_free(tgt);
		split_entries_free(&ne);
	}

	mkt_free_aligned(pt_e);
	mkt_free(dists);
	mkt_free_aligned(pt_cL);
	mkt_free(heads);
	return moved;
}

/* ----------------------------------------------------------------
 * Split
 * ---------------------------------------------------------------- */

int
mkt_posting_split(
		MktIndexBase		 *base,
		BlockNumber			  head,
		const MktSplitConfig *cfg,
		const MktSplitEnv	 *env,
		MktSplitResult		 *out)
{
	MktSplitResult res = {0};

	if (base == NULL || env == NULL || env->fetch_vector == NULL ||
		head == InvalidBlockNumber)
		return -1;

	/* Phase-1 scope: flat tree + RaBitQ centroid format only. */
	if (base->nlevels != 1 || base->centroid_format != MKT_CENTROID_FMT_RABITQ)
		return -1;

	RaBitQParams *params = mkt_index_ensure_rabitq(base);
	if (params == NULL || base->pt_global_mean == NULL)
		return -1;

	Dimension dim				= base->dim;
	uint32_t  min_split_entries = (cfg && cfg->min_split_entries)
										? cfg->min_split_entries
										: 2;

	/* 1. Snapshot the live entries + their vectors. */
	SplitEntries ent;
	collect_entries(base->posting_storage, dim, head, env, &ent);
	if (ent.count < min_split_entries || ent.count < 2)
	{
		split_entries_free(&ent);
		if (out != NULL)
			*out = res;
		return 0; /* declined: too few entries */
	}

	/* 2. 2-means over the full-precision vectors. */
	KMeansOptions opts = MKT_KMEANS_OPTIONS_DEFAULT;
	if (cfg != NULL)
	{
		if (cfg->km_max_iter)
			opts.max_iterations = cfg->km_max_iter;
		if (cfg->km_seed)
			opts.seed = cfg->km_seed;
	}
	KMeansResult *km =
			mkt_kmeans_f32(ent.vecs, ent.count, dim, 2, base->metric, &opts);
	if (km == NULL)
	{
		split_entries_free(&ent);
		return -1;
	}
	if (km->cluster_sizes[0] == 0 || km->cluster_sizes[1] == 0)
	{
		/* Degenerate: all vectors collapsed to one centroid. Decline. */
		mkt_kmeans_result_destroy(km);
		split_entries_free(&ent);
		if (out != NULL)
			*out = res;
		return 0;
	}

	/* 3. Rotate the two new centroids into P^T space. */
	float *pt_c0 = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	float *pt_c1 = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	mkt_rabitq_rotate(params, km->centroids, pt_c0);
	mkt_rabitq_rotate(params, km->centroids + dim, pt_c1);

	/* 4. Create the two new heads. head0 reuses the old cluster id; head1
	 * gets a fresh id (the next leaf index). */
	BlockNumber h0 = new_posting_head(
			base->posting_storage, dim, ent.cluster_id, pt_c0);
	BlockNumber h1 =
			new_posting_head(base->posting_storage, dim, base->nlist, pt_c1);

	/* 5. Re-encode each entry's residual against its assigned new centroid and
	 * append it to the corresponding head. */
	RaBitQScratch scratch;
	mkt_rabitq_scratch_init(&scratch, dim);
	float *pt_v = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);

	for (uint32_t i = 0; i < ent.count; i++)
	{
		mkt_rabitq_rotate(params, ent.vecs + (size_t)i * dim, pt_v);
		BlockNumber h = (km->assignments[i] == 0) ? h0 : h1;
		mkt_posting_insert_one(
				base->posting_storage,
				params,
				dim,
				h,
				ent.tids[i],
				pt_v,
				&scratch,
				false,
				0 /* re-inserts during a split never re-trigger a split */);
	}
	res.count0 = km->cluster_sizes[0];
	res.count1 = km->cluster_sizes[1];

	/* 6. Encode the two routing centroids (relative to the global mean) in the
	 * centroid page's RaBitQ format. encode_from_pt over (pt_c - pt_mean) is
	 * identical to encode_into(c, mean) used at build (both quantize the same
	 * rotated residual). */
	RaBitQData *rd0	   = mkt_alloc(MKT_RABITQ_DATA_SIZE(dim));
	RaBitQData *rd1	   = mkt_alloc(MKT_RABITQ_DATA_SIZE(dim));
	float	   *pt_res = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	for (Dimension d = 0; d < dim; d++)
		pt_res[d] = pt_c0[d] - base->pt_global_mean[d];
	mkt_rabitq_encode_from_pt(params, pt_res, rd0, &scratch);
	for (Dimension d = 0; d < dim; d++)
		pt_res[d] = pt_c1[d] - base->pt_global_mean[d];
	mkt_rabitq_encode_from_pt(params, pt_res, rd1, &scratch);

	/* 7. Flip the centroid tree: locate the old leaf entry + chain tail. */
	BlockNumber found_page, tail_page;
	uint32_t	found_idx;
	uint8_t		level = 0;
	int			rc	  = find_leaf_and_tail(
			   base->centroid_storage,
			   base->first_centroid,
			   head,
			   &found_page,
			   &found_idx,
			   &tail_page,
			   &level);
	if (rc != 0)
	{
		/* Should not happen: the head must be reachable from the tree. */
		mkt_rabitq_scratch_cleanup(&scratch);
		mkt_free(rd0);
		mkt_free(rd1);
		mkt_free_aligned(pt_res);
		mkt_free_aligned(pt_v);
		mkt_free_aligned(pt_c0);
		mkt_free_aligned(pt_c1);
		mkt_kmeans_result_destroy(km);
		split_entries_free(&ent);
		return -1;
	}

	/* Overwrite the old leaf entry: repoint at h0, routing centroid c0. */
	{
		Page fp = mkt_storage_write_page(base->centroid_storage, found_page);
		mkt_centroid_page_overwrite_entry(fp, dim, found_idx, h0, rd0);
		mkt_storage_commit_page(base->centroid_storage, found_page);
	}

	/* Append the new leaf entry for h1 (routing centroid c1); chain a new
	 * level-0 page if the tail is full. */
	{
		Page tp = mkt_storage_write_page(base->centroid_storage, tail_page);
		bool appended = mkt_centroid_page_add_entry(
				tp, dim, h1, 0, MKT_CENTROID_FLAG_LEAF, rd1);
		mkt_storage_commit_page(base->centroid_storage, tail_page);

		if (!appended)
		{
			BlockNumber np;
			Page npg = mkt_storage_new_page(base->centroid_storage, &np);
			mkt_centroid_page_init_fmt(npg, level, base->centroid_format);
			mkt_centroid_page_add_entry(
					npg, dim, h1, 0, MKT_CENTROID_FLAG_LEAF, rd1);
			mkt_storage_commit_page(base->centroid_storage, np);

			Page link =
					mkt_storage_write_page(base->centroid_storage, tail_page);
			MKT_CENTROID_OPAQUE(link)->next_blkno = np;
			mkt_storage_commit_page(base->centroid_storage, tail_page);
		}
	}

	/*
	 * 8. Retire the old chain, now unreachable via the tree. A concurrent
	 * scanner may still hold a stale head pointer read before the flip, so the
	 * backend that can see snapshots (PG) defers reclaim behind an XID gate,
	 * keeping the chain readable meanwhile. Where there are no such scanners
	 * (env->retire_chain == NULL, e.g. standalone), tombstone it immediately.
	 */
	retire_or_tombstone_chain(base, env, head);

	base->nlist += 1;

	/*
	 * 9. LIRE boundary reassignment: pull entries in neighboring leaves that
	 * are now closer to a new centroid. Restores the nearest-partition
	 * invariant across the new boundary. Runs after the flip so the neighbors'
	 * heads and centroids are stable. Reuses pt_c0/pt_c1 and the scratch.
	 */
	uint32_t reassigned = 0;
	if (cfg != NULL && cfg->reassign_neighbors > 0)
		reassigned = reassign_neighbors(
				base,
				env,
				params,
				&scratch,
				h0,
				h1,
				pt_c0,
				pt_c1,
				cfg->reassign_neighbors);

	res.did_split  = true;
	res.head0	   = h0;
	res.head1	   = h1;
	res.new_nlist  = base->nlist;
	res.reassigned = reassigned;

	mkt_rabitq_scratch_cleanup(&scratch);
	mkt_free(rd0);
	mkt_free(rd1);
	mkt_free_aligned(pt_res);
	mkt_free_aligned(pt_v);
	mkt_free_aligned(pt_c0);
	mkt_free_aligned(pt_c1);
	mkt_kmeans_result_destroy(km);
	split_entries_free(&ent);

	if (out != NULL)
		*out = res;
	return 0;
}
