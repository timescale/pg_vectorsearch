/*
 * posting_split.c - Incremental posting-list split (see header)
 */

#include <math.h>

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

/* True if a posting head is still routable (not retired/tombstoned). */
static bool
posting_head_live(MktStorage *st, BlockNumber head)
{
	Page	 page  = mkt_storage_read_page(st, head);
	uint16_t flags = mkt_posting_opaque(page)->flags;
	mkt_storage_release_page(st, head);
	return (flags &
			(MKT_POSTING_PAGE_DELETED | MKT_POSTING_PAGE_TOMBSTONED)) == 0;
}

/*
 * Routing table over the flat tree's live leaves: parallel arrays of head
 * block and P^T-space centroid. Used to route a reassignment candidate to its
 * true nearest posting -- our IVF-tree equivalent of LIRE's SPTAG search.
 * Retired heads (the split's old chain, merge-poisoned leaves) are excluded so
 * nothing routes into a dead list.
 */
typedef struct LeafTable
{
	BlockNumber *head; /* [n] */
	float		*pt;   /* [n * dim], P^T space */
	uint32_t	 n;
} LeafTable;

static void
leaf_table_build(MktIndexBase *base, Dimension dim, LeafTable *lt)
{
	uint32_t cap = base->nlist + 8;
	lt->head	 = mkt_alloc((size_t)cap * sizeof(BlockNumber));
	lt->pt		 = mkt_alloc_aligned((size_t)cap * dim * sizeof(float), 64);
	lt->n		 = 0;

	BlockNumber cblk = base->first_centroid;
	while (cblk != InvalidBlockNumber)
	{
		Page page = mkt_storage_read_page(base->centroid_storage, cblk);
		const MktCentroidPageOpaque *op	  = MKT_CENTROID_OPAQUE(page);
		uint16_t					 cnt  = op->entry_count;
		BlockNumber					 next = op->next_blkno;
		for (uint16_t i = 0; i < cnt && lt->n < cap; i++)
		{
			BlockNumber ch = mkt_centroid_meta(page, i)->child_blkno;
			if (ch == InvalidBlockNumber)
				continue;
			if (!posting_head_live(base->posting_storage, ch))
				continue;
			lt->head[lt->n] = ch;
			read_head_pt_centroid(
					base->posting_storage,
					ch,
					dim,
					lt->pt + (size_t)lt->n * dim);
			lt->n++;
		}
		mkt_storage_release_page(base->centroid_storage, cblk);
		cblk = next;
	}
}

static void
leaf_table_free(LeafTable *lt)
{
	mkt_free(lt->head);
	mkt_free_aligned(lt->pt);
	lt->head = NULL;
	lt->pt	 = NULL;
	lt->n	 = 0;
}

/* Nearest live leaf head to pt_v (exact scan; off the hot path). */
static BlockNumber
route_nearest_leaf(const LeafTable *lt, Dimension dim, const float *pt_v)
{
	BlockNumber best  = InvalidBlockNumber;
	float		bestd = INFINITY;
	for (uint32_t i = 0; i < lt->n; i++)
	{
		float d = mkt_l2_distance_squared(pt_v, lt->pt + (size_t)i * dim, dim);
		if (d < bestd)
		{
			bestd = d;
			best  = lt->head[i];
		}
	}
	return best;
}

/* One posting examined for reassignment, with a per-entry disposition. */
typedef struct AffectedList
{
	BlockNumber	 head;	  /* the posting being examined */
	SplitEntries ent;	  /* its live entries + full-precision vectors */
	BlockNumber *target;  /* [ent.count]: move target, or Invalid = stay */
	uint32_t	 movers;  /* entries with a target */
	BlockNumber	 newhead; /* fresh head if rewritten, else Invalid */
} AffectedList;

/*
 * LIRE reassignment after a split (paper section 3.3). Restores nearest-
 * partition assignment (NPA) across the new boundary by re-examining the two
 * split halves and the k nearest neighbor leaves of the deleted centroid A_o:
 *
 *   push-out (cond 1): a vector in h0/h1 with D(v,A_o) <= D(v,c_i) for BOTH
 * new centroids may belong in a pre-existing neighbor. pull-in  (cond 2): a
 * vector in a neighbor with D(v,c_i) <= D(v,A_o) for SOME new centroid may
 * belong in h0/h1 (or elsewhere).
 *
 * The two conditions are a cheap necessary-condition filter; each surviving
 * candidate is routed to its true nearest live leaf and moved only if that
 * differs from where it sits (the NPA re-check drops false positives). Moves
 * are applied by rewriting each source posting in place (fresh head + repoint
 * + retire), so no per-entry delete is needed (FASTSCAN-safe). A candidate can
 * land in ANY posting, not just h0/h1. Everything is in rotated (P^T) space
 * using the heads' stored pt_centroids, which are exact. Returns the number of
 * entries moved.
 */
static uint32_t
reassign_lire(
		MktIndexBase	  *base,
		const MktSplitEnv *env,
		RaBitQParams	  *params,
		RaBitQScratch	  *scratch,
		const float		  *pt_ao,
		BlockNumber		   h0,
		BlockNumber		   h1,
		const float		  *pt_c0,
		const float		  *pt_c1,
		uint32_t		   k_neighbors)
{
	Dimension dim = base->dim;
	if (k_neighbors == 0)
		return 0;

	/* 1. Snapshot the live leaves as a read-only routing table. */
	LeafTable lt;
	leaf_table_build(base, dim, &lt);

	/* 2. Rank neighbor leaves (excluding h0/h1) by distance to A_o; keep k. */
	uint32_t	 ncand = 0;
	BlockNumber *cand  = mkt_alloc(
			 (size_t)(lt.n ? lt.n : 1) * sizeof(BlockNumber));
	float *cd = mkt_alloc((size_t)(lt.n ? lt.n : 1) * sizeof(float));
	for (uint32_t i = 0; i < lt.n; i++)
	{
		if (lt.head[i] == h0 || lt.head[i] == h1)
			continue;
		cand[ncand] = lt.head[i];
		cd[ncand] =
				mkt_l2_distance_squared(lt.pt + (size_t)i * dim, pt_ao, dim);
		ncand++;
	}
	uint32_t nsel = k_neighbors < ncand ? k_neighbors : ncand;
	for (uint32_t s = 0; s < nsel; s++)
	{
		uint32_t best = s;
		for (uint32_t j = s + 1; j < ncand; j++)
			if (cd[j] < cd[best])
				best = j;
		float tmpd		 = cd[s];
		cd[s]			 = cd[best];
		cd[best]		 = tmpd;
		BlockNumber tmph = cand[s];
		cand[s]			 = cand[best];
		cand[best]		 = tmph;
	}

	/* 3. Affected set: the two split halves + the selected neighbors. */
	uint32_t	  naff = 2 + nsel;
	AffectedList *aff  = mkt_alloc((size_t)naff * sizeof(AffectedList));
	for (uint32_t a = 0; a < naff; a++)
	{
		aff[a].head	   = (a == 0) ? h0 : (a == 1) ? h1 : cand[a - 2];
		aff[a].target  = NULL;
		aff[a].movers  = 0;
		aff[a].newhead = InvalidBlockNumber;
	}

	float *pt_v = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);

	/* 4. Plan: classify every entry via the filter, then route candidates. */
	for (uint32_t a = 0; a < naff; a++)
	{
		bool is_split_half = (a < 2);
		collect_entries(
				base->posting_storage, dim, aff[a].head, env, &aff[a].ent);
		uint32_t cnt = aff[a].ent.count;
		if (cnt == 0)
			continue;
		aff[a].target = mkt_alloc((size_t)cnt * sizeof(BlockNumber));
		for (uint32_t i = 0; i < cnt; i++)
		{
			aff[a].target[i] = InvalidBlockNumber; /* stay */
			mkt_rabitq_rotate(params, aff[a].ent.vecs + (size_t)i * dim, pt_v);
			float dao = mkt_l2_distance_squared(pt_v, pt_ao, dim);
			float d0  = mkt_l2_distance_squared(pt_v, pt_c0, dim);
			float d1  = mkt_l2_distance_squared(pt_v, pt_c1, dim);

			bool candidate = is_split_half ? (dao <= d0 && dao <= d1)  /* c1 */
										   : (d0 <= dao || d1 <= dao); /* c2 */
			if (!candidate)
				continue;

			BlockNumber tgt = route_nearest_leaf(&lt, dim, pt_v);
			if (tgt != InvalidBlockNumber && tgt != aff[a].head)
			{
				aff[a].target[i] = tgt;
				aff[a].movers++;
			}
		}
	}

	/* 5. Allocate a fresh head for each posting that loses entries. */
	for (uint32_t a = 0; a < naff; a++)
	{
		if (aff[a].movers == 0)
			continue;
		float *pt_c = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
		read_head_pt_centroid(base->posting_storage, aff[a].head, dim, pt_c);
		aff[a].newhead = new_posting_head(
				base->posting_storage, dim, aff[a].ent.cluster_id, pt_c);
		mkt_free_aligned(pt_c);
	}

	/* 6. Re-insert the stayers of each rewritten posting into its new head. */
	for (uint32_t a = 0; a < naff; a++)
	{
		if (aff[a].movers == 0)
			continue;
		for (uint32_t i = 0; i < aff[a].ent.count; i++)
		{
			if (aff[a].target[i] != InvalidBlockNumber)
				continue; /* mover: handled in step 7 */
			mkt_rabitq_rotate(params, aff[a].ent.vecs + (size_t)i * dim, pt_v);
			mkt_posting_insert_one(
					base->posting_storage,
					params,
					dim,
					aff[a].newhead,
					aff[a].ent.tids[i],
					pt_v,
					scratch,
					false,
					0);
		}
	}

	/*
	 * 7. Insert every mover into its routed target -- redirected to the
	 * target's new head when that target is itself being rewritten.
	 */
	uint32_t moved = 0;
	for (uint32_t a = 0; a < naff; a++)
	{
		if (aff[a].movers == 0)
			continue;
		for (uint32_t i = 0; i < aff[a].ent.count; i++)
		{
			BlockNumber tgt = aff[a].target[i];
			if (tgt == InvalidBlockNumber)
				continue;
			for (uint32_t b = 0; b < naff; b++)
				if (aff[b].head == tgt && aff[b].newhead != InvalidBlockNumber)
				{
					tgt = aff[b].newhead;
					break;
				}
			mkt_rabitq_rotate(params, aff[a].ent.vecs + (size_t)i * dim, pt_v);
			mkt_posting_insert_one(
					base->posting_storage,
					params,
					dim,
					tgt,
					aff[a].ent.tids[i],
					pt_v,
					scratch,
					false,
					0);
			moved++;
		}
	}

	/* 8. Repoint each rewritten leaf slot and retire the old chain. */
	for (uint32_t a = 0; a < naff; a++)
	{
		if (aff[a].movers == 0)
			continue;
		BlockNumber found_page, tail_page;
		uint32_t	found_idx;
		uint8_t		level;
		if (find_leaf_and_tail(
					base->centroid_storage,
					base->first_centroid,
					aff[a].head,
					&found_page,
					&found_idx,
					&tail_page,
					&level) == 0)
		{
			Page cp =
					mkt_storage_write_page(base->centroid_storage, found_page);
			mkt_centroid_page_set_child(cp, found_idx, aff[a].newhead);
			mkt_storage_commit_page(base->centroid_storage, found_page);
		}
		retire_or_tombstone_chain(base, env, aff[a].head);
	}

	for (uint32_t a = 0; a < naff; a++)
	{
		if (aff[a].target != NULL)
			mkt_free(aff[a].target);
		split_entries_free(&aff[a].ent);
	}
	mkt_free_aligned(pt_v);
	mkt_free(aff);
	mkt_free(cand);
	mkt_free(cd);
	leaf_table_free(&lt);
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

	/* 3. Rotate the two new centroids into P^T space. Also capture the old
	 * (deleted) centroid A_o now, before the tree flip repoints its leaf slot
	 * and step 8 retires its head -- LIRE reassignment needs A_o's value as
	 * the necessary-condition threshold. */
	float *pt_c0 = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	float *pt_c1 = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	float *pt_ao = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	mkt_rabitq_rotate(params, km->centroids, pt_c0);
	mkt_rabitq_rotate(params, km->centroids + dim, pt_c1);
	read_head_pt_centroid(base->posting_storage, head, dim, pt_ao);

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
		mkt_free_aligned(pt_ao);
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
	 * 9. LIRE boundary reassignment (paper section 3.3): restore NPA across
	 * the new boundary by re-examining the two split halves (push-out) and the
	 * k nearest neighbor leaves of A_o (pull-in), routing each candidate to
	 * its true nearest live leaf. Runs after the flip and the retire above so
	 * the routing table sees h0/h1 and excludes the dead old chain.
	 */
	uint32_t reassigned = 0;
	if (cfg != NULL && cfg->reassign_neighbors > 0)
		reassigned = reassign_lire(
				base,
				env,
				params,
				&scratch,
				pt_ao,
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
	mkt_free_aligned(pt_ao);
	mkt_kmeans_result_destroy(km);
	split_entries_free(&ent);

	if (out != NULL)
		*out = res;
	return 0;
}

/* ----------------------------------------------------------------
 * Merge
 * ---------------------------------------------------------------- */

BlockNumber
mkt_posting_merge_find_target(MktIndexBase *base, BlockNumber head)
{
	if (base == NULL || head == InvalidBlockNumber || base->nlevels != 1)
		return InvalidBlockNumber;
	Dimension dim = base->dim;

	/* Enumerate the other leaf heads (single-buffer-safe: collect block
	 * numbers first, then read posting heads). */
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
			if (ch != head && ch != InvalidBlockNumber)
				heads[nheads++] = ch;
		}
		mkt_storage_release_page(base->centroid_storage, cblk);
		cblk = next;
	}

	/* Nearest other leaf to the source centroid (rotated space). Skip heads
	 * that are already retired (a prior merge in the same pass may have
	 * poisoned/retired a leaf that is still present in the tree). */
	float *pt_src = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	float *pt_cL  = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	read_head_pt_centroid(base->posting_storage, head, dim, pt_src);
	BlockNumber target = InvalidBlockNumber;
	float		best   = 0.0f;
	for (uint32_t i = 0; i < nheads; i++)
	{
		Page hp = mkt_storage_read_page(base->posting_storage, heads[i]);
		const MktPostingPageOpaque *hop = mkt_posting_opaque(hp);
		bool live = (hop->flags & (MKT_POSTING_PAGE_TOMBSTONED |
								   MKT_POSTING_PAGE_DELETED)) == 0;
		if (live)
			memcpy(pt_cL,
				   mkt_posting_pt_centroid(hp),
				   (size_t)dim * sizeof(float));
		mkt_storage_release_page(base->posting_storage, heads[i]);
		if (!live)
			continue;
		float d = mkt_l2_distance_squared(pt_src, pt_cL, dim);
		if (target == InvalidBlockNumber || d < best)
		{
			best   = d;
			target = heads[i];
		}
	}

	mkt_free_aligned(pt_cL);
	mkt_free_aligned(pt_src);
	mkt_free(heads);
	return target;
}

int
mkt_posting_merge_into(
		MktIndexBase	  *base,
		BlockNumber		   head,
		BlockNumber		   target,
		const MktSplitEnv *env,
		MktMergeResult	  *out)
{
	MktMergeResult res = {0};

	if (base == NULL || env == NULL || env->fetch_vector == NULL ||
		head == InvalidBlockNumber || target == InvalidBlockNumber)
		return -1;
	if (base->nlevels != 1 || base->centroid_format != MKT_CENTROID_FMT_RABITQ)
		return -1;

	RaBitQParams *params = mkt_index_ensure_rabitq(base);
	if (params == NULL || base->pt_global_mean == NULL)
		return -1;
	Dimension dim = base->dim;

	/* Move the source entries into the target, re-encoded against its
	 * centroid. */
	SplitEntries ent;
	collect_entries(base->posting_storage, dim, head, env, &ent);

	RaBitQScratch scratch;
	mkt_rabitq_scratch_init(&scratch, dim);
	float *pt_e = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	for (uint32_t i = 0; i < ent.count; i++)
	{
		mkt_rabitq_rotate(params, ent.vecs + (size_t)i * dim, pt_e);
		mkt_posting_insert_one(
				base->posting_storage,
				params,
				dim,
				target,
				ent.tids[i],
				pt_e,
				&scratch,
				false,
				0);
	}

	/* Poison the source leaf so routing skips it, then retire its chain. */
	BlockNumber found_page, tail_page;
	uint32_t	found_idx;
	uint8_t		level;
	if (find_leaf_and_tail(
				base->centroid_storage,
				base->first_centroid,
				head,
				&found_page,
				&found_idx,
				&tail_page,
				&level) == 0)
	{
		Page cp = mkt_storage_write_page(base->centroid_storage, found_page);
		mkt_centroid_page_poison_entry(cp, dim, found_idx);
		mkt_storage_commit_page(base->centroid_storage, found_page);
	}
	retire_or_tombstone_chain(base, env, head);

	if (base->nlist > 0)
		base->nlist -= 1;

	res.did_merge = true;
	res.target	  = target;
	res.moved	  = ent.count;
	res.new_nlist = base->nlist;

	mkt_rabitq_scratch_cleanup(&scratch);
	mkt_free_aligned(pt_e);
	split_entries_free(&ent);

	if (out != NULL)
		*out = res;
	return 0;
}

int
mkt_posting_merge(
		MktIndexBase	  *base,
		BlockNumber		   head,
		const MktSplitEnv *env,
		MktMergeResult	  *out)
{
	BlockNumber target = mkt_posting_merge_find_target(base, head);
	if (target == InvalidBlockNumber)
	{
		if (out != NULL)
		{
			MktMergeResult res = {0};
			*out			   = res;
		}
		return 0; /* only leaf left — nothing to merge into */
	}
	return mkt_posting_merge_into(base, head, target, env, out);
}

/* ----------------------------------------------------------------
 * Centroid compaction
 * ---------------------------------------------------------------- */

/*
 * Reclaim poisoned leaf slots (f_add == +inf, left by merge) from the flat
 * level-0 centroid chain. Rewrites the chain in place, keeping only live
 * entries and preserving the page links, so routing no longer scores the dead
 * slots. Returns the number of slots reclaimed, or a negative value on error.
 *
 * Not concurrency-safe against scans on its own (an entry can transiently
 * appear on two pages as it migrates); run it where no scan races it, or wire
 * the PG path via a fresh-chain-and-flip (a later commit). RaBitQ centroid,
 * flat tree only.
 */
int
mkt_centroid_compact(MktIndexBase *base)
{
	if (base == NULL || base->nlevels != 1 ||
		base->centroid_format != MKT_CENTROID_FMT_RABITQ)
		return -1;

	Dimension	dim		  = base->dim;
	uint32_t	data_size = mkt_centroid_data_size(dim, base->centroid_format);
	MktStorage *cs		  = base->centroid_storage;

	/* Pass 1: count chain pages and live (non-poisoned) leaf entries. */
	uint32_t	nblk = 0, nlive = 0;
	BlockNumber cblk = base->first_centroid;
	while (cblk != InvalidBlockNumber)
	{
		Page						 page = mkt_storage_read_page(cs, cblk);
		const MktCentroidPageOpaque *op	  = MKT_CENTROID_OPAQUE(page);
		uint16_t					 cnt  = op->entry_count;
		BlockNumber					 next = op->next_blkno;
		nblk++;
		for (uint16_t i = 0; i < cnt; i++)
			if (!isinf(mkt_centroid_data(page, i, dim)->f_add))
				nlive++;
		mkt_storage_release_page(cs, cblk);
		cblk = next;
	}
	if (nblk == 0)
		return 0;

	/* Pass 2: snapshot the chain layout and the surviving entries. */
	BlockNumber *blocks = mkt_alloc((size_t)nblk * sizeof(BlockNumber));
	BlockNumber *nexts	= mkt_alloc((size_t)nblk * sizeof(BlockNumber));
	uint8_t		*levels = mkt_alloc((size_t)nblk);
	BlockNumber *lchild = nlive ? mkt_alloc(
										  (size_t)nlive * sizeof(BlockNumber))
								: NULL;
	uint16_t *lcc = nlive ? mkt_alloc((size_t)nlive * sizeof(uint16_t)) : NULL;
	uint16_t *lflags = nlive ? mkt_alloc((size_t)nlive * sizeof(uint16_t))
							 : NULL;
	uint8_t	 *ldata	 = nlive ? mkt_alloc((size_t)nlive * data_size) : NULL;

	uint32_t bi = 0, li = 0, removed = 0;
	cblk = base->first_centroid;
	while (cblk != InvalidBlockNumber)
	{
		Page						 page = mkt_storage_read_page(cs, cblk);
		const MktCentroidPageOpaque *op	  = MKT_CENTROID_OPAQUE(page);
		uint16_t					 cnt  = op->entry_count;
		blocks[bi]						  = cblk;
		nexts[bi]						  = op->next_blkno;
		levels[bi]						  = op->level;
		bi++;
		for (uint16_t i = 0; i < cnt; i++)
		{
			const RaBitQData *d = mkt_centroid_data(page, i, dim);
			if (isinf(d->f_add))
			{
				removed++;
				continue;
			}
			const MktCentroidEntryMeta *m = mkt_centroid_meta(page, i);
			lchild[li]					  = m->child_blkno;
			lcc[li]						  = m->child_count;
			lflags[li]					  = m->flags;
			memcpy(ldata + (size_t)li * data_size, d, data_size);
			li++;
		}
		BlockNumber next = op->next_blkno;
		mkt_storage_release_page(cs, cblk);
		cblk = next;
	}

	/* Pass 3: reinit each chain page (restoring its link + level) and refill
	 * with the surviving entries. Since we only removed entries, they always
	 * fit back into the same pages; any trailing page is left empty but
	 * linked.
	 */
	if (removed > 0)
	{
		uint32_t out = 0;
		for (uint32_t b = 0; b < nblk; b++)
		{
			Page page = mkt_storage_write_page(cs, blocks[b]);
			mkt_centroid_page_init_fmt(page, levels[b], base->centroid_format);
			MKT_CENTROID_OPAQUE(page)->next_blkno = nexts[b];
			while (out < li && mkt_centroid_page_has_room(page, dim, false))
			{
				mkt_centroid_page_add_entry(
						page,
						dim,
						lchild[out],
						lcc[out],
						lflags[out],
						ldata + (size_t)out * data_size);
				out++;
			}
			mkt_storage_commit_page(cs, blocks[b]);
		}
	}

	mkt_free(blocks);
	mkt_free(nexts);
	mkt_free(levels);
	if (nlive)
	{
		mkt_free(lchild);
		mkt_free(lcc);
		mkt_free(lflags);
		mkt_free(ldata);
	}
	return (int)removed;
}
