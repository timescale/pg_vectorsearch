/*
 * posting_split.c - Incremental posting-list split (see header)
 */

#include "algo/kmeans.h"
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
			char *content = is_first ? mkt_posting_content_first(page, dim)
									 : mkt_posting_content(page);
			for (uint16_t i = 0; i < cnt; i++)
			{
				MktPostingEntryHeader *hdr =
						mkt_posting_entry_at(content, i, dim);
				if (hdr->meta.flags & MKT_POSTING_FLAG_DELETED)
					continue;

				if (out->count == out->cap)
					split_entries_grow(out, dim);

				ItemPointerData tid = hdr->meta.tid;
				if (env->fetch_vector(
							env->ctx,
							tid,
							out->vecs + (size_t)out->count * dim,
							dim))
				{
					out->tids[out->count] = tid;
					out->count++;
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
				false);
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

	/* 8. Tombstone the old chain (now unreachable) for later reclaim. */
	{
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

	base->nlist += 1;

	res.did_split = true;
	res.head0	  = h0;
	res.head1	  = h1;
	res.new_nlist = base->nlist;

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
