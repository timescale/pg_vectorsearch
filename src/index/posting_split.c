/*
 * posting_split.c - Incremental posting-list split (see header)
 */

#include <string.h>

#include "algo/distance.h"
#include "algo/kmeans.h"
#include "algo/vecops.h"
#include "core/injection.h"
#include "core/memory.h"
#include "index/centroid_page.h"
#include "index/posting_build.h"
#include "index/posting_page.h"
#include "index/posting_split.h"
#include "quant/rabitq.h"

/* ----------------------------------------------------------------
 * Entry collection
 * ---------------------------------------------------------------- */

/*
 * A split streams its list; it does not hold it.
 *
 * Holding every vector at full precision costs n * dim * 4 bytes, which makes
 * the tool for splitting an oversized list fail in proportion to how oversized
 * the list is -- the one case where it has to work. So the list is streamed
 * twice instead: once to count it and draw a bounded sample to cluster on, and
 * once to write each entry into the list whose centroid is nearest. Peak
 * memory is the sample plus the page builders, whatever the list holds.
 *
 * Clustering a sample rather than the whole list is the trade the bulk build
 * already makes for the same reason. A list that fits the budget is sampled
 * whole, so nothing changes for the sizes a split normally meets.
 */
typedef struct SplitSample
{
	float	*vecs;	/* [alloc * dim], the clustering input */
	uint32_t cap;	/* ceiling the budget pays for */
	uint32_t alloc; /* entries the buffer holds; grows toward cap */
	uint32_t count; /* sampled so far */

	uint32_t n_live;	   /* fetchable entries in the list */
	uint32_t n_degenerate; /* of those, with no defined distance */
	uint32_t eligible;	   /* entries the sample could have taken */
	uint64_t rng;		   /* reservoir draws; seeded, so runs repeat */

	Dimension		   dim;
	DistanceMetric	   metric;
	const MktSplitEnv *env;
	float			  *one; /* scratch for the entry being fetched */
} SplitSample;

/* The j'th centroid of a packed [n * dim] array. */
static inline const float *
centroid_at(const float *centroids, uint32_t j, Dimension dim)
{
	return centroids + (size_t)j * dim;
}

/*
 * Deterministic draws for the reservoir. splitmix64: a few operations, and
 * seeded from the split's own seed so a given list samples the same way twice.
 * Choosing which sampled point to displace does not need more than this.
 */
static inline uint64_t
splitmix64_next(uint64_t *state)
{
	uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
	z		   = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
	z		   = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
	return z ^ (z >> 31);
}

/* The head's own idea of how many entries the list holds. Only an estimate for
 * sizing the sample: it does not know about entries whose vector can no longer
 * be fetched. */
static uint32_t
head_live_count(MktStorage *storage, BlockNumber head)
{
	Page	 p = mkt_storage_read_page(storage, head);
	uint32_t n = mkt_posting_head_live_count(p);
	mkt_storage_release_page(storage, head);
	return n;
}

/* True when `vec` has no defined distance under `metric` -- a zero-norm vector
 * under cosine. Such an entry can never be a result, and must stay out of the
 * clustering: its distance to every centroid is undefined, and a cluster of
 * nothing else gets a zero-norm mean that k-means leaves unnormalized. */
static bool
vector_is_degenerate(const float *vec, Dimension dim, DistanceMetric metric)
{
	/* Squared norm -- zero iff the norm is zero, without the sqrt. */
	return metric == DISTANCE_COSINE && mkt_l2_norm_squared(vec, dim) == 0.0f;
}

/* ----------------------------------------------------------------
 * Chain traversal
 * ---------------------------------------------------------------- */

/*
 * Call `cb` once per live entry of the chain at `head`, in chain order --
 * once per indexed vector, that is, not once per page or per fastscan group.
 * Callers see a flat sequence of tids and nothing about how the pages are
 * laid out: this is the single place that knows either page format, so every
 * pass over a list shares it.
 *
 * It yields tids rather than vectors deliberately. Fetching is the caller's
 * job (see sample_one_vector and write_one_vector), which is what keeps this
 * cheap and independent of where the vectors live.
 *
 * `cluster_id_out` receives the list's cluster id from the head page.
 */
/*
 * Called once per live entry. Runs in a context the walk resets after every
 * entry, so anything the callback allocates -- including whatever the
 * backend's fetch allocates underneath it, which for a toasted vector is a
 * whole TOAST reassembly -- goes away with that entry. The contract that
 * buys: a callback must not allocate anything it needs after returning.
 * Buffers that outlive an entry belong to the caller and are allocated
 * before the walk.
 */
typedef void (*ChainTidCb)(void *state, ItemPointerData tid);

/* State a tid walk carries across its pages. */
typedef struct ChainTidsCtx
{
	Dimension		 dim;
	ChainTidCb		 cb;
	void			*state;
	ItemPointerData *tids; /* one page's worth, caller-owned */
	uint32_t		 cap;
	MktMemCtx		 entry_ctx;
	void (*prefetch)(void *, ItemPointerData);
	void	 *prefetch_ctx;
	uint32_t *cluster_id_out;
} ChainTidsCtx;

static bool
walk_one_page_tids(MktPostingChainPos *pos, void *state)
{
	ChainTidsCtx			   *ctx	  = state;
	const MktPostingPageOpaque *op	  = mkt_posting_opaque(pos->page);
	Dimension					dim	  = ctx->dim;
	uint32_t					cnt	  = op->entry_count;
	uint32_t					ntids = 0;

	if (pos->first && ctx->cluster_id_out != NULL)
		*ctx->cluster_id_out = op->cluster_id;

	if (cnt > ctx->cap)
		cnt = ctx->cap; /* not reachable for a well-formed page */

	if (!(op->flags & MKT_POSTING_PAGE_TOMBSTONED))
	{
		char *content = mkt_posting_page_content(pos->page, dim);

		if (op->flags & MKT_POSTING_PAGE_FASTSCAN)
		{
			/* SoA: tids live in fixed 32-entry group sections. The last
			 * group may be partial; entry_count bounds the valid slots.
			 * Live pages have no per-entry delete flag (deletes tombstone
			 * the whole page, handled above). */
			uint32_t ngroups = (cnt + MKT_FASTSCAN_GROUP - 1) /
							   MKT_FASTSCAN_GROUP;
			for (uint32_t g = 0; g < ngroups; g++)
			{
				ItemPointerData *gt = mkt_fastscan_group_tids(content, g, dim);
				uint32_t		 base_i = g * MKT_FASTSCAN_GROUP;
				uint32_t		 valid	= (cnt - base_i) < MKT_FASTSCAN_GROUP
												? (cnt - base_i)
												: MKT_FASTSCAN_GROUP;
				for (uint32_t v = 0; v < valid; v++)
					ctx->tids[ntids++] = gt[v];
			}
		}
		else
		{
			for (uint32_t i = 0; i < cnt; i++)
			{
				MktPostingEntryHeader *hdr =
						mkt_posting_entry_at(content, i, dim);
				if (hdr->meta.flags & MKT_POSTING_FLAG_DELETED)
					continue;
				ctx->tids[ntids++] = hdr->meta.tid;
			}
		}
	}

	/*
	 * The callback may write through this same storage -- the writing pass
	 * appends to page builders, which flush a full page as they go -- and
	 * the backend holds one page at a time, so a write during the callback
	 * would take over the slot this page occupies. Hence the tids were
	 * copied out above, and the page goes before any callback runs.
	 */
	mkt_posting_chain_release(pos);

	/*
	 * Dispatch, starting the read for the next tid's block while the
	 * current one is being fetched. Only when the block changes: runs of
	 * tids share a heap block (a low-dimension table packs many rows per
	 * page), and re-requesting a block already in flight buys nothing.
	 */
	BlockNumber prefetched = InvalidBlockNumber;
	MktMemCtx	outer	   = mkt_memctx_switch(ctx->entry_ctx);

	for (uint32_t i = 0; i < ntids; i++)
	{
		if (ctx->prefetch != NULL && i + 1 < ntids)
		{
			BlockNumber nb = ItemPointerGetBlockNumber(&ctx->tids[i + 1]);
			if (nb != prefetched)
			{
				ctx->prefetch(ctx->prefetch_ctx, ctx->tids[i + 1]);
				prefetched = nb;
			}
		}
		ctx->cb(ctx->state, ctx->tids[i]);
		mkt_memctx_reset(ctx->entry_ctx);
	}

	mkt_memctx_switch(outer);
	return true;
}

static void
walk_chain_tids(
		MktStorage		  *storage,
		Dimension		   dim,
		BlockNumber		   head,
		ChainTidCb		   cb,
		void			  *state,
		const MktSplitEnv *env,
		uint32_t		  *cluster_id_out)
{
	/*
	 * One page's worth of tids, sized to whichever page format packs the
	 * most, from the same helpers page init uses to set max_entries -- so
	 * the cap cannot disagree with what a page reports.
	 */
	uint32_t		 cap  = mkt_posting_max_entries_any_format(dim);
	ItemPointerData *tids = mkt_alloc((size_t)cap * sizeof(ItemPointerData));

	/*
	 * Reset between entries, so per-entry allocations cannot accumulate over
	 * a list. This belongs here rather than in each backend's fetch: the walk
	 * is what knows there is an iteration, and leaving it to the fetch gives
	 * the guarantee on whichever backend remembered to implement it. Created
	 * after the tids buffer above, which has to outlive the resets.
	 */
	ChainTidsCtx ctx = {
			.dim	   = dim,
			.cb		   = cb,
			.state	   = state,
			.tids	   = tids,
			.cap	   = cap,
			.entry_ctx = mkt_memctx_create(
					mkt_memctx_current(), "mktann split entry"),
			.prefetch		= (env != NULL) ? env->prefetch_vector : NULL,
			.prefetch_ctx	= (env != NULL) ? env->ctx : NULL,
			.cluster_id_out = cluster_id_out,
	};

	mkt_posting_chain_walk(storage, head, walk_one_page_tids, &ctx);

	mkt_memctx_delete(ctx.entry_ctx);
	mkt_free(tids);
}

/* ----------------------------------------------------------------
 * Pass 1: count and sample
 * ---------------------------------------------------------------- */

/*
 * One entry of the counting pass. Every fetchable entry is counted; one in
 * every `stride` is copied into the sample, until the sample is full. An entry
 * whose vector can no longer be fetched drops out of the split entirely, which
 * is why the count here -- not the head's live_count -- is what the size
 * decisions use.
 */
static void
sample_one_vector(void *state, ItemPointerData tid)
{
	SplitSample *s	 = (SplitSample *)state;
	Dimension	 dim = s->dim;

	if (!s->env->fetch_vector(s->env->ctx, tid, s->one, dim))
		return;

	s->n_live++;
	if (vector_is_degenerate(s->one, dim, s->metric))
	{
		s->n_degenerate++;
		return;
	}
	/*
	 * Under cosine the clustering kernels take 1 - dot(x, c) and require unit
	 * vectors, and the bulk build normalizes before it encodes -- so the split
	 * has to as well, or it would cluster by magnitude and write codes the
	 * scan does not expect. A backend that already stores unit vectors is
	 * unaffected: normalizing again changes nothing.
	 */
	if (s->metric == DISTANCE_COSINE)
		mkt_l2_normalize(s->one, dim);

	if (s->count < s->cap)
	{
		/* Filling the reservoir. The buffer starts at what the head's count
		 * suggested and grows if that came out low. */
		if (s->count == s->alloc)
		{
			uint32_t want = (s->alloc < s->cap / 2) ? s->alloc * 2 : s->cap;
			s->vecs = mkt_realloc(s->vecs, (size_t)want * dim * sizeof(float));
			s->alloc = want;
		}
		memcpy(s->vecs + (size_t)s->count * dim,
			   s->one,
			   (size_t)dim * sizeof(float));
		s->count++;
	}
	else
	{
		/* Full: this entry takes a slot with probability cap/eligible, which
		 * leaves every eligible entry equally likely to be in the sample. */
		uint64_t j = splitmix64_next(&s->rng) % ((uint64_t)s->eligible + 1);
		if (j < s->cap)
			memcpy(s->vecs + j * dim, s->one, (size_t)dim * sizeof(float));
	}
	s->eligible++;
}

/*
 * Draw the clustering sample. `est_entries` (the head's live_count) only sets
 * the stride, so an inaccurate estimate costs sample size, never correctness:
 * too high a stride simply samples fewer than the budget allows, and too low
 * a one stops at the budget.
 */
static void
sample_chain_vectors(
		MktIndexBase	  *base,
		Dimension		   dim,
		BlockNumber		   head,
		const MktSplitEnv *env,
		uint32_t		   cap,
		uint32_t		   est_entries,
		uint64_t		   seed,
		SplitSample		  *out,
		uint32_t		  *cluster_id_out)
{
	memset(out, 0, sizeof(*out));
	out->dim	= dim;
	out->metric = base->metric;
	out->env	= env;
	out->cap	= cap;
	out->rng	= seed;

	/*
	 * Size the buffer to what this list is expected to need, not to what the
	 * budget would allow: a generous budget must not cost a generous
	 * allocation on a list that does not fill it. The buffer grows if the
	 * estimate was low.
	 */
	uint32_t want = (est_entries < cap) ? est_entries : cap;
	if (want < 64)
		want = 64;
	out->alloc = want;
	out->vecs  = mkt_alloc((size_t)want * dim * sizeof(float));
	out->one   = mkt_alloc((size_t)dim * sizeof(float));

	walk_chain_tids(
			base->posting_storage,
			dim,
			head,
			sample_one_vector,
			out,
			env,
			cluster_id_out);
}

/* ----------------------------------------------------------------
 * Undersized-cluster drop
 * ---------------------------------------------------------------- */

/* What a sample tally of `tally` implies about the real cluster size. */
static uint32_t
estimate_cluster_size(uint32_t tally, uint32_t sampled, uint32_t n_live)
{
	if (sampled == 0)
		return 0;
	if (sampled == n_live)
		return tally; /* the list was sampled whole -- exact */
	return (uint32_t)((double)tally * (double)n_live / (double)sampled + 0.5);
}

/*
 * Drop the clusters holding too few entries to be worth a posting list of
 * their own, and compact the surviving centroids into `out` (k' * dim floats,
 * in their original order). Returns k', or 0 if fewer than two survive.
 *
 * "Dropping a cluster" means dropping its centroid, before anything has been
 * assigned to it. Nothing moves as a result: the writing pass sends each entry
 * to the nearest centroid that survived, which is where a fold would have put
 * the entries this cluster would have held.
 *
 * k-means minimizes distortion, not partition size, so a cluster can come out
 * arbitrarily small -- an outlier becomes a partition of one. Writing that out
 * as a posting list costs a whole page to hold one entry, spends a leaf
 * centroid on a single vector it alone can route, and inflates nlist with
 * partitions holding almost no data -- which skews the automatic probe count
 * and the cost model, since both are derived from nlist.
 *
 * Survivors keep the position k-means elected for them; recomputing one to
 * account for a handful of absorbed outliers would encode the many worse to
 * encode the few better, and the resulting drift in nearest-partition
 * assignment is what a reassign pass exists to clean up.
 *
 * Sizes come from the sample, scaled. A cluster with too few of the sample's
 * points to clear the floor is either genuinely small or too small to measure,
 * and both are reasons not to give it a list of its own.
 */
static uint32_t
drop_undersized_clusters(
		const KMeansResult *km,
		const float		   *sample_vecs,
		uint32_t			k,
		Dimension			dim,
		DistanceMetric		metric,
		uint32_t			sampled,
		uint32_t			n_live,
		uint32_t			floor_entries,
		float			   *out)
{
	uint32_t tally[MKT_SPLIT_MAX_PARTS] = {0};
	bool	 live[MKT_SPLIT_MAX_PARTS];
	/* The sample's assignments, rewritten as centroids are dropped so the
	 * tallies keep describing what the writing pass will do. */
	ClusterId *owner = mkt_alloc((size_t)sampled * sizeof(ClusterId));

	for (uint32_t i = 0; i < sampled; i++)
	{
		owner[i] = km->assignments[i];
		tally[owner[i]]++;
	}
	for (uint32_t j = 0; j < k; j++)
		live[j] = true;

	uint32_t nlive = k;

	for (;;)
	{
		/* Smallest surviving centroid that still misses the floor. */
		uint32_t victim	   = UINT32_MAX;
		uint32_t victim_sz = 0;
		for (uint32_t j = 0; j < k; j++)
		{
			if (!live[j])
				continue;
			uint32_t sz = estimate_cluster_size(tally[j], sampled, n_live);
			if (sz >= floor_entries)
				continue;
			if (victim == UINT32_MAX || sz < victim_sz)
			{
				victim	  = j;
				victim_sz = sz;
			}
		}
		if (victim == UINT32_MAX)
			break; /* every survivor clears the floor */

		if (nlive <= 2)
		{
			/*
			 * Dropping another would leave one partition, which is not a
			 * split. Let the caller widen or decline: the data is one dense
			 * region plus stragglers, and emitting a partition of a handful of
			 * entries to be able to claim a split serves nobody.
			 */
			return 0;
		}

		/* Re-home the victim's share of the sample so the tallies keep
		 * describing what the writing pass will do -- each of its points to
		 * the nearest centroid that survives, which is where that pass will
		 * send the entries it stood for. Only its own points can move. */
		for (uint32_t i = 0; i < sampled; i++)
		{
			if (owner[i] != (ClusterId)victim)
				continue;

			Vec32Ref v		= {sample_vecs + (size_t)i * dim, dim};
			uint32_t best	= UINT32_MAX;
			Distance best_d = 0;
			for (uint32_t j = 0; j < k; j++)
			{
				if (!live[j] || j == victim)
					continue;
				Vec32Ref c = {centroid_at(km->centroids, j, dim), dim};
				Distance d = mkt_distance(v, c, metric);
				if (best == UINT32_MAX || d < best_d)
				{
					best   = j;
					best_d = d;
				}
			}
			owner[i] = (ClusterId)best;
			tally[best]++;
		}

		tally[victim] = 0;
		live[victim]  = false;
		nlive--;
	}

	if (nlive < 2)
		return 0;

	/* Compact survivors, preserving order: the caller gives the first one the
	 * retiring list's cluster id, and reordering would hand it elsewhere. */
	uint32_t next = 0;
	for (uint32_t j = 0; j < k; j++)
	{
		if (!live[j])
			continue;
		memcpy(out + (size_t)next * dim,
			   centroid_at(km->centroids, j, dim),
			   (size_t)dim * sizeof(float));
		next++;
	}
	return nlive;
}

/* ----------------------------------------------------------------
 * Pass 2: assign and write
 * ---------------------------------------------------------------- */

/*
 * Streaming writer: k page builders, and each entry appended to the one whose
 * centroid is nearest. The builders assemble pages in their own memory and
 * only touch storage to flush a full page, so holding k of them open costs k
 * page images rather than k pinned buffers.
 */
typedef struct SplitWriter
{
	MktPostingBuilder *builders;
	uint32_t		   k;
	const float		  *centroids; /* k * dim */
	uint32_t		  *counts;	  /* entries written per builder */
	Dimension		   dim;
	DistanceMetric	   metric;
	const MktSplitEnv *env;
	float			  *one;
} SplitWriter;

static void
write_one_vector(void *state, ItemPointerData tid)
{
	SplitWriter *w	 = (SplitWriter *)state;
	Dimension	 dim = w->dim;

	if (!w->env->fetch_vector(w->env->ctx, tid, w->one, dim))
		return; /* gone from the heap since the counting pass */

	uint32_t target;
	bool	 degenerate = vector_is_degenerate(w->one, dim, w->metric);

	/* Same unit-vector contract as the sampling pass, and for the same two
	 * reasons: the centroids were elected in that space, and the builder
	 * encodes against them. */
	if (!degenerate && w->metric == DISTANCE_COSINE)
		mkt_l2_normalize(w->one, dim);

	if (degenerate)
	{
		/*
		 * No centroid can claim it and it can never be a result, so the only
		 * thing the choice affects is how evenly the lists come out: give it
		 * to whichever is smallest so far.
		 */
		target = 0;
		for (uint32_t j = 1; j < w->k; j++)
			if (w->counts[j] < w->counts[target])
				target = j;
	}
	else
	{
		Vec32Ref v	  = {w->one, dim};
		Distance best = 0;
		target		  = 0;
		for (uint32_t j = 0; j < w->k; j++)
		{
			Vec32Ref c = {centroid_at(w->centroids, j, dim), dim};
			Distance d = mkt_distance(v, c, w->metric);
			if (j == 0 || d < best)
			{
				best   = d;
				target = j;
			}
		}
	}

	mkt_posting_builder_add_ex(&w->builders[target], tid, w->one, degenerate);
	w->counts[target]++;
}

static void
tombstone_page(MktPostingPageOpaque *op, void *state)
{
	(void)state;
	op->flags |= MKT_POSTING_PAGE_TOMBSTONED;
}

void
mkt_posting_chain_tombstone(MktStorage *storage, BlockNumber head)
{
	mkt_posting_chain_mutate(storage, head, tombstone_page, NULL);
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
	/*
	 * Define every output up front. Only found_page and tail_page are
	 * meaningful when this fails, and the other two are written solely on
	 * the path that finds the leaf -- so a caller that skipped the return
	 * value, or a compiler that cannot correlate found_idx's write with
	 * found_page's, sees an indeterminate value (GCC warns about exactly
	 * that at -O2).
	 */
	*found_page = InvalidBlockNumber;
	*tail_page	= InvalidBlockNumber;
	*found_idx	= 0;
	*level		= 0;

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
 * True if `page` has room for `n` more centroid entries in the split's format.
 * Generalizes mkt_centroid_page_has_room (which checks one) to n, so the flip
 * can decide up front whether all k-1 new leaves fit on the old leaf's page.
 * Uses data_size (not leaf_data_size) to match mkt_centroid_page_add_entry.
 */
static bool
centroid_page_has_room_for(Page page, Dimension dim, uint32_t n)
{
	PageHeader		  h	  = (PageHeader)page;
	MktCentroidFormat fmt = mkt_centroid_page_format(page);
	size_t			  per = mkt_centroid_meta_size(fmt) +
				 mkt_centroid_data_size(dim, fmt);
	size_t lower = mkt_centroid_meta_end(
			page, MKT_CENTROID_OPAQUE(page)->entry_count);
	return lower + (size_t)n * per <= (size_t)h->pd_upper;
}

/* ----------------------------------------------------------------
 * Split
 * ---------------------------------------------------------------- */

/*
 * Draw the clustering sample and cluster it. Writes up to
 * MKT_SPLIT_MAX_PARTS centroids into `centroids`, which the caller owns and
 * which outlives this phase, and returns how many stand (>= 2), 0 when the
 * split is declined, -1 on error.
 *
 * Called inside a context of its own, which the caller deletes: the sample
 * buffer is the split's largest allocation -- the whole budget -- and the
 * k-means workspace sits beside it, so both have to be gone before the
 * writing pass allocates. Nothing here frees anything by hand, deliberately.
 * mkt_free does nothing on the standalone allocator, where memory returns
 * only when a context is reset, so hand-freeing would bound the peak on
 * PostgreSQL and silently not bound it standalone. A context boundary means
 * the same thing on both.
 *
 * The centroids are the only result that outlives the phase, so they are
 * written into the caller's buffer rather than allocated here -- otherwise
 * they would die with the context. Its size does not depend on the sample:
 * the width can grow while widening, so it is always MKT_SPLIT_MAX_PARTS
 * wide.
 */
static int
sample_and_cluster(
		MktIndexBase		 *base,
		Dimension			  dim,
		BlockNumber			  head,
		const MktSplitConfig *cfg,
		const MktSplitEnv	 *env,
		uint32_t			  min_split_entries,
		float				 *centroids,
		uint32_t			 *cluster_id_out)
{
	/*
	 * 1. Count the list and draw the clustering sample, in one pass.
	 *
	 * The head's live_count only estimates the size -- it does not account for
	 * entries whose vector can no longer be fetched -- so it sets the sample
	 * stride and nothing else. The count this pass returns is what the size
	 * decisions use.
	 */
	uint64_t budget = (cfg != NULL && cfg->sample_budget_bytes)
							? cfg->sample_budget_bytes
							: MKT_SPLIT_SAMPLE_BUDGET_BYTES;
	uint32_t cap	= mkt_split_sample_cap(budget, dim);
	if (cap == 0)
		return -1; /* the budget cannot pay for a split at this dimension */
	uint32_t est = head_live_count(base->posting_storage, head);

	SplitSample smp;
	uint32_t	cluster_id = 0;
	uint64_t	seed	   = (cfg != NULL && cfg->km_seed) ? cfg->km_seed
														   : MKT_SPLIT_SAMPLE_SEED;
	sample_chain_vectors(
			base, dim, head, env, cap, est, seed, &smp, &cluster_id);

	if (smp.n_live < min_split_entries || smp.n_live < 2)
		return 0; /* declined: too few entries */

	/*
	 * 2. Re-check the size now that the list has been counted, then pick k.
	 *
	 * With a target size the caller gets both from one number: the split is
	 * declined unless the counted entries still exceed the trigger, and the
	 * width is round(count / target) so each new list rests at the target.
	 * Without a target, k defaults to 2 and cfg->nparts may override.
	 *
	 * Clamp to [2, MKT_SPLIT_MAX_PARTS] and to the sample size (k-means needs
	 * k <= n). A list needing more than MKT_SPLIT_MAX_PARTS is narrowed to the
	 * cap and stays above the trigger; a further pass splits it again.
	 */
	uint32_t target = (cfg != NULL) ? cfg->target_entries : 0;
	uint32_t k;

	if (target > 0)
	{
		if ((uint64_t)smp.n_live <= mkt_split_trigger(target))
			return 0; /* declined: inside the operating band */

		/*
		 * Round, do not ceil. The width has to put each new list *at* the
		 * target, and at the trigger the count is just over target*factor:
		 * ceil would give factor+1 parts and land them all below the target
		 * (0.67*target at factor 2), so a fresh list would start below where
		 * it is supposed to rest. Rounding gives exactly `factor` parts there
		 * -- a bisection at factor 2, as in LIRE -- and keeps parts within
		 * [0.75, 1.25] of the target everywhere else.
		 */
		uint64_t want = ((uint64_t)smp.n_live + target / 2) / target;
		if (want < 2)
			want = 2;
		k = (want > MKT_SPLIT_MAX_PARTS) ? MKT_SPLIT_MAX_PARTS
										 : (uint32_t)want;
	}
	else
	{
		k = (cfg != NULL && cfg->nparts > 2) ? cfg->nparts : 2;
	}

	/* Only entries a centroid can represent were sampled. */
	if (smp.count < 2)
	{
		/* Nothing to cluster: every entry is unreachable under this metric,
		 * and no arrangement of them improves a query. */
		return 0;
	}

	KMeansOptions opts = MKT_KMEANS_OPTIONS_DEFAULT;
	if (cfg != NULL)
	{
		if (cfg->km_max_iter)
			opts.max_iterations = cfg->km_max_iter;
		if (cfg->km_seed)
			opts.seed = cfg->km_seed;
	}

	if (k > MKT_SPLIT_MAX_PARTS)
		k = MKT_SPLIT_MAX_PARTS;
	if (k > smp.count)
		k = smp.count;

	/*
	 * When the list did not fit the budget, ask for no more partitions than
	 * the sample can speak for -- see MKT_SPLIT_MIN_SAMPLE_PER_PART. A list
	 * too big for its budget is split less far per pass, not split badly.
	 * A list sampled whole needs no such limit: its tallies are the real
	 * sizes, not estimates of them.
	 */
	uint32_t clusterable = smp.n_live - smp.n_degenerate;
	if (smp.count < clusterable)
	{
		uint32_t k_sample_max = smp.count / MKT_SPLIT_MIN_SAMPLE_PER_PART;
		if (k_sample_max < 2)
			k_sample_max = 2;
		if (k > k_sample_max)
			k = k_sample_max;
	}

	/*
	 * The floor a partition must clear to be worth its own centroid. With a
	 * target that is the bottom of the operating band, so a split never emits
	 * a list maintenance would immediately want to merge away. Without one
	 * there is no size to reason from and the caller is asking for an
	 * unconditional split, so only empty clusters are dropped.
	 */
	uint32_t floor_entries = (target > 0) ? target / MKT_SPLIT_TRIGGER_FACTOR
										  : 1;
	if (floor_entries < 1)
		floor_entries = 1;

	/*
	 * Cluster the sample, then drop the clusters holding too few entries to be
	 * worth a list of their own (see drop_undersized_clusters -- an empty
	 * cluster is just the extreme case, so no separate empty check is
	 * needed).
	 *
	 * If dropping cannot leave two centroids standing, widen and retry before
	 * giving up. At the trigger the width is a bisection, so a list that is
	 * one dense region plus a straggler has no second partition clearing the
	 * floor and would decline -- and, the k-means seed being fixed, decline
	 * again on every later pass, leaving a list above the trigger forever. One
	 * more partition usually resolves it: the bulk splits in two and the
	 * straggler goes to whichever half is nearer. Bounded, because the answer
	 * for genuinely unsplittable data (identical vectors, say) is to decline
	 * rather than to keep re-clustering.
	 */
	uint32_t nparts = 0;

	for (uint32_t attempt = 0; attempt <= MKT_SPLIT_WIDEN_ATTEMPTS; attempt++)
	{
		KMeansResult *km = mkt_kmeans_f32(
				smp.vecs, smp.count, dim, k, base->metric, &opts);
		if (km == NULL)
			return -1;

		nparts = drop_undersized_clusters(
				km,
				smp.vecs,
				k,
				dim,
				base->metric,
				smp.count,
				smp.n_live - smp.n_degenerate,
				floor_entries,
				centroids);
		mkt_kmeans_result_destroy(km);

		if (nparts >= 2)
			break;
		if (k + 1 > MKT_SPLIT_MAX_PARTS || k + 1 > smp.count)
			break;
		k++;
	}
	if (nparts < 2)
		return 0; /* degenerate: no two partitions clear the floor */

	*cluster_id_out = cluster_id;
	return (int)nparts;
}

/*
 * Release what the writing pass allocated. Both the error path out of the
 * flip and the normal end need exactly this, and a seventh allocation added
 * to one copy and not the other is how that kind of pair goes wrong.
 *
 * A context would also collapse the two, but the writing pass has nothing to
 * release early -- everything here lives until the split ends -- so the
 * problem is duplication, not peak, and this is the smaller tool for it.
 */
static void
write_phase_cleanup(
		RaBitQScratch *scratch,
		RaBitQData	 **rd,
		uint32_t	   k,
		float		  *pt_res,
		float		  *pt_c,
		float		  *centroids)
{
	mkt_rabitq_scratch_cleanup(scratch);
	for (uint32_t j = 0; j < k; j++)
		mkt_free(rd[j]);
	mkt_free_aligned(pt_res);
	mkt_free_aligned(pt_c);
	mkt_free(centroids);
}

/*
 * Split the oversized posting list at `head` into two balanced lists.
 *
 * Locking. The caller holds an exclusive lock on this cluster (in PG a
 * heavyweight page lock on the head; a no-op standalone). That lock, not the
 * index-relation lock, is the concurrency gate: convert/split/insert take only
 * the index's RowExclusiveLock, which does not conflict with itself, so the
 * per-cluster lock is what serializes this split against inserts and other
 * splits of the same list and makes the snapshot of the old entries and the
 * centroid flip atomic with respect to writers. The caller releases it once
 * this returns.
 *
 * Why new blocks instead of an in-place rewrite. The split builds two fresh
 * chains, then repoints the centroid leaf at them in a single page write (the
 * flip in step 7). That flip is the atomic commit point: before it the old
 * list is authoritative, after it the two new lists are. Consequences:
 *   - Crash-safety: a crash can only leak the freshly written, still
 *     unreferenced pages; it can never corrupt or lose entries, because the
 *     old chain is untouched until the flip and the new chains are fully
 *     committed before it.
 *   - Concurrency: a scanner that read the leaf pointer *before* the flip
 *     still holds the old head and must see the complete pre-split list. New
 *     blocks leave that old chain byte-for-byte intact, so such a scanner
 *     reads it correctly. An in-place reorder would move entries out from
 *     under that scanner (it would miss whatever was routed to the other new
 *     list). This is why the old chain is retired, not overwritten: a backend
 *     with MVCC snapshots defers reclaiming it behind a visibility gate
 *     (env->retire_chain) so an in-flight scan is never cut off; standalone,
 *     where no snapshot-holding scanner exists, it is tombstoned immediately.
 *     A retired chain's pages are not reused: they stay allocated to the
 *     relation once the chain is tombstoned.
 */
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
	/*
	 * 1-2. Sample, count and cluster, in a context that goes away with the
	 * sample buffer inside it. See sample_and_cluster: the phase boundary is
	 * a context boundary so that it bounds the peak on both allocators, not
	 * just on the one where mkt_free frees.
	 */
	float *centroids = mkt_alloc(
			(size_t)MKT_SPLIT_MAX_PARTS * dim * sizeof(float));

	MktMemCtx work =
			mkt_memctx_create(mkt_memctx_current(), "mktann split sample");
	MktMemCtx old = mkt_memctx_switch(work);

	uint32_t cluster_id = 0;
	int		 nparts		= sample_and_cluster(
			 base,
			 dim,
			 head,
			 cfg,
			 env,
			 min_split_entries,
			 centroids,
			 &cluster_id);

	mkt_memctx_switch(old);
	mkt_memctx_delete(work);

	if (nparts < 2)
	{
		if (out != NULL)
			*out = res;
		return nparts < 0 ? -1 : 0;
	}
	uint32_t k = (uint32_t)nparts;

	/*
	 * 3. Write the k new lists in one streaming pass, encoding each entry
	 * against the centroid of the list it lands in. New lists take the index's
	 * posting format, so a split also upgrades a list that had drifted to AoS
	 * back to fastscan (a re-optimization point). new_head[0] reuses the old
	 * cluster id; new_head[1..] get fresh ids.
	 */
	/*
	 * Reserve the ids durably before using them -- see
	 * MktSplitEnv.reserve_nlist. Raised first, so a crash before the new
	 * leaves are reachable leaves the count too high, which costs nothing but
	 * a gap in the ids; the other order hands the same ids out twice.
	 */
	uint32_t first_new_id = base->nlist;
	base->nlist += (k - 1);
	if (env->reserve_nlist != NULL)
		env->reserve_nlist(env->ctx, base->nlist);

	bool		  fastscan = (base->fastscan != 0);
	BlockNumber	  new_head[MKT_SPLIT_MAX_PARTS];
	RaBitQData	 *rd[MKT_SPLIT_MAX_PARTS];
	uint32_t	  counts[MKT_SPLIT_MAX_PARTS] = {0};
	float		 *pt_c	 = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	float		 *pt_res = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	RaBitQScratch scratch;
	mkt_rabitq_scratch_init(&scratch, dim);

	MktPostingBuilder *builders = mkt_alloc(
			(size_t)k * sizeof(MktPostingBuilder));
	for (uint32_t j = 0; j < k; j++)
	{
		const float *centroid = centroid_at(centroids, j, dim);
		mkt_rabitq_rotate(params, centroid, pt_c);

		/* Not `cid`: that is a command id in PostgreSQL. */
		uint32_t new_cluster = (j == 0) ? cluster_id : first_new_id + (j - 1);
		mkt_posting_builder_init_fmt(
				&builders[j],
				base->posting_storage,
				params,
				dim,
				new_cluster,
				centroid,
				pt_c,
				fastscan);

		/* Routing centroid for the leaf entry: encode (pt_c - pt_mean), which
		 * matches encode_into(c, mean) used at build. */
		rd[j] = mkt_alloc(MKT_RABITQ_DATA_SIZE(dim));
		for (Dimension d = 0; d < dim; d++)
			pt_res[d] = pt_c[d] - base->pt_global_mean[d];
		mkt_rabitq_encode_from_pt(params, pt_res, rd[j], &scratch);
	}

	SplitWriter w = {
			.builders  = builders,
			.k		   = k,
			.centroids = centroids,
			.counts	   = counts,
			.dim	   = dim,
			.metric	   = base->metric,
			.env	   = env,
			.one	   = mkt_alloc((size_t)dim * sizeof(float)),
	};
	walk_chain_tids(
			base->posting_storage, dim, head, write_one_vector, &w, env, NULL);
	mkt_free(w.one);

	for (uint32_t j = 0; j < k; j++)
		new_head[j] = mkt_posting_builder_finish(&builders[j]);
	for (uint32_t j = 0; j < k; j++)
		mkt_posting_builder_cleanup(&builders[j]);
	mkt_free(builders);
	/* 4. Flip the centroid tree: locate the old leaf entry + chain tail. */
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
		write_phase_cleanup(&scratch, rd, k, pt_res, pt_c, centroids);
		return -1;
	}

	/*
	 * Test hook: fires with the new chains fully written and committed but
	 * the tree still pointing at the old head, which is the window the
	 * crash-safety argument rests on. A crash here must leave the old list
	 * authoritative and leak only the new, unreferenced pages.
	 */
	MKT_INJECTION_POINT("mktann-split-before-flip");

	/*
	 * Flip: repoint the old leaf at new_head[0] and add leaves for the k-1
	 * extra heads.
	 *
	 * Fast path (the common flat single-page tree): when the old leaf and the
	 * chain tail are the same centroid page and all k-1 extra leaves fit on
	 * it, do the whole flip in one page write. A concurrent scan reading that
	 * page then observes only the pre- or post-flip state -- never both the
	 * old head and a new head -- so it never scans (then dedups) the same
	 * vectors twice. This is one buffer-lock hold and one WAL record, cheaper
	 * than the per-entry commits below, not a heavyweight lock.
	 */
	uint32_t appended_centroid_pages = 0;
	bool	 single_page			 = false;
	if (found_page == tail_page)
	{
		Page cp = mkt_storage_read_page(base->centroid_storage, found_page);
		single_page = centroid_page_has_room_for(cp, dim, k - 1);
		mkt_storage_release_page(base->centroid_storage, found_page);
	}

	if (single_page)
	{
		Page fp = mkt_storage_write_page(base->centroid_storage, found_page);
		mkt_centroid_page_overwrite_entry(
				fp, dim, found_idx, new_head[0], rd[0]);
		for (uint32_t j = 1; j < k; j++)
			mkt_centroid_page_add_entry(
					fp, dim, new_head[j], 0, MKT_CENTROID_FLAG_LEAF, rd[j]);
		mkt_storage_commit_page(base->centroid_storage, found_page);
	}
	else
	{
		/*
		 * Fallback (tail on another page, or the page can't hold all k-1): a
		 * multi-page flip can't be one atomic write. Append the k-1 new leaves
		 * first -- chaining a new level-0 page as the tail fills -- then
		 * overwrite the old leaf last. Appending first keeps the old leaf on
		 * the full old list until that final overwrite, so a mid-flip scan
		 * never misses entries; it may briefly reach both a new head and the
		 * old head, but those duplicate vectors are dropped by the top-k's id
		 * dedup (the same path that dedups SOAR replicas), so results stay
		 * correct.
		 */
		for (uint32_t j = 1; j < k; j++)
		{
			Page tp =
					mkt_storage_write_page(base->centroid_storage, tail_page);
			bool appended = mkt_centroid_page_add_entry(
					tp, dim, new_head[j], 0, MKT_CENTROID_FLAG_LEAF, rd[j]);
			mkt_storage_commit_page(base->centroid_storage, tail_page);

			if (!appended)
			{
				BlockNumber np;
				Page npg = mkt_storage_new_page(base->centroid_storage, &np);
				mkt_centroid_page_init_fmt(npg, level, base->centroid_format);
				mkt_centroid_page_add_entry(
						npg,
						dim,
						new_head[j],
						0,
						MKT_CENTROID_FLAG_LEAF,
						rd[j]);
				mkt_storage_commit_page(base->centroid_storage, np);

				Page link = mkt_storage_write_page(
						base->centroid_storage, tail_page);
				MKT_CENTROID_OPAQUE(link)->next_blkno = np;
				mkt_storage_commit_page(base->centroid_storage, tail_page);

				tail_page = np;
				appended_centroid_pages++;
			}
		}

		Page fp = mkt_storage_write_page(base->centroid_storage, found_page);
		mkt_centroid_page_overwrite_entry(
				fp, dim, found_idx, new_head[0], rd[0]);
		mkt_storage_commit_page(base->centroid_storage, found_page);
	}

	/*
	 * 5. Retire the old chain, now unreachable via the tree. A concurrent
	 * scanner may still hold a stale head pointer read before the flip, so the
	 * backend that can see snapshots (PG) defers reclaim behind an XID gate,
	 * keeping the chain readable meanwhile. Where there are no such scanners
	 * (env->retire_chain == NULL, e.g. standalone), tombstone it immediately.
	 */
	if (env->retire_chain != NULL)
		env->retire_chain(env->ctx, base->posting_storage, head);
	else
		mkt_posting_chain_tombstone(base->posting_storage, head);

	res.did_split = true;
	res.nparts	  = k;
	for (uint32_t j = 0; j < k; j++)
	{
		res.head[j]	 = new_head[j];
		res.count[j] = counts[j];
	}
	res.new_nlist		   = base->nlist;
	res.new_centroid_pages = appended_centroid_pages;
	base->ncentroid_pages += appended_centroid_pages;

	write_phase_cleanup(&scratch, rd, k, pt_res, pt_c, centroids);

	if (out != NULL)
		*out = res;
	return 0;
}
