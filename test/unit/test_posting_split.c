/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_posting_split.c - Unit tests for incremental posting-list split
 *
 * Builds a single-partition paged index, splits it with prism_posting_split,
 * and verifies:
 *   - the split preserves every entry and produces non-empty leaves (2-way and
 *     k-way via nparts)
 *   - every vector is still retrievable through the mutated centroid tree
 *   - recall after an incremental split matches a from-scratch 2-partition
 *     build (within noise)
 */

#include <math.h>
#include <string.h>

#include "algo/distance.h"
#include "core/memory.h"
#include "index/centroid_page.h"
#include "index/index_build.h"
#include "index/posting_page.h"
#include "index/posting_split.h"
#include "quant/rabitq.h"
#include "standalone/index.h"
#include "standalone/query.h"
#include "vs_test.h"

TEST_GROUP(PostingSplit);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

/*
 * Two well-separated blobs so 2-means recovers a clean partition: the first
 * half clusters around 0, the second half around +8 in every dimension.
 */
static float *
make_two_blobs(uint32_t nvecs, uint32_t dim, uint32_t seed)
{
	srand(seed);
	float *data = vs_alloc((size_t)nvecs * dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
	{
		float center = (i < nvecs / 2) ? 0.0f : 8.0f;
		for (uint32_t j = 0; j < dim; j++)
			data[(size_t)i * dim + j] = center +
										(float)(rand() % 2000 - 1000) /
												1000.0f;
	}
	return data;
}

/* Three well-separated blobs (centers 0, 8, 16) so k-means with k=3 recovers a
 * clean partition -- for the k-way (nparts > 2) split. */
static float *
make_three_blobs(uint32_t nvecs, uint32_t dim, uint32_t seed)
{
	srand(seed);
	float *data = vs_alloc((size_t)nvecs * dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
	{
		float center = (float)((i / (nvecs / 3 + 1)) * 8);
		for (uint32_t j = 0; j < dim; j++)
			data[(size_t)i * dim + j] = center +
										(float)(rand() % 2000 - 1000) /
												1000.0f;
	}
	return data;
}

/*
 * The one build every test here starts from: paged posting lists with RaBitQ
 * centroids, built single-threaded from an in-RAM array. Only the metric and
 * the fastscan width ever vary, so the named wrappers below pass those and
 * share everything else.
 */
static PrismIndex *
build_paged_with(
		const float	  *vecs,
		uint32_t	   nvecs,
		uint32_t	   dim,
		uint32_t	   nlist,
		DistanceMetric metric,
		int			   fastscan)
{
	PrismIndexConfig config = {
			.nlist		   = nlist,
			.metric		   = metric,
			.centroid_fmt  = PRISM_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
			.posting_fmt   = PRISM_POSTING_FMT_PAGES,
			.fastscan	   = fastscan,
			.nworkers	   = 0,
	};
	VsArraySource src;
	vs_array_source_init(&src, vecs, nvecs, dim);
	return prism_index_build(&src.base, &config, NULL);
}

static PrismIndex *
build_paged(const float *vecs, uint32_t nvecs, uint32_t dim, uint32_t nlist)
{
	return build_paged_with(vecs, nvecs, dim, nlist, DISTANCE_L2, 0);
}

/* Build under cosine, where a zero-norm vector has no defined distance. */
static PrismIndex *
build_paged_cosine(
		const float *vecs, uint32_t nvecs, uint32_t dim, uint32_t nlist)
{
	return build_paged_with(vecs, nvecs, dim, nlist, DISTANCE_COSINE, 0);
}

/* Build with fastscan-packed posting pages (SoA groups), to exercise the
 * fastscan branch of the split's entry collection. */
static PrismIndex *
build_paged_fastscan(
		const float *vecs, uint32_t nvecs, uint32_t dim, uint32_t nlist)
{
	return build_paged_with(vecs, nvecs, dim, nlist, DISTANCE_L2, 16);
}

/*
 * The preamble nearly every test below opens with: `nvecs_` clustered vectors
 * of `dim_` dimensions, built into a single posting list for the split to cut
 * up. Declares dim, n, vecs and idx, so bodies keep referring to them by
 * name.
 */
#define DECLARE_TWO_BLOB_INDEX(nvecs_, dim_, seed_)     \
	uint32_t	dim = (dim_), n = (nvecs_);             \
	float	   *vecs = make_two_blobs(n, dim, (seed_)); \
	PrismIndex *idx	 = build_paged(vecs, n, dim, 1)

/* The same, with three blobs, for the k-way (nparts > 2) splits. */
#define DECLARE_THREE_BLOB_INDEX(nvecs_, dim_, seed_)     \
	uint32_t	dim = (dim_), n = (nvecs_);               \
	float	   *vecs = make_three_blobs(n, dim, (seed_)); \
	PrismIndex *idx	 = build_paged(vecs, n, dim, 1)

/* Split env: fetch full-precision vectors from the in-RAM store by id. */
typedef struct FetchCtx
{
	const float *all;
} FetchCtx;

static bool
fetch_vec(void *ctx, ItemPointerData tid, float *out, Dimension dim)
{
	const float *all = ((FetchCtx *)ctx)->all;
	uint32_t	 vid = prism_posting_get_vector_id(&tid);
	memcpy(out, all + (size_t)vid * dim, (size_t)dim * sizeof(float));
	return true;
}

/* True if the posting head at `blk` is stored in fastscan (SoA) format. */
static bool
head_is_fastscan(PrismIndexBase *base, BlockNumber blk)
{
	Page p	= vs_storage_read_page(base->posting_storage, blk);
	bool fs = (prism_posting_opaque(p)->flags & PRISM_POSTING_PAGE_FASTSCAN) !=
			  0;
	vs_storage_release_page(base->posting_storage, blk);
	return fs;
}

/* Number of centroid pages linked from the (single-level) tree root. */
static uint32_t
centroid_page_count(PrismIndexBase *base)
{
	uint32_t	n	= 0;
	BlockNumber blk = base->first_centroid;
	while (blk != InvalidBlockNumber && n <= 1024)
	{
		Page		p	 = vs_storage_read_page(base->centroid_storage, blk);
		BlockNumber next = PRISM_CENTROID_OPAQUE(p)->next_blkno;
		vs_storage_release_page(base->centroid_storage, blk);
		blk = next;
		n++;
	}
	return n;
}

/*
 * Leaf entries across the flat centroid chain, split into the real ones and
 * the never-followed fillers fill_centroid_page adds (child_blkno invalid).
 * Lets a test say what the flip changed rather than only what it produced.
 */
static void
count_leaf_entries(PrismIndexBase *base, uint32_t *total, uint32_t *fillers)
{
	*total			= 0;
	*fillers		= 0;
	BlockNumber blk = base->first_centroid;

	while (blk != InvalidBlockNumber)
	{
		Page p = vs_storage_read_page(base->centroid_storage, blk);
		PrismCentroidPageOpaque *op	  = PRISM_CENTROID_OPAQUE(p);
		uint32_t				 n	  = op->entry_count;
		BlockNumber				 next = op->next_blkno;

		for (uint32_t i = 0; i < n; i++)
		{
			const PrismCentroidEntryMeta *m = prism_centroid_meta(p, i);

			if ((m->flags & PRISM_CENTROID_FLAG_LEAF) == 0)
				continue;
			(*total)++;
			if (m->child_blkno == InvalidBlockNumber)
				(*fillers)++;
		}
		vs_storage_release_page(base->centroid_storage, blk);
		blk = next;
	}
}

static void
brute_force_knn(
		const float *vecs,
		uint32_t	 nvecs,
		uint32_t	 dim,
		const float *query,
		uint32_t	 k,
		uint32_t	*ids)
{
	float *dists = vs_alloc(nvecs * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
	{
		float d = 0;
		for (uint32_t j = 0; j < dim; j++)
		{
			float diff = vecs[(size_t)i * dim + j] - query[j];
			d += diff * diff;
		}
		dists[i] = d;
	}
	for (uint32_t i = 0; i < k; i++)
	{
		uint32_t best = 0;
		for (uint32_t j = 1; j < nvecs; j++)
			if (dists[j] < dists[best])
				best = j;
		ids[i]		= best;
		dists[best] = INFINITY;
	}
	vs_free(dists);
}

static double
measure_recall(
		PrismIndex	*idx,
		const float *vecs,
		uint32_t	 nvecs,
		uint32_t	 dim,
		uint32_t	 k,
		uint32_t	 nprobe,
		uint32_t	 nq)
{
	PrismQueryCtx *q	   = prism_query_ctx_create(idx, k, nprobe);
	uint32_t	   hits	   = 0;
	uint32_t	   res[16] = {0};
	uint32_t	   gt[16]  = {0};
	uint32_t	   stride  = nvecs / nq;

	for (uint32_t i = 0; i < nq; i++)
	{
		const float *query = vecs + (size_t)(i * stride) * dim;
		uint32_t	 c	   = prism_query_exec(
				q, query, k, nprobe, VS_DISTANCE_MODE_ASYMMETRIC, true, res);
		brute_force_knn(vecs, nvecs, dim, query, k, gt);
		for (uint32_t a = 0; a < c; a++)
			for (uint32_t g = 0; g < k; g++)
				if (res[a] == gt[g])
					hits++;
	}
	prism_query_ctx_destroy(q);
	return (double)hits / ((double)nq * k);
}

/* ----------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------- */

TEST(split_preserves_entries_and_structure)
{
	DECLARE_TWO_BLOB_INDEX(1200, 16, 42);
	ASSERT_NOT_NULL(idx, "paged build should succeed");
	ASSERT_EQ(idx->base.nlist, 1u, "starts as a single partition");
	ASSERT_EQ(idx->base.nlevels, 1u, "flat tree");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitResult res;
	int				 rc = prism_posting_split(
			 &idx->base, idx->first_posting, NULL, &env, &res);

	ASSERT_EQ(rc, 0, "split should succeed");
	ASSERT_TRUE(res.did_split, "split should happen");
	ASSERT_EQ(res.new_nlist, 2u, "one leaf becomes two");
	ASSERT_EQ(idx->base.nlist, 2u, "nlist bumped in base");
	/* The split streams the chain twice and never holds it, so this equality
	 * is what says the second pass placed every entry the first pass counted
	 * -- including any the reservoir never sampled. */
	ASSERT_EQ(res.count[0] + res.count[1], n, "every entry preserved");
	ASSERT_TRUE(res.count[0] > 0 && res.count[1] > 0, "both leaves non-empty");

	prism_index_destroy(idx);
}

/* Records that the retire hook fired and matches the default behavior (make
 * the old chain unreachable) so the split stays correct. */
static int split_retire_calls = 0;

static void
test_retire_chain(void *ctx, VsStorage *storage, BlockNumber head)
{
	(void)ctx;
	split_retire_calls++;
	prism_posting_chain_tombstone(storage, head);
}

/* A backend with MVCC snapshots supplies env->retire_chain to defer reclaim;
 * exercise that dispatch branch (the other tests leave it NULL). */
TEST(split_invokes_retire_chain_hook)
{
	DECLARE_TWO_BLOB_INDEX(800, 16, 7);
	ASSERT_NOT_NULL(idx, "paged build should succeed");

	FetchCtx fc		   = {idx->all_vectors};
	split_retire_calls = 0;
	PrismSplitEnv env  = {
			 .fetch_vector = fetch_vec,
			 .retire_chain = test_retire_chain,
			 .ctx		   = &fc,
	 };
	PrismSplitResult res;
	int				 rc = prism_posting_split(
			 &idx->base, idx->first_posting, NULL, &env, &res);

	ASSERT_EQ(rc, 0, "split should succeed");
	ASSERT_TRUE(res.did_split, "split should happen");
	ASSERT_EQ(
			split_retire_calls,
			1,
			"retire_chain invoked once for the old chain");

	prism_index_destroy(idx);
}

/* N-way split: cfg.nparts = 3 turns one oversized list into three right-sized
 * lists in a single pass (vs. two 2-way passes). */
TEST(split_three_way)
{
	DECLARE_THREE_BLOB_INDEX(1500, 16, 11);
	ASSERT_NOT_NULL(idx, "paged build should succeed");
	/* The counts below are 1 + (k - 1); assert the 1, so a fixture that
	 * stopped building a single list says so here instead of surfacing as a
	 * wrong post-split count. */
	ASSERT_EQ(idx->base.nlist, 1u, "fixture builds one list");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitConfig cfg = {.nparts = 3};
	PrismSplitResult res;
	int				 rc = prism_posting_split(
			 &idx->base, idx->first_posting, &cfg, &env, &res);

	ASSERT_EQ(rc, 0, "split should succeed");
	ASSERT_TRUE(res.did_split, "split should happen");
	ASSERT_EQ(res.nparts, 3u, "one leaf becomes three");
	ASSERT_EQ(res.new_nlist, 3u, "nlist is three after a 3-way split");
	ASSERT_EQ(idx->base.nlist, 3u, "nlist bumped by k-1 = 2");
	ASSERT_EQ(
			res.count[0] + res.count[1] + res.count[2],
			n,
			"every entry preserved across the three lists");
	ASSERT_TRUE(
			res.count[0] > 0 && res.count[1] > 0 && res.count[2] > 0,
			"all three leaves non-empty");

	prism_index_destroy(idx);
}

TEST(split_keeps_vectors_retrievable)
{
	DECLARE_TWO_BLOB_INDEX(1200, 16, 7);
	ASSERT_NOT_NULL(idx, "build ok");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitResult res;
	ASSERT_EQ(
			prism_posting_split(
					&idx->base, idx->first_posting, NULL, &env, &res),
			0,
			"split ok");
	ASSERT_TRUE(res.did_split, "split happened");

	/* Self-retrieval: each vector should find itself as the top-1 through the
	 * mutated tree (rerank makes this exact). Sample every 5th vector. */
	PrismQueryCtx *q	   = prism_query_ctx_create(idx, 1, 2);
	uint32_t	   checked = 0, found = 0;
	for (uint32_t vid = 0; vid < n; vid += 5)
	{
		uint32_t res_id = UINT32_MAX;
		uint32_t c		= prism_query_exec(
				 q,
				 vecs + (size_t)vid * dim,
				 1,
				 2,
				 VS_DISTANCE_MODE_ASYMMETRIC,
				 true,
				 &res_id);
		checked++;
		if (c == 1 && res_id == vid)
			found++;
	}
	prism_query_ctx_destroy(q);

	double self_recall = (double)found / (double)checked;
	ASSERT_TRUE(
			self_recall > 0.98, "nearly all vectors retrievable post-split");

	prism_index_destroy(idx);
}

TEST(split_fastscan_posting)
{
	/* Fastscan posting pages store TIDs in SoA 32-groups; the split must read
	 * them correctly (the AoS reader would misalign). */
	uint32_t	dim = 16, n = 1200;
	float	   *vecs = make_two_blobs(n, dim, 99);
	PrismIndex *idx	 = build_paged_fastscan(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx, "fastscan build ok");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitResult res;
	ASSERT_EQ(
			prism_posting_split(
					&idx->base, idx->first_posting, NULL, &env, &res),
			0,
			"split ok");
	ASSERT_TRUE(res.did_split, "split happened");
	ASSERT_EQ(
			res.count[0] + res.count[1], n, "every entry preserved from SoA");

	PrismQueryCtx *q	   = prism_query_ctx_create(idx, 1, 2);
	uint32_t	   checked = 0, found = 0;
	for (uint32_t vid = 0; vid < n; vid += 5)
	{
		uint32_t res_id = UINT32_MAX;
		uint32_t c		= prism_query_exec(
				 q,
				 vecs + (size_t)vid * dim,
				 1,
				 2,
				 VS_DISTANCE_MODE_ASYMMETRIC,
				 true,
				 &res_id);
		checked++;
		if (c == 1 && res_id == vid)
			found++;
	}
	prism_query_ctx_destroy(q);
	ASSERT_TRUE(
			(double)found / checked > 0.98,
			"retrievable after fastscan split");

	prism_index_destroy(idx);
}

TEST(split_declines_below_threshold)
{
	DECLARE_TWO_BLOB_INDEX(1200, 16, 5);
	ASSERT_NOT_NULL(idx, "build ok");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitConfig cfg = {.min_split_entries = 1000000}; /* never reached */
	PrismSplitResult res;
	int				 rc = prism_posting_split(
			 &idx->base, idx->first_posting, &cfg, &env, &res);

	ASSERT_EQ(rc, 0, "declining is not an error");
	ASSERT_FALSE(res.did_split, "too few entries -> no split");
	ASSERT_EQ(idx->base.nlist, 1u, "nlist unchanged");

	prism_index_destroy(idx);
}

TEST(split_handles_degenerate_data)
{
	/* All identical vectors: the two centroids coincide. The split must not
	 * crash and must preserve every entry regardless of how 2-means breaks the
	 * tie. */
	uint32_t dim = 16, n = 300;
	float	*vecs = vs_alloc((size_t)n * dim * sizeof(float));
	for (uint32_t i = 0; i < n; i++)
		for (uint32_t j = 0; j < dim; j++)
			vecs[(size_t)i * dim + j] = 1.5f;

	PrismIndex *idx = build_paged(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx, "build ok");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitResult res;
	int				 rc = prism_posting_split(
			 &idx->base, idx->first_posting, NULL, &env, &res);

	/* Identical vectors cannot be partitioned into two groups a centroid
	 * tells apart, and widening cannot change that -- so this is the case
	 * where declining is the right answer, and it must decline rather than
	 * emit partitions that only differ by which duplicate landed where. */
	ASSERT_EQ(rc, 0, "split should not error on degenerate data");
	ASSERT_FALSE(res.did_split, "identical vectors -> declined");
	ASSERT_EQ(idx->base.nlist, 1u, "declined -> nlist unchanged");

	prism_index_destroy(idx);
}

TEST(split_recall_matches_rebuild)
{
	uint32_t dim = 16, n = 1200, k = 10, nprobe = 2, nq = 20;
	float	*vecs = make_two_blobs(n, dim, 123);

	/* Incremental: build one partition, then split it. */
	PrismIndex *idx_a = build_paged(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx_a, "build A ok");
	FetchCtx		 fc	 = {idx_a->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitResult res;
	ASSERT_EQ(
			prism_posting_split(
					&idx_a->base, idx_a->first_posting, NULL, &env, &res),
			0,
			"split ok");
	ASSERT_TRUE(res.did_split, "split happened");
	double recall_a = measure_recall(idx_a, vecs, n, dim, k, nprobe, nq);

	/* From scratch: build two partitions directly. */
	PrismIndex *idx_b = build_paged(vecs, n, dim, 2);
	ASSERT_NOT_NULL(idx_b, "build B ok");
	double recall_b = measure_recall(idx_b, vecs, n, dim, k, nprobe, nq);

	ASSERT_TRUE(recall_a > 0.5, "post-split recall is reasonable");
	ASSERT_TRUE(
			fabs(recall_a - recall_b) < 0.15,
			"incremental split recall tracks a from-scratch build");

	prism_index_destroy(idx_a);
	prism_index_destroy(idx_b);
}

/*
 * Routing quality with nprobe < number of lists. The other tests probe every
 * leaf (nprobe >= nlist), so they pass even if the split's new centroids route
 * poorly. Here a 3-way split leaves three lists but each query visits only one
 * (nprobe = 1): a query at a blob center must still find its own vectors,
 * which holds only if the new centroids place each blob in its own list.
 */
TEST(split_routing_nprobe_below_k)
{
	DECLARE_THREE_BLOB_INDEX(1500, 16, 11);
	ASSERT_NOT_NULL(idx, "build ok");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitConfig cfg = {.nparts = 3};
	PrismSplitResult res;
	ASSERT_EQ(
			prism_posting_split(
					&idx->base, idx->first_posting, &cfg, &env, &res),
			0,
			"split ok");
	ASSERT_TRUE(res.did_split && res.nparts == 3u, "3-way split happened");

	/* nprobe = 1: each query reaches a single list. */
	PrismQueryCtx *q	   = prism_query_ctx_create(idx, 1, 1);
	uint32_t	   checked = 0, found = 0;
	for (uint32_t vid = 0; vid < n; vid += 7)
	{
		uint32_t res_id = UINT32_MAX;
		uint32_t c		= prism_query_exec(
				 q,
				 vecs + (size_t)vid * dim,
				 1,
				 1,
				 VS_DISTANCE_MODE_ASYMMETRIC,
				 true,
				 &res_id);
		checked++;
		if (c == 1 && res_id == vid)
			found++;
	}
	prism_query_ctx_destroy(q);
	ASSERT_TRUE(
			(double)found / checked > 0.9,
			"single-probe self-retrieval works -> new centroids route well");

	prism_index_destroy(idx);
}

/*
 * Multi-page centroid overflow: with nlist == the centroid page capacity the
 * one leaf page is full, so a split's extra leaf cannot be appended in place
 * and must chain a second centroid page (the fallback flip path). Exercise
 * that path and confirm the tree grew a page and every entry survived.
 */
/* Fill the single flat centroid page with dummy leaves until it is full,
 * returning the number added. The standalone builder never produces a flat
 * (nlevels == 1) tree whose leaf page is near capacity -- it promotes to a
 * multi-level tree first -- but a long-lived flat index does approach it as
 * repeated splits accumulate leaves on one level. This constructs that state
 * directly so the split's multi-page fallback flip is exercised. */
static uint32_t
fill_centroid_page(PrismIndexBase *base, Dimension dim)
{
	void *rd = vs_alloc0(VS_RABITQ_DATA_SIZE(dim));
	Page  p	 = vs_storage_write_page(
			  base->centroid_storage, base->first_centroid);
	uint32_t added = 0;
	/* child_blkno is an invalid, never-followed leaf: the split only touches
	 * the real leaf it is splitting, so these fillers just consume page room.
	 */
	while (prism_centroid_page_add_entry(
			p, dim, InvalidBlockNumber, 0, PRISM_CENTROID_FLAG_LEAF, rd))
		added++;
	vs_storage_commit_page(base->centroid_storage, base->first_centroid);
	vs_free(rd);
	return added;
}

/*
 * Multi-page centroid overflow: when the flat leaf page has no room for the
 * split's extra leaf, the flip cannot be a single atomic write and must chain
 * a second centroid page (the fallback path). Fill the page, then split, and
 * confirm the tree grew a second page while every entry survived.
 */
TEST(split_overflows_centroid_page)
{
	DECLARE_TWO_BLOB_INDEX(400, 16, 2024);
	ASSERT_NOT_NULL(idx, "build ok");
	ASSERT_EQ(idx->base.nlevels, 1u, "flat tree");
	ASSERT_EQ(centroid_page_count(&idx->base), 1u, "one centroid page");

	uint32_t filled = fill_centroid_page(&idx->base, dim);
	ASSERT_TRUE(filled > 0, "page filled to capacity with dummy leaves");

	/* State going in, so the assertions after the split describe a change
	 * rather than a state the build might have been in all along. */
	uint32_t total_before, fillers_before;
	count_leaf_entries(&idx->base, &total_before, &fillers_before);
	ASSERT_EQ(total_before, filled + 1, "the real leaf, plus the fillers");
	ASSERT_EQ(fillers_before, filled, "and only the fillers are unfollowed");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitResult res;
	int				 rc = prism_posting_split(
			 &idx->base, idx->first_posting, NULL, &env, &res);
	ASSERT_EQ(rc, 0, "split ok");
	ASSERT_TRUE(res.did_split, "split happened");
	ASSERT_EQ(
			res.count[0] + res.count[1],
			n,
			"every entry preserved through the fallback flip");
	ASSERT_TRUE(
			res.count[0] > 0 && res.count[1] > 0, "both new leaves filled");
	ASSERT_EQ(
			centroid_page_count(&idx->base),
			2u,
			"extra leaf chained a second centroid page (fallback path)");

	/* One leaf became two and nothing else moved: had the flip written the
	 * extra leaf over a filler instead of chaining a page, the totals would
	 * be unchanged and a filler would have gone missing. */
	uint32_t total_after, fillers_after;
	count_leaf_entries(&idx->base, &total_after, &fillers_after);
	ASSERT_EQ(total_after, filled + 2, "one leaf became two");
	ASSERT_EQ(fillers_after, filled, "the flip clobbered no filler");

	prism_index_destroy(idx);
}

/*
 * New heads inherit the index's posting format: an AoS build yields AoS heads,
 * a fastscan build yields fastscan heads (the split re-optimizes rather than
 * leaving stale AoS behind). The regress test checks this in PG; assert it
 * directly on the pages here too.
 */
TEST(split_new_heads_keep_format)
{
	uint32_t dim = 16, n = 1200;
	float	*vecs = make_two_blobs(n, dim, 3);

	PrismIndex *aos = build_paged(vecs, n, dim, 1);
	ASSERT_FALSE(
			head_is_fastscan(&aos->base, aos->first_posting),
			"AoS build starts AoS");
	FetchCtx		 fca = {aos->all_vectors};
	PrismSplitEnv	 ea	 = {.fetch_vector = fetch_vec, .ctx = &fca};
	PrismSplitResult ra;
	ASSERT_EQ(
			prism_posting_split(
					&aos->base, aos->first_posting, NULL, &ea, &ra),
			0,
			"aos split ok");
	ASSERT_TRUE(ra.did_split, "aos split happened");
	ASSERT_FALSE(
			head_is_fastscan(&aos->base, ra.head[0]),
			"AoS build -> AoS new head");
	ASSERT_FALSE(
			head_is_fastscan(&aos->base, ra.head[1]),
			"AoS build -> AoS new head");
	prism_index_destroy(aos);

	PrismIndex *fs = build_paged_fastscan(vecs, n, dim, 1);
	ASSERT_TRUE(
			head_is_fastscan(&fs->base, fs->first_posting),
			"fastscan build starts fastscan");
	FetchCtx		 fcf = {fs->all_vectors};
	PrismSplitEnv	 ef	 = {.fetch_vector = fetch_vec, .ctx = &fcf};
	PrismSplitResult rf;
	ASSERT_EQ(
			prism_posting_split(&fs->base, fs->first_posting, NULL, &ef, &rf),
			0,
			"fastscan split ok");
	ASSERT_TRUE(rf.did_split, "fastscan split happened");
	ASSERT_TRUE(
			head_is_fastscan(&fs->base, rf.head[0]),
			"fastscan build -> fastscan new head");
	ASSERT_TRUE(
			head_is_fastscan(&fs->base, rf.head[1]),
			"fastscan build -> fastscan new head");
	prism_index_destroy(fs);
}

/* fetch_vector reporting false (e.g. a dead heap tuple) drops that entry from
 * the split rather than crashing; the survivors are all preserved. */
static bool
fetch_vec_drop_odd(void *ctx, ItemPointerData tid, float *out, Dimension dim)
{
	uint32_t vid = prism_posting_get_vector_id(&tid);
	if (vid & 1u)
		return false; /* simulate an unavailable (dead) tuple */
	return fetch_vec(ctx, tid, out, dim);
}

/*
 * With a target size the split derives its own width and re-checks the size
 * itself: 1200 entries at a target of 100 is 12x the target, so one pass
 * produces round(1200/100) = 12 lists rather than bisecting.
 */
TEST(split_target_derives_width)
{
	DECLARE_TWO_BLOB_INDEX(1200, 16, 31);
	ASSERT_NOT_NULL(idx, "build ok");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitConfig cfg = {.target_entries = 100};
	PrismSplitResult res;
	ASSERT_EQ(
			prism_posting_split(
					&idx->base, idx->first_posting, &cfg, &env, &res),
			0,
			"split ok");
	ASSERT_TRUE(res.did_split, "split happened");
	ASSERT_EQ(res.nparts, 12u, "width is round(count / target)");

	uint32_t total = 0;
	for (uint32_t j = 0; j < res.nparts; j++)
		total += res.count[j];
	ASSERT_EQ(total, n, "entries preserved");

	prism_index_destroy(idx);
}

/*
 * Rounding, not rounding up, is what puts a new list *on* the target. Just
 * over the trigger the ratio is a shade above the factor: rounding up would
 * ask for one partition more than the entries justify and land every one of
 * them below the target, so a fresh list would start beneath the size it is
 * supposed to rest at. Two sizes where the two rules disagree -- 2T+1 (ceil 3,
 * round 2) and 2.4T (ceil 3, round 2).
 */
TEST(split_target_width_rounds_not_ceils)
{
	uint32_t dim = 16, target = 100;

	struct
	{
		uint32_t n;
		uint32_t want_parts;
	} cases[] = {
			{2 * target + 1, 2}, /* 201 -> ceil 3, round 2 */
			{240, 2},			 /* 2.4T -> ceil 3, round 2 */
	};

	for (uint32_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
	{
		float	   *vecs = make_two_blobs(cases[c].n, dim, 21 + c);
		PrismIndex *idx	 = build_paged(vecs, cases[c].n, dim, 1);
		ASSERT_NOT_NULL(idx, "build ok");

		FetchCtx		 fc	 = {idx->all_vectors};
		PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
		PrismSplitConfig cfg = {.target_entries = target};
		PrismSplitResult res;
		ASSERT_EQ(
				prism_posting_split(
						&idx->base, idx->first_posting, &cfg, &env, &res),
				0,
				"split ok");
		ASSERT_TRUE(res.did_split, "above the trigger -> split");
		ASSERT_EQ(
				res.nparts,
				cases[c].want_parts,
				"rounded width, not rounded up");

		prism_index_destroy(idx);
	}
}

/*
 * A list inside the operating band is left alone. 1200 entries against a
 * target of 1000 is above the target but below the trigger (2x), so the split
 * declines rather than halving a list that has room to grow.
 */
TEST(split_target_declines_within_band)
{
	DECLARE_TWO_BLOB_INDEX(1200, 16, 31);
	ASSERT_NOT_NULL(idx, "build ok");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitConfig cfg = {.target_entries = 1000};
	PrismSplitResult res;
	ASSERT_EQ(
			prism_posting_split(
					&idx->base, idx->first_posting, &cfg, &env, &res),
			0,
			"declining is not an error");
	ASSERT_FALSE(res.did_split, "below the trigger -> no split");
	ASSERT_EQ(idx->base.nlist, 1u, "declined -> nlist unchanged");

	prism_index_destroy(idx);
}

/*
 * The target is checked against the *collected* count, not the head's
 * live_count. Here half the entries are unfetchable, so 1200 live_count
 * collects to 600 -- above a target of 400 but below its 800 trigger, so the
 * split must decline even though live_count says it is oversized. This is the
 * LIRE re-check after garbage collection.
 */
TEST(split_target_rechecks_after_collection)
{
	DECLARE_TWO_BLOB_INDEX(1200, 16, 31);
	ASSERT_NOT_NULL(idx, "build ok");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec_drop_odd, .ctx = &fc};
	PrismSplitConfig cfg = {.target_entries = 400};
	PrismSplitResult res;
	ASSERT_EQ(
			prism_posting_split(
					&idx->base, idx->first_posting, &cfg, &env, &res),
			0,
			"declining is not an error");
	ASSERT_FALSE(
			res.did_split, "600 fetchable entries is below the 800 trigger");

	prism_index_destroy(idx);
}

/*
 * One dense blob plus `nout` far-flung outliers, each in its own direction.
 * k-means gives every outlier a partition of its own, which is what the
 * undersized-cluster fold has to clean up.
 */
static float *
make_blob_with_outliers(
		uint32_t nvecs, uint32_t dim, uint32_t nout, uint32_t seed)
{
	srand(seed);
	float *data = vs_alloc((size_t)nvecs * dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
		for (uint32_t j = 0; j < dim; j++)
			data[(size_t)i * dim + j] = (float)(rand() % 2000 - 1000) /
										1000.0f;

	/* Push the last nout vectors far out, along distinct axes. */
	for (uint32_t o = 0; o < nout; o++)
	{
		float *v = data + (size_t)(nvecs - 1 - o) * dim;
		for (uint32_t j = 0; j < dim; j++)
			v[j] = 0.0f;
		v[o % dim] = 500.0f + (float)o * 100.0f;
	}
	return data;
}

/*
 * A partition of one is worse than no partition: it wastes a page, gives a
 * centroid that routes a single vector, and inflates nlist (which the probe
 * count and cost model derive from). So an undersized cluster is folded into
 * its nearest surviving one -- every resulting list clears the floor, and no
 * entries are lost on the way.
 */
TEST(split_folds_undersized_clusters)
{
	uint32_t	dim = 16, n = 1200, nout = 3;
	float	   *vecs = make_blob_with_outliers(n, dim, nout, 7);
	PrismIndex *idx	 = build_paged(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx, "build ok");

	/* Target 100 -> trigger 200 (clearing 1200), floor 50. */
	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitConfig cfg = {.target_entries = 100};
	PrismSplitResult res;
	ASSERT_EQ(
			prism_posting_split(
					&idx->base, idx->first_posting, &cfg, &env, &res),
			0,
			"split ok");
	ASSERT_TRUE(res.did_split, "split happened");
	/* The width asked for round(1200/100) = 12; each outlier came out as its
	 * own cluster and was folded away, so fewer lists are written than were
	 * requested. Without this the floor assertion below could pass on a run
	 * where k-means happened not to isolate anything. */
	ASSERT_TRUE(res.nparts < 12u, "undersized clusters were folded away");

	uint32_t total = 0;
	for (uint32_t j = 0; j < res.nparts; j++)
	{
		ASSERT_TRUE(
				res.count[j] >= 50u,
				"every new list clears the floor (target / factor)");
		total += res.count[j];
	}
	ASSERT_EQ(total, n, "folding loses no entries");

	prism_index_destroy(idx);
}

/*
 * When folding cannot leave two partitions standing, the split widens by one
 * partition rather than declining. A dense region plus a single straggler has
 * no second partition clearing the floor at the bisection width, and the
 * k-means seed is fixed -- so declining here would decline identically on
 * every later pass and leave the list above the trigger forever. Widening
 * splits the bulk and folds the straggler into whichever half is nearer.
 */
TEST(split_widens_when_fold_leaves_one)
{
	uint32_t	dim = 16, n = 600, nout = 1;
	float	   *vecs = make_blob_with_outliers(n, dim, nout, 9);
	PrismIndex *idx	 = build_paged(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx, "build ok");

	/* Target 250 -> trigger 500 (which 600 clears), floor 125, and width
	 * round(600/250) = 2. At two partitions the straggler is its own, so the
	 * fold would leave one -- the split has to widen to get anywhere. */
	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitConfig cfg = {.target_entries = 250};
	PrismSplitResult res;
	ASSERT_EQ(
			prism_posting_split(
					&idx->base, idx->first_posting, &cfg, &env, &res),
			0,
			"split ok");
	ASSERT_TRUE(res.did_split, "widening found a split where two could not");

	uint32_t total = 0;
	for (uint32_t j = 0; j < res.nparts; j++)
	{
		ASSERT_TRUE(res.count[j] >= 125u, "every new list clears the floor");
		total += res.count[j];
	}
	ASSERT_EQ(total, n, "widening loses no entries");

	prism_index_destroy(idx);
}

static bool
fetch_vec_none(void *ctx, ItemPointerData tid, float *out, Dimension dim)
{
	(void)ctx;
	(void)tid;
	(void)out;
	(void)dim;
	return false; /* every tuple is gone */
}

/*
 * Count entries stamped unreachable (+inf estimate, zero error) across an
 * AoS posting chain -- the encoding that keeps a vector with no defined
 * distance out of every query's top-k threshold heap.
 */
static bool
add_entry_count(PrismPostingChainPos *pos, void *state)
{
	*(uint32_t *)state += prism_posting_opaque(pos->page)->entry_count;
	return true;
}

typedef struct UnreachableCtx
{
	Dimension dim;
	uint32_t  n;
} UnreachableCtx;

/* A reassigned-away entry is stamped f_add = inf with a zero error. */
static bool
count_unreachable_page(PrismPostingChainPos *pos, void *state)
{
	UnreachableCtx				 *ctx = state;
	const PrismPostingPageOpaque *op  = prism_posting_opaque(pos->page);
	char *content = prism_posting_page_content(pos->page, ctx->dim);

	for (uint16_t i = 0; i < op->entry_count; i++)
	{
		const PrismPostingEntryHeader *h =
				prism_posting_entry_at(content, i, ctx->dim);

		if (isinf(h->f_add) && h->f_error == 0.0f)
			ctx->n++;
	}
	return true;
}

static uint32_t
chain_entry_count(PrismIndexBase *base, BlockNumber head)
{
	uint32_t n = 0;

	prism_posting_chain_walk(base->posting_storage, head, add_entry_count, &n);
	return n;
}

static uint32_t
count_unreachable(PrismIndexBase *base, BlockNumber head, Dimension dim)
{
	UnreachableCtx ctx = {.dim = dim};

	prism_posting_chain_walk(
			base->posting_storage, head, count_unreachable_page, &ctx);
	return ctx.n;
}

/*
 * A zero-norm vector has no direction, so under cosine its distance to
 * anything is undefined. The build and insert paths encode such rows
 * unreachable, because a naive encoding looks mid-range from every query and
 * enters the top-k threshold heap as an impostor, tightening the pruning
 * bound against genuine neighbours. A split re-encodes every entry it moves,
 * so it has to carry the same stamp -- otherwise routine maintenance silently
 * undoes the invariant for the lists it touches.
 */
TEST(split_keeps_zero_vectors_unreachable)
{
	uint32_t dim = 8, nreal = 400, nzero = 40, n = nreal + nzero;
	float	*vecs = vs_alloc((size_t)n * dim * sizeof(float));

	/* Distinct directions, so cosine distances are well separated. */
	srand(5);
	for (uint32_t i = 0; i < nreal; i++)
		for (uint32_t j = 0; j < dim; j++)
			vecs[(size_t)i * dim + j] = 1.0f +
										(float)(rand() % 1000) / 1000.0f;
	/* Zero vectors in the tail. */
	for (uint32_t i = nreal; i < n; i++)
		for (uint32_t j = 0; j < dim; j++)
			vecs[(size_t)i * dim + j] = 0.0f;

	PrismIndex *idx = build_paged_cosine(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx, "cosine build ok");
	/* The build's own stamp, so the count after the split is preservation
	 * and not the split having arrived at it by itself. */
	ASSERT_EQ(
			count_unreachable(&idx->base, idx->first_posting, dim),
			nzero,
			"build stamped the zero vectors unreachable");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitConfig cfg = {.target_entries = 100};
	PrismSplitResult res;
	ASSERT_EQ(
			prism_posting_split(
					&idx->base, idx->first_posting, &cfg, &env, &res),
			0,
			"split ok");
	ASSERT_TRUE(res.did_split, "split happened");

	uint32_t total = 0, unreachable = 0;
	for (uint32_t j = 0; j < res.nparts; j++)
	{
		total += res.count[j];
		unreachable += count_unreachable(&idx->base, res.head[j], dim);
	}
	ASSERT_EQ(total, n, "every entry preserved, zero vectors included");
	ASSERT_EQ(
			unreachable,
			nzero,
			"every zero vector is still stamped unreachable after the split");

	prism_index_destroy(idx);
}

/*
 * A split must not hold the list it is splitting. It streams the chain twice
 * instead -- once to count it and draw a bounded sample to cluster on, once to
 * write each entry to the list whose centroid is nearest -- so the memory it
 * takes follows the sample ceiling, not the list's size. That is what lets a
 * list that has grown far past the trigger still be splittable, which is
 * exactly when a split has to work.
 *
 * Here the ceiling admits a third of the list, so every decision is made from
 * a sample: the width still comes from the counted entries, every entry still
 * reaches a list, and the partition is still good enough to find a vector by
 * its own value.
 */
TEST(split_samples_within_a_budget)
{
	uint32_t dim = 16, n = 1200;
	/* A budget that pays for about 400 sampled points -- a third of the
	 * list -- so every decision below is made from a sample. */
	uint64_t budget = prism_split_fixed_bytes(dim) +
					  400ull * (dim * sizeof(float) +
								PRISM_SPLIT_SAMPLE_POINT_OVERHEAD);
	float	   *vecs = make_two_blobs(n, dim, 77);
	PrismIndex *idx	 = build_paged(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx, "build ok");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitConfig cfg = {
			.target_entries		 = 100,
			.sample_budget_bytes = budget,
	};
	PrismSplitResult res;
	ASSERT_EQ(
			prism_posting_split(
					&idx->base, idx->first_posting, &cfg, &env, &res),
			0,
			"split ok");
	ASSERT_TRUE(res.did_split, "split happened");
	/*
	 * The width is round(counted / target) = 12, of which a couple of parts
	 * can still be dropped for looking too small in the sample. Had the width
	 * been taken from the sample instead it would be round(400/100) = 4, so
	 * anything above that proves the counted entries drove it -- an exact
	 * count would only be asserting which clusters the sample happened to
	 * make look small.
	 */
	ASSERT_TRUE(
			res.nparts > 4u,
			"width comes from the counted entries, not the sample");
	ASSERT_TRUE(res.nparts <= 12u, "and no wider than the width asked for");

	uint32_t total = 0;
	for (uint32_t j = 0; j < res.nparts; j++)
		total += res.count[j];
	ASSERT_EQ(total, n, "every entry reached a list, sample or not");

	/* The lists are still usable: probing every leaf, each vector is found by
	 * querying its own value -- so the streaming pass put entries where their
	 * centroid says they are. */
	PrismQueryCtx *q = prism_query_ctx_create(idx, 1, PRISM_SPLIT_MAX_PARTS);
	uint32_t	   checked = 0, found = 0;
	for (uint32_t vid = 0; vid < n; vid += 13)
	{
		uint32_t out[1] = {0};
		uint32_t c		= prism_query_exec(
				 q,
				 vecs + (size_t)vid * dim,
				 1,
				 PRISM_SPLIT_MAX_PARTS,
				 VS_DISTANCE_MODE_ASYMMETRIC,
				 true,
				 out);
		checked++;
		if (c > 0 && out[0] == vid)
			found++;
	}
	prism_query_ctx_destroy(q);
	ASSERT_TRUE(
			found * 10 >= checked * 9,
			"a sampled split still routes vectors to their own list");

	prism_index_destroy(idx);
}

/*
 * Bytes a split allocates for a list of `n` vectors under `budget`, measured
 * by running it inside a context of its own.
 */
static size_t
measure_split_bytes(uint32_t n, uint32_t dim, uint64_t budget, uint32_t target)
{
	float	   *vecs = make_two_blobs(n, dim, 5);
	PrismIndex *idx	 = build_paged(vecs, n, dim, 1);
	FetchCtx	fc	 = {idx->all_vectors};

	PrismSplitEnv	 env = {.fetch_vector = fetch_vec, .ctx = &fc};
	PrismSplitConfig cfg = {
			.target_entries		 = target,
			.sample_budget_bytes = budget,
	};
	PrismSplitResult res;

	VsMemCtx ctx = vs_memctx_create(vs_memctx_current(), "split-measure");
	VsMemCtx old = vs_memctx_switch(ctx);
	int		 rc	 = prism_posting_split(
			  &idx->base, idx->first_posting, &cfg, &env, &res);
	vs_memctx_switch(old);

	size_t bytes = vs_memctx_total_allocated(ctx);
	vs_memctx_delete(ctx);
	prism_index_destroy(idx);

	return (rc == 0 && res.did_split) ? bytes : 0;
}

/*
 * The bound has to be on bytes, not just on behaviour: a change that put the
 * whole list back in memory would still pass every functional test above. So
 * measure it. Ten times the list under the same budget must not cost ten times
 * the memory -- if it does, something is again holding the list rather than
 * streaming it.
 */
TEST(split_memory_is_bounded_by_budget)
{
	uint32_t dim   = 64;
	uint32_t small = 1000;
	uint32_t big   = 10000;
	/* Enough for about 500 sampled points, so both lists have to sample. */
	uint64_t budget = prism_split_fixed_bytes(dim) +
					  500ull * (dim * sizeof(float) +
								PRISM_SPLIT_SAMPLE_POINT_OVERHEAD);

	size_t small_bytes = measure_split_bytes(small, dim, budget, 100);
	size_t big_bytes   = measure_split_bytes(big, dim, budget, 100);

	ASSERT_TRUE(small_bytes > 0, "the small list split");
	ASSERT_TRUE(big_bytes > 0, "the big list split");

	/*
	 * Ten times the entries, and the vectors alone would be 2.4 MB more if
	 * they were held. Allow generous headroom for what does grow with the
	 * list -- the pages the new lists are written into -- while still failing
	 * an allocation that follows the list's vectors.
	 */
	size_t held = (size_t)big * dim * sizeof(float);
	ASSERT_TRUE(
			big_bytes < small_bytes + held / 2,
			"a ten-times longer list does not cost the list's vectors");

	vs_free(NULL);
}

/*
 * A list whose entries have all gone from the heap. The size the head records
 * says the list is far over the trigger, but nothing can be fetched, so there
 * is nothing to cluster and nothing to write -- the split has to decline and
 * leave the list alone rather than build empty lists or divide by zero. The
 * entries stay where they are for VACUUM to remove.
 */
TEST(split_declines_when_nothing_is_fetchable)
{
	DECLARE_TWO_BLOB_INDEX(1200, 16, 3);
	ASSERT_NOT_NULL(idx, "build ok");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec_none, .ctx = &fc};
	PrismSplitConfig cfg = {.target_entries = 100};
	PrismSplitResult res;

	ASSERT_EQ(
			prism_posting_split(
					&idx->base, idx->first_posting, &cfg, &env, &res),
			0,
			"declining is not an error");
	ASSERT_FALSE(res.did_split, "nothing fetchable -> no split");
	ASSERT_EQ(res.nparts, 0u, "and no lists were created");
	ASSERT_EQ(idx->base.nlist, 1u, "declined -> nlist unchanged");

	prism_index_destroy(idx);
}

/*
 * Target list size: the flat constant above PRISM_TARGET_ENTRIES_PER_LIST^2
 * rows, and prism_auto_nlist's sqrt floor below it (where a hardcoded constant
 * would fight the build). An explicit nlist wins over both.
 */
TEST(target_entries_per_list_regimes)
{
	uint32_t sq = PRISM_TARGET_ENTRIES_PER_LIST *
				  PRISM_TARGET_ENTRIES_PER_LIST;

	ASSERT_EQ(
			prism_target_entries_per_list((double)sq, 0),
			(uint32_t)PRISM_TARGET_ENTRIES_PER_LIST,
			"at the crossover the two regimes agree");
	ASSERT_EQ(
			prism_target_entries_per_list(1000000.0, 0),
			(uint32_t)PRISM_TARGET_ENTRIES_PER_LIST,
			"above the crossover it is the flat target");
	ASSERT_EQ(
			prism_target_entries_per_list(100000000.0, 0),
			(uint32_t)PRISM_TARGET_ENTRIES_PER_LIST,
			"and stays flat as the dataset grows");

	/* Below the crossover the sqrt floor gives ~sqrt(count) per list. */
	ASSERT_EQ(
			prism_target_entries_per_list(10000.0, 0),
			100u,
			"below the crossover it tracks sqrt(count)");
	ASSERT_TRUE(
			prism_target_entries_per_list(10000.0, 0) <
					(uint32_t)PRISM_TARGET_ENTRIES_PER_LIST,
			"small tables target smaller lists, not the constant");

	/* An explicit nlist is honoured, so maintenance does not override it. */
	ASSERT_EQ(
			prism_target_entries_per_list(10000.0, 20),
			500u,
			"explicit nlist wins");

	/* Degenerate inputs stay in range. */
	ASSERT_EQ(prism_target_entries_per_list(0.0, 0), 1u, "empty -> 1");
	ASSERT_EQ(prism_target_entries_per_list(1.0, 0), 1u, "single row -> 1");
}

TEST(split_drops_unfetchable_vectors)
{
	DECLARE_TWO_BLOB_INDEX(1200, 16, 17);
	ASSERT_NOT_NULL(idx, "build ok");
	/* All of them are indexed to start with, so the halving below is the
	 * split dropping what it cannot fetch, not a short build. */
	ASSERT_EQ(
			chain_entry_count(&idx->base, idx->first_posting),
			n,
			"every entry indexed before the split");

	FetchCtx		 fc	 = {idx->all_vectors};
	PrismSplitEnv	 env = {.fetch_vector = fetch_vec_drop_odd, .ctx = &fc};
	PrismSplitResult res;
	int				 rc = prism_posting_split(
			 &idx->base, idx->first_posting, NULL, &env, &res);

	ASSERT_EQ(rc, 0, "split tolerates unfetchable entries");
	/* Half the entries (n/2 = 600) remain fetchable, far above the split
	 * threshold, so the split must still happen -- assert it did, otherwise
	 * the drop path is never exercised. */
	ASSERT_TRUE(res.did_split, "split still happens with the fetchable half");
	ASSERT_EQ(
			res.count[0] + res.count[1],
			n / 2,
			"only the even-id (fetchable) half is kept");

	prism_index_destroy(idx);
}
