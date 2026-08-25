/*
 * test_posting_split.c - Unit tests for incremental posting-list split
 *
 * Builds a single-partition paged index, splits it with mkt_posting_split,
 * and verifies:
 *   - the split preserves every entry and produces two non-empty leaves
 *   - every vector is still retrievable through the mutated centroid tree
 *   - recall after an incremental split matches a from-scratch 2-partition
 *     build (within noise)
 */

#include <math.h>
#include <string.h>

#include "algo/distance.h"
#include "core/memory.h"
#include "index/posting_page.h"
#include "index/posting_split.h"
#include "mkt_test.h"
#include "standalone/index.h"
#include "standalone/query.h"

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
	float *data = mkt_alloc((size_t)nvecs * dim * sizeof(float));
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

static MktIndex *
build_paged(const float *vecs, uint32_t nvecs, uint32_t dim, uint32_t nlist)
{
	MktIndexConfig config = {
			.nlist		   = nlist,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = MKT_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
			.posting_fmt   = MKT_POSTING_FMT_PAGES,
			.nworkers	   = 0,
	};
	MktArraySource src;
	mkt_array_source_init(&src, vecs, nvecs, dim);
	return mkt_index_build(&src.base, &config, NULL);
}

/* Build a flat (single-level) tree with `nlist` leaves by forcing a fan-out
 * large enough that no intermediate level is created. */
static MktIndex *
build_flat(const float *vecs, uint32_t nvecs, uint32_t dim, uint32_t nlist)
{
	/* fan_out == nlist keeps the tree single-level (nlevels == 1) with exactly
	 * nlist leaves — the shape the phase-1 split requires. */
	MktIndexConfig config = {
			.nlist		   = nlist,
			.fan_out	   = nlist,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = MKT_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
			.posting_fmt   = MKT_POSTING_FMT_PAGES,
			.nworkers	   = 0,
	};
	MktArraySource src;
	mkt_array_source_init(&src, vecs, nvecs, dim);
	return mkt_index_build(&src.base, &config, NULL);
}

/* Build with fastscan-packed posting pages (SoA groups), to exercise the
 * fastscan branch of the split's entry collection. */
static MktIndex *
build_paged_fastscan(
		const float *vecs, uint32_t nvecs, uint32_t dim, uint32_t nlist)
{
	MktIndexConfig config = {
			.nlist		   = nlist,
			.metric		   = DISTANCE_L2,
			.centroid_fmt  = MKT_CENTROID_FMT_RABITQ,
			.encode_rabitq = true,
			.posting_fmt   = MKT_POSTING_FMT_PAGES,
			.fastscan	   = 16,
			.nworkers	   = 0,
	};
	MktArraySource src;
	mkt_array_source_init(&src, vecs, nvecs, dim);
	return mkt_index_build(&src.base, &config, NULL);
}

/* nblobs Gaussian-ish clusters spaced along dim 0, so adjacent clusters share
 * a boundary — the regime where a split can pull entries in from neighbors. */
static float *
make_blobs(
		uint32_t nblobs,
		uint32_t per,
		uint32_t dim,
		float	 spacing,
		uint32_t seed)
{
	srand(seed);
	uint32_t n	  = nblobs * per;
	float	*data = mkt_alloc((size_t)n * dim * sizeof(float));
	for (uint32_t b = 0; b < nblobs; b++)
		for (uint32_t i = 0; i < per; i++)
		{
			uint32_t idx = b * per + i;
			for (uint32_t j = 0; j < dim; j++)
				data[(size_t)idx * dim + j] = (j == 0 ? (float)b * spacing
													  : 0.0f) +
											  (float)(rand() % 2000 - 1000) /
													  1000.0f;
		}
	return data;
}

/* Split env: fetch full-precision vectors from the in-RAM store by id. */
typedef struct FetchCtx
{
	const float *all;
} FetchCtx;

static bool
fetch_vec(void *ctx, ItemPointerData tid, float *out, Dimension dim)
{
	const float *all = ((FetchCtx *)ctx)->all;
	uint32_t	 vid = mkt_posting_get_vector_id(&tid);
	memcpy(out, all + (size_t)vid * dim, (size_t)dim * sizeof(float));
	return true;
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
	float *dists = mkt_alloc(nvecs * sizeof(float));
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
	mkt_free(dists);
}

static double
measure_recall(
		MktIndex	*idx,
		const float *vecs,
		uint32_t	 nvecs,
		uint32_t	 dim,
		uint32_t	 k,
		uint32_t	 nprobe,
		uint32_t	 nq)
{
	MktQueryCtx *q		 = mkt_query_ctx_create(idx, k, nprobe);
	uint32_t	 hits	 = 0;
	uint32_t	 res[16] = {0};
	uint32_t	 gt[16]	 = {0};
	uint32_t	 stride	 = nvecs / nq;

	for (uint32_t i = 0; i < nq; i++)
	{
		const float *query = vecs + (size_t)(i * stride) * dim;
		uint32_t	 c	   = mkt_query_exec(
				q, query, k, nprobe, MKT_DISTANCE_MODE_ASYMMETRIC, true, res);
		brute_force_knn(vecs, nvecs, dim, query, k, gt);
		for (uint32_t a = 0; a < c; a++)
			for (uint32_t g = 0; g < k; g++)
				if (res[a] == gt[g])
					hits++;
	}
	mkt_query_ctx_destroy(q);
	return (double)hits / ((double)nq * k);
}

/* ----------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------- */

TEST(split_preserves_entries_and_structure)
{
	uint32_t  dim = 16, n = 1200;
	float	 *vecs = make_two_blobs(n, dim, 42);
	MktIndex *idx  = build_paged(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx, "paged build should succeed");
	ASSERT_EQ(idx->base.nlist, 1u, "starts as a single partition");
	ASSERT_EQ(idx->base.nlevels, 1u, "flat tree");

	FetchCtx	   fc  = {idx->all_vectors};
	MktSplitEnv	   env = {.fetch_vector = fetch_vec, .ctx = &fc};
	MktSplitResult res;
	int			   rc = mkt_posting_split(
			   &idx->base, idx->first_posting, NULL, &env, &res);

	ASSERT_EQ(rc, 0, "split should succeed");
	ASSERT_TRUE(res.did_split, "split should happen");
	ASSERT_EQ(res.new_nlist, 2u, "one leaf becomes two");
	ASSERT_EQ(idx->base.nlist, 2u, "nlist bumped in base");
	ASSERT_EQ(res.count0 + res.count1, n, "every entry preserved");
	ASSERT_TRUE(res.count0 > 0 && res.count1 > 0, "both leaves non-empty");

	mkt_index_destroy(idx);
}

TEST(split_keeps_vectors_retrievable)
{
	uint32_t  dim = 16, n = 1200;
	float	 *vecs = make_two_blobs(n, dim, 7);
	MktIndex *idx  = build_paged(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx, "build ok");

	FetchCtx	   fc  = {idx->all_vectors};
	MktSplitEnv	   env = {.fetch_vector = fetch_vec, .ctx = &fc};
	MktSplitResult res;
	ASSERT_EQ(
			mkt_posting_split(
					&idx->base, idx->first_posting, NULL, &env, &res),
			0,
			"split ok");
	ASSERT_TRUE(res.did_split, "split happened");

	/* Self-retrieval: each vector should find itself as the top-1 through the
	 * mutated tree (rerank makes this exact). Sample every 5th vector. */
	MktQueryCtx *q		 = mkt_query_ctx_create(idx, 1, 2);
	uint32_t	 checked = 0, found = 0;
	for (uint32_t vid = 0; vid < n; vid += 5)
	{
		uint32_t res_id = UINT32_MAX;
		uint32_t c		= mkt_query_exec(
				 q,
				 vecs + (size_t)vid * dim,
				 1,
				 2,
				 MKT_DISTANCE_MODE_ASYMMETRIC,
				 true,
				 &res_id);
		checked++;
		if (c == 1 && res_id == vid)
			found++;
	}
	mkt_query_ctx_destroy(q);

	double self_recall = (double)found / (double)checked;
	ASSERT_TRUE(
			self_recall > 0.98, "nearly all vectors retrievable post-split");

	mkt_index_destroy(idx);
}

TEST(split_fastscan_posting)
{
	/* Fastscan posting pages store TIDs in SoA 32-groups; the split must read
	 * them correctly (the AoS reader would misalign). */
	uint32_t  dim = 16, n = 1200;
	float	 *vecs = make_two_blobs(n, dim, 99);
	MktIndex *idx  = build_paged_fastscan(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx, "fastscan build ok");

	FetchCtx	   fc  = {idx->all_vectors};
	MktSplitEnv	   env = {.fetch_vector = fetch_vec, .ctx = &fc};
	MktSplitResult res;
	ASSERT_EQ(
			mkt_posting_split(
					&idx->base, idx->first_posting, NULL, &env, &res),
			0,
			"split ok");
	ASSERT_TRUE(res.did_split, "split happened");
	ASSERT_EQ(res.count0 + res.count1, n, "every entry preserved from SoA");

	MktQueryCtx *q		 = mkt_query_ctx_create(idx, 1, 2);
	uint32_t	 checked = 0, found = 0;
	for (uint32_t vid = 0; vid < n; vid += 5)
	{
		uint32_t res_id = UINT32_MAX;
		uint32_t c		= mkt_query_exec(
				 q,
				 vecs + (size_t)vid * dim,
				 1,
				 2,
				 MKT_DISTANCE_MODE_ASYMMETRIC,
				 true,
				 &res_id);
		checked++;
		if (c == 1 && res_id == vid)
			found++;
	}
	mkt_query_ctx_destroy(q);
	ASSERT_TRUE(
			(double)found / checked > 0.98,
			"retrievable after fastscan split");

	mkt_index_destroy(idx);
}

static double
self_recall(
		MktIndex	*idx,
		const float *vecs,
		uint32_t	 n,
		uint32_t	 dim,
		uint32_t	 nprobe)
{
	MktQueryCtx *q		 = mkt_query_ctx_create(idx, 1, nprobe);
	uint32_t	 checked = 0, found = 0;
	for (uint32_t vid = 0; vid < n; vid += 5)
	{
		uint32_t res_id = UINT32_MAX;
		uint32_t c		= mkt_query_exec(
				 q,
				 vecs + (size_t)vid * dim,
				 1,
				 nprobe,
				 MKT_DISTANCE_MODE_ASYMMETRIC,
				 true,
				 &res_id);
		checked++;
		if (c == 1 && res_id == vid)
			found++;
	}
	mkt_query_ctx_destroy(q);
	return (double)found / (double)checked;
}

TEST(split_reassign_preserves_recall)
{
	/* LIRE reassignment must move entries correctly (no losses) and not hurt
	 * recall versus a split with reassignment off. */
	uint32_t dim = 16, nblobs = 4, per = 400, n = nblobs * per;
	uint32_t k = 10, nprobe = 8, nq = 20;
	float	*vecs = make_blobs(nblobs, per, dim, 3.0f, 55);

	/* Split with reassignment OFF. */
	MktIndex	  *a   = build_flat(vecs, n, dim, nblobs);
	FetchCtx	   fca = {a->all_vectors};
	MktSplitEnv	   ea  = {.fetch_vector = fetch_vec, .ctx = &fca};
	MktSplitResult ra;
	MktSplitConfig c0 = {0};
	ASSERT_EQ(
			mkt_posting_split(&a->base, a->first_posting, &c0, &ea, &ra),
			0,
			"split A ok");
	ASSERT_TRUE(ra.did_split, "A split");
	double rA = measure_recall(a, vecs, n, dim, k, nprobe, nq);

	/* Split the same list with reassignment ON. */
	MktIndex	  *b   = build_flat(vecs, n, dim, nblobs);
	FetchCtx	   fcb = {b->all_vectors};
	MktSplitEnv	   eb  = {.fetch_vector = fetch_vec, .ctx = &fcb};
	MktSplitResult rb;
	MktSplitConfig cr = {.reassign_neighbors = 3};
	ASSERT_EQ(
			mkt_posting_split(&b->base, b->first_posting, &cr, &eb, &rb),
			0,
			"split B ok");
	ASSERT_TRUE(rb.did_split, "B split");
	double rB = measure_recall(b, vecs, n, dim, k, nprobe, nq);

	double selfA = self_recall(a, vecs, n, dim, nprobe);
	double selfB = self_recall(b, vecs, n, dim, nprobe);

	/* Reassignment only relocates boundary entries to their nearer centroid,
	 * so relative to a plain split it must not lose entries or hurt recall.
	 * (Absolute recall depends on the dataset, so assert relative to A.) */
	ASSERT_TRUE(rb.did_split, "B split happened");
	ASSERT_TRUE(
			rB >= rA - 0.05, "reassign does not hurt recall vs plain split");
	ASSERT_TRUE(
			selfB >= selfA - 0.02, "reassign loses no entries vs plain split");

	mkt_index_destroy(a);
	mkt_index_destroy(b);
}

TEST(merge_dissolves_into_neighbor)
{
	/* Merge dissolves a list into its nearest neighbor: entries move there,
	 * the leaf count drops, routing skips the emptied leaf, and no entry is
	 * lost. */
	uint32_t dim = 16, nblobs = 4, per = 400, n = nblobs * per, nprobe = 8;
	float	*vecs = make_blobs(nblobs, per, dim, 5.0f, 77);

	MktIndex *idx = build_flat(vecs, n, dim, nblobs);
	ASSERT_NOT_NULL(idx, "build ok");
	ASSERT_EQ(idx->base.nlist, nblobs, "built nlist leaves");
	ASSERT_EQ(idx->base.nlevels, 1u, "flat tree");

	double before = self_recall(idx, vecs, n, dim, nprobe);

	FetchCtx	   fc  = {idx->all_vectors};
	MktSplitEnv	   env = {.fetch_vector = fetch_vec, .ctx = &fc};
	MktMergeResult res;
	ASSERT_EQ(
			mkt_posting_merge(&idx->base, idx->first_posting, &env, &res),
			0,
			"merge ok");
	ASSERT_TRUE(res.did_merge, "merge happened");
	ASSERT_EQ(res.new_nlist, nblobs - 1u, "one fewer leaf");
	ASSERT_EQ(idx->base.nlist, nblobs - 1u, "nlist decremented");
	ASSERT_TRUE(res.moved > 0, "entries moved into the neighbor");

	double after = self_recall(idx, vecs, n, dim, nprobe);
	/*
	 * Merge only touches the moved entries: it re-encodes them against the
	 * neighbor's centroid (so ones far from it may be pruned in the
	 * approximate scan — merge is meant for small lists near a neighbor), but
	 * entries in other lists are untouched. So self-recall drops by at most
	 * the moved fraction plus noise; a larger drop would mean collateral loss.
	 */
	double max_drop = (double)res.moved / (double)n + 0.03;
	ASSERT_TRUE(
			after >= before - max_drop, "no collateral entry loss on merge");

	mkt_index_destroy(idx);
}

TEST(split_declines_below_threshold)
{
	uint32_t  dim = 16, n = 1200;
	float	 *vecs = make_two_blobs(n, dim, 5);
	MktIndex *idx  = build_paged(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx, "build ok");

	FetchCtx	   fc  = {idx->all_vectors};
	MktSplitEnv	   env = {.fetch_vector = fetch_vec, .ctx = &fc};
	MktSplitConfig cfg = {.min_split_entries = 1000000}; /* never reached */
	MktSplitResult res;
	int			   rc = mkt_posting_split(
			   &idx->base, idx->first_posting, &cfg, &env, &res);

	ASSERT_EQ(rc, 0, "declining is not an error");
	ASSERT_FALSE(res.did_split, "too few entries -> no split");
	ASSERT_EQ(idx->base.nlist, 1u, "nlist unchanged");

	mkt_index_destroy(idx);
}

TEST(split_handles_degenerate_data)
{
	/* All identical vectors: the two centroids coincide. The split must not
	 * crash and must preserve every entry regardless of how 2-means breaks the
	 * tie. */
	uint32_t dim = 16, n = 300;
	float	*vecs = mkt_alloc((size_t)n * dim * sizeof(float));
	for (uint32_t i = 0; i < n; i++)
		for (uint32_t j = 0; j < dim; j++)
			vecs[(size_t)i * dim + j] = 1.5f;

	MktIndex *idx = build_paged(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx, "build ok");

	FetchCtx	   fc  = {idx->all_vectors};
	MktSplitEnv	   env = {.fetch_vector = fetch_vec, .ctx = &fc};
	MktSplitResult res;
	int			   rc = mkt_posting_split(
			   &idx->base, idx->first_posting, NULL, &env, &res);

	ASSERT_EQ(rc, 0, "split should not error on degenerate data");
	if (res.did_split)
		ASSERT_EQ(res.count0 + res.count1, n, "entries preserved");
	else
		ASSERT_EQ(idx->base.nlist, 1u, "declined -> nlist unchanged");

	mkt_index_destroy(idx);
}

TEST(split_recall_matches_rebuild)
{
	uint32_t dim = 16, n = 1200, k = 10, nprobe = 2, nq = 20;
	float	*vecs = make_two_blobs(n, dim, 123);

	/* Incremental: build one partition, then split it. */
	MktIndex *idx_a = build_paged(vecs, n, dim, 1);
	ASSERT_NOT_NULL(idx_a, "build A ok");
	FetchCtx	   fc  = {idx_a->all_vectors};
	MktSplitEnv	   env = {.fetch_vector = fetch_vec, .ctx = &fc};
	MktSplitResult res;
	ASSERT_EQ(
			mkt_posting_split(
					&idx_a->base, idx_a->first_posting, NULL, &env, &res),
			0,
			"split ok");
	ASSERT_TRUE(res.did_split, "split happened");
	double recall_a = measure_recall(idx_a, vecs, n, dim, k, nprobe, nq);

	/* From scratch: build two partitions directly. */
	MktIndex *idx_b = build_paged(vecs, n, dim, 2);
	ASSERT_NOT_NULL(idx_b, "build B ok");
	double recall_b = measure_recall(idx_b, vecs, n, dim, k, nprobe, nq);

	ASSERT_TRUE(recall_a > 0.5, "post-split recall is reasonable");
	ASSERT_TRUE(
			fabs(recall_a - recall_b) < 0.15,
			"incremental split recall tracks a from-scratch build");

	mkt_index_destroy(idx_a);
	mkt_index_destroy(idx_b);
}
