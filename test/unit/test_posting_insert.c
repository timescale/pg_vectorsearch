/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_posting_insert.c - Unit tests for runtime insert primitives
 *
 * Exercises src/index/posting_insert.c against the array-backed test storage:
 * - insert appends to the chain and maintains head live_count / tail_blkno
 * - inserted entries are found by the cluster scan
 * - insert grows a new overflow page when the tail is full
 * - inserts into a FASTSCAN cluster append AoS pages that the fastscan scan
 *   still merges (mixed-format chain)
 */

#include <math.h>
#include <string.h>

#include "core/memory.h"
#include "core/platform.h"
#include "index/posting_build.h"
#include "index/posting_insert.h"
#include "index/posting_page.h"
#include "index/posting_scan.h"
#include "index/storage.h"
#include "posting_fixtures.h"
#include "quant/rabitq.h"
#include "standalone/pg_compat.h"
#include "vs_test.h"

TEST_GROUP(PostingInsert);
TEST_MEMCTX_FIXTURE();

/* Rotate + append one vector via the primitive under test. */
static void
insert_vec(
		TestPageStorage *st,
		RaBitQParams	*params,
		Dimension		 dim,
		BlockNumber		 head,
		uint32_t		 vid,
		const float		*vec,
		RaBitQScratch	*scratch)
{
	float *pt = vs_alloc_aligned((size_t)dim * sizeof(float), 64);
	vs_rabitq_rotate(params, vec, pt);
	prism_posting_insert_one(
			&st->base,
			params,
			dim,
			head,
			vid_to_tid(vid),
			pt,
			scratch,
			false,
			NULL);
}

/* Scan a cluster with a generous top-k; return how many entries land in it. */
static uint32_t
scan_count(
		TestPageStorage *st,
		RaBitQParams	*params,
		Dimension		 dim,
		const float		*centroid,
		BlockNumber		 head,
		uint32_t		 k,
		bool			 fastscan)
{
	RaBitQQueryState qstate;
	setup_query_state(&qstate, params, centroid, dim);

	PrismPostingScan scan;
	prism_posting_scan_init(
			&scan,
			&st->base,
			NULL,
			params,
			dim,
			prism_posting_max_entries(dim));
	if (fastscan)
		prism_posting_scan_enable_fastscan(&scan, 16);
	VsTopK topk;
	vs_topk_init(&topk, k);

	prism_posting_scan_begin_cluster(&scan, &qstate, head);
	if (fastscan)
		prism_posting_scan_cluster_fastscan(&scan, &topk);
	else
		prism_posting_scan_cluster(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);

	uint32_t n = topk.cand_count;
	vs_topk_cleanup(&topk);
	prism_posting_scan_cleanup(&scan);
	return n;
}

/*
 * Run a fastscan scan against one cluster and report whether a specific
 * vector shows up anywhere in the top k results.
 *
 * "vid" is short for vector id. These unit tests have no real PostgreSQL
 * heap, so each indexed vector is identified by a plain integer instead
 * of a heap TID, and that integer is what gets packed into the on-disk
 * TID field in its place (see vid_to_tid and prism_posting_get_vector_id).
 *
 * This checks membership -- is this vid anywhere in the results? -- not
 * ranking -- how high did it score? That distinction is the point: a
 * deleted-and-marked ("tombstoned") entry must be absent no matter how
 * good a match it would otherwise look like, so a test for that has to
 * check whether the id shows up at all, not just where it lands.
 */
static bool
scan_contains_vid(
		TestPageStorage	 *st,
		RaBitQParams	 *params,
		Dimension		  dim,
		RaBitQQueryState *qstate,
		BlockNumber		  head,
		uint32_t		  k,
		uint32_t		  vid)
{
	PrismPostingScan scan;
	prism_posting_scan_init(
			&scan,
			&st->base,
			NULL,
			params,
			dim,
			prism_posting_max_entries(dim));
	prism_posting_scan_enable_fastscan(&scan, 16);
	VsTopK topk;
	vs_topk_init(&topk, k);

	prism_posting_scan_begin_cluster(&scan, qstate, head);
	prism_posting_scan_cluster_fastscan(&scan, &topk);
	prism_posting_scan_end_cluster(&scan);

	VsTopKEntry *entries = vs_alloc(topk.cand_count * sizeof(VsTopKEntry));
	uint32_t	 n;
	vs_topk_extract_sorted(&topk, entries, &n);

	bool found = false;
	for (uint32_t i = 0; i < n; i++)
		if (prism_posting_decode_vector_id(entries[i].id) == vid)
		{
			found = true;
			break;
		}

	vs_free(entries);
	vs_topk_cleanup(&topk);
	prism_posting_scan_cleanup(&scan);
	return found;
}

/*
 * Print whether this test run actually exercised the AVX-512 kernel it is
 * meant to test.
 *
 * The fastscan format scores 32 vectors at once using AVX-512
 * instructions, when the CPU running the test actually has AVX-512. On a
 * machine without it, the same code quietly falls back to an ordinary
 * scalar loop that checks one vector at a time instead. Both give the
 * same answer, so a test that only checks pass/fail can't tell which one
 * it went through. That matters here because these specific tests exist
 * to catch a bug in the AVX-512 version: on a machine without AVX-512,
 * they would keep passing for an unrelated reason (the scalar fallback
 * works fine) while never touching the code they are supposed to be
 * checking. Printing which path ran turns that silent gap into a visible
 * one in the test output.
 */
static void
log_fastscan_avx512_coverage(void)
{
	TEST_PRINT(
			"AVX-512: %s\n",
			vs_has_all_simd(VS_SIMD_AVX512_BW)
					? "available -- masked-compare path exercised"
					: "NOT available -- falling back to the scalar skip "
					  "path; this run does not cover fastscan_prune_"
					  "group_avx512");
}

/*
 * Independently recount how many entries in a cluster's chain are still
 * live (not deleted), by walking every page from scratch. This checks the
 * head page's own running live_count against a number computed a
 * completely different way -- the real insert/delete code never walks
 * the chain like this, so a bug there wouldn't also be baked into this
 * count.
 *
 * The two posting-page formats record a dead entry differently. An AoS
 * page flags each entry individually. A fastscan page instead packs 32
 * vectors together into one "group" so a single CPU instruction can
 * score all 32 at once; within a group, "lane" just means "this vector's
 * slot among the 32," and a dead vector is recorded as one set bit in a
 * 32-bit tombstone_mask for its group -- one bit per lane -- rather than
 * a flag on the entry itself.
 */
typedef struct LiveCtx
{
	Dimension dim;
	uint32_t  live;
} LiveCtx;

static bool
count_live_page(PrismPostingChainPos *pos, void *state)
{
	LiveCtx						 *ctx = state;
	const PrismPostingPageOpaque *op  = prism_posting_opaque(pos->page);
	uint32_t					  n	  = op->entry_count;

	if (op->flags & PRISM_POSTING_PAGE_FASTSCAN)
	{
		char	*content = prism_posting_page_content(pos->page, ctx->dim);
		uint32_t ngroups = (n + VS_FASTSCAN_GROUP - 1) / VS_FASTSCAN_GROUP;

		/* A group's tombstone_mask only ever has bits set for lanes that
		 * actually hold an entry -- nothing sets a bit for an empty lane,
		 * and a freshly built page starts fully zeroed -- so it's safe to
		 * count set bits across the whole 32-bit mask even for the last,
		 * partially-filled group; there's no need to mask off the unused
		 * high lanes first. */
		for (uint32_t g = 0; g < ngroups; g++)
		{
			uint32_t g_count = n - g * VS_FASTSCAN_GROUP;
			if (g_count > VS_FASTSCAN_GROUP)
				g_count = VS_FASTSCAN_GROUP;
			uint32_t mask =
					*prism_fastscan_group_tombstone_mask(content, g, ctx->dim);
			ctx->live += g_count - (uint32_t)__builtin_popcount(mask);
		}
		return true;
	}

	char *content = prism_posting_page_content(pos->page, ctx->dim);

	for (uint32_t i = 0; i < n; i++)
	{
		const PrismPostingEntryHeader *h =
				prism_posting_entry_at(content, i, ctx->dim);

		if (!(h->meta.flags & PRISM_POSTING_FLAG_DELETED))
			ctx->live++;
	}
	return true;
}

static uint32_t
chain_live_count(TestPageStorage *st, Dimension dim, BlockNumber head)
{
	LiveCtx ctx = {.dim = dim};

	prism_posting_chain_walk(&st->base, head, count_live_page, &ctx);
	return ctx.live;
}

/* ---------------------------------------------------------------- */

TEST(insert_appends_and_counts)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 42);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(8, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, 5, false);

	RaBitQScratch scratch;
	vs_rabitq_scratch_init(&scratch, dim);
	for (uint32_t i = 0; i < 3; i++)
		insert_vec(
				&storage,
				params,
				dim,
				head,
				100 + i,
				vecs + (size_t)(5 + i) * dim,
				&scratch);

	ASSERT_EQ(
			8,
			chain_live_count(&storage, dim, head),
			"chain should hold built + inserted");

	Page hp = vs_storage_read_page(&storage.base, head);
	ASSERT_NEQ(
			InvalidBlockNumber,
			prism_posting_head_tail(hp),
			"tail_blkno should be filled after first insert");
	ASSERT_EQ(
			8,
			prism_posting_head_live_count(hp),
			"live_count should track built + inserted");
	vs_storage_release_page(&storage.base, head);

	vs_rabitq_scratch_cleanup(&scratch);
	vs_rabitq_destroy(params);
}

TEST(insert_then_scan_finds_all)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 7);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(8, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, 5, false);
	RaBitQScratch scratch;
	vs_rabitq_scratch_init(&scratch, dim);
	for (uint32_t i = 0; i < 3; i++)
		insert_vec(
				&storage,
				params,
				dim,
				head,
				100 + i,
				vecs + (size_t)(5 + i) * dim,
				&scratch);

	ASSERT_EQ(
			8,
			scan_count(&storage, params, dim, centroid, head, 64, false),
			"scan should surface built + inserted entries");

	vs_rabitq_scratch_cleanup(&scratch);
	vs_rabitq_destroy(params);
}

TEST(insert_grows_overflow_page)
{
	Dimension		dim		  = 768;
	uint32_t		first_cap = prism_posting_max_entries_first(dim);
	uint32_t		ninsert	  = first_cap + 5; /* force a 2nd page */
	TestPageStorage storage	  = make_test_storage(32);
	RaBitQParams   *params	  = vs_rabitq_create(dim, 99);
	float		   *centroid  = vs_alloc0(dim * sizeof(float));
	float		   *vecs	  = make_test_vectors(ninsert + 1, dim);

	/* Start from a 1-entry cluster, then insert past the first page's cap. */
	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, 1, false);
	RaBitQScratch scratch;
	vs_rabitq_scratch_init(&scratch, dim);
	for (uint32_t i = 0; i < ninsert; i++)
		insert_vec(
				&storage,
				params,
				dim,
				head,
				100 + i,
				vecs + (size_t)(1 + i) * dim,
				&scratch);

	Page hp = vs_storage_read_page(&storage.base, head);
	ASSERT_NEQ(
			InvalidBlockNumber,
			prism_posting_opaque(hp)->next_blkno,
			"head should chain to an overflow page");
	vs_storage_release_page(&storage.base, head);

	ASSERT_EQ(
			1 + ninsert,
			chain_live_count(&storage, dim, head),
			"all entries reachable across the chain");

	vs_rabitq_scratch_cleanup(&scratch);
	vs_rabitq_destroy(params);
}

TEST(insert_into_fastscan_cluster_mixed_chain)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 3);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	uint32_t		nbuilt	 = 40; /* > 1 fastscan group */
	float		   *vecs	 = make_test_vectors(nbuilt + 3, dim);

	/* FASTSCAN base + AoS overflow inserts in the same chain. */
	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, nbuilt, true);
	RaBitQScratch scratch;
	vs_rabitq_scratch_init(&scratch, dim);
	for (uint32_t i = 0; i < 3; i++)
		insert_vec(
				&storage,
				params,
				dim,
				head,
				100 + i,
				vecs + (size_t)(nbuilt + i) * dim,
				&scratch);

	ASSERT_EQ(
			nbuilt + 3,
			scan_count(&storage, params, dim, centroid, head, 128, true),
			"fastscan scan merges packed base + AoS overflow inserts");

	vs_rabitq_scratch_cleanup(&scratch);
	vs_rabitq_destroy(params);
}

/* Dead-TID predicate for the tombstone test: a TID is dead if its vector id
 * is in the set. */
typedef struct DeadSet
{
	const uint32_t *vids;
	uint32_t		n;
} DeadSet;

static bool
vid_is_dead(ItemPointerData tid, void *state)
{
	const DeadSet *d   = (const DeadSet *)state;
	uint32_t	   vid = prism_posting_get_vector_id(&tid);
	for (uint32_t i = 0; i < d->n; i++)
		if (d->vids[i] == vid)
			return true;
	return false;
}

TEST(tombstone_marks_and_scan_skips)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 11);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(8, dim);

	/* 5 built + 3 inserted = 8 live AoS entries. */
	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, 5, false);
	RaBitQScratch scratch;
	vs_rabitq_scratch_init(&scratch, dim);
	for (uint32_t i = 0; i < 3; i++)
		insert_vec(
				&storage,
				params,
				dim,
				head,
				100 + i,
				vecs + (size_t)(5 + i) * dim,
				&scratch);

	ASSERT_EQ(
			8,
			chain_live_count(&storage, dim, head),
			"all 8 live before delete");

	/* Tombstone two built ids and one inserted id. */
	const uint32_t dead_vids[] = {1, 3, 101};
	DeadSet		   dead		   = {.vids = dead_vids, .n = 3};
	uint32_t	   marked	   = prism_posting_tombstone_chain(
			   &storage.base, dim, head, vid_is_dead, &dead);

	ASSERT_EQ(3, marked, "three entries tombstoned");
	ASSERT_EQ(
			5,
			chain_live_count(&storage, dim, head),
			"live count drops by the tombstoned entries");

	Page hp = vs_storage_read_page(&storage.base, head);
	ASSERT_EQ(
			5,
			prism_posting_head_live_count(hp),
			"head live_count decremented");
	vs_storage_release_page(&storage.base, head);

	ASSERT_EQ(
			5,
			scan_count(&storage, params, dim, centroid, head, 64, false),
			"scan skips tombstoned entries");

	/* Re-tombstoning the same ids marks nothing new (idempotent). */
	ASSERT_EQ(
			0,
			prism_posting_tombstone_chain(
					&storage.base, dim, head, vid_is_dead, &dead),
			"already-deleted entries are not re-counted");

	vs_rabitq_scratch_cleanup(&scratch);
	vs_rabitq_destroy(params);
}

/* Whole-page tombstone: when every entry on a page is dead, the page gets the
 * PRISM_POSTING_PAGE_TOMBSTONED flag and the scan skips it. Covers AoS, where
 * entries are also individually flagged. */
TEST(tombstone_all_flags_aos_page)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 5);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(8, dim);

	/* 8 entries on a single AoS head page. */
	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, 8, false);

	Page hp_before = vs_storage_read_page(&storage.base, head);
	ASSERT_TRUE(
			(prism_posting_opaque(hp_before)->flags &
			 PRISM_POSTING_PAGE_TOMBSTONED) == 0,
			"the built page starts untombstoned");
	ASSERT_EQ(8, prism_posting_head_live_count(hp_before), "with 8 live");
	vs_storage_release_page(&storage.base, head);
	ASSERT_EQ(
			8,
			scan_count(&storage, params, dim, centroid, head, 64, false),
			"and the scan reads all 8");

	const uint32_t dead_vids[] = {0, 1, 2, 3, 4, 5, 6, 7};
	DeadSet		   dead		   = {.vids = dead_vids, .n = 8};
	ASSERT_EQ(
			8,
			prism_posting_tombstone_chain(
					&storage.base, dim, head, vid_is_dead, &dead),
			"all 8 entries tombstoned");

	Page hp = vs_storage_read_page(&storage.base, head);
	ASSERT_TRUE(
			(prism_posting_opaque(hp)->flags &
			 PRISM_POSTING_PAGE_TOMBSTONED) != 0,
			"fully-dead AoS page is flagged tombstoned");
	ASSERT_EQ(0, prism_posting_head_live_count(hp), "live_count is zero");
	vs_storage_release_page(&storage.base, head);

	ASSERT_EQ(
			0,
			scan_count(&storage, params, dim, centroid, head, 64, false),
			"scan skips the tombstoned page");

	vs_rabitq_destroy(params);
}

TEST(tombstone_all_then_insert_unflags_aos_and_scans_new_entry)
{
	Dimension		dim		= 128;
	TestPageStorage storage = make_test_storage(32);
	RaBitQParams   *params	= vs_rabitq_create(dim, 5);
	RaBitQScratch	scratch;
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(8, dim);
	float		   *new_vec	 = vecs;

	vs_rabitq_scratch_init(&scratch, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, 8, false);

	const uint32_t dead_vids[] = {0, 1, 2, 3, 4, 5, 6, 7};
	DeadSet		   dead		   = {.vids = dead_vids, .n = 8};
	ASSERT_EQ(
			8,
			prism_posting_tombstone_chain(
					&storage.base, dim, head, vid_is_dead, &dead),
			"all AoS entries tombstoned");
	ASSERT_EQ(
			0,
			scan_count(&storage, params, dim, centroid, head, 64, false),
			"scan sees no live entries after full tombstone");

	insert_vec(&storage, params, dim, head, 1000, new_vec, &scratch);

	Page hp = vs_storage_read_page(&storage.base, head);
	ASSERT_TRUE(
			(prism_posting_opaque(hp)->flags &
			 PRISM_POSTING_PAGE_TOMBSTONED) == 0,
			"reinsertion clears tombstone state on AoS page");
	ASSERT_EQ(
			1,
			prism_posting_head_live_count(hp),
			"live_count reflects new row");
	vs_storage_release_page(&storage.base, head);

	ASSERT_EQ(
			1,
			scan_count(&storage, params, dim, centroid, head, 64, false),
			"scan returns the newly inserted entry");

	vs_rabitq_scratch_cleanup(&scratch);
	vs_rabitq_destroy(params);
}

/* A wholly-dead FASTSCAN page still gets PRISM_POSTING_PAGE_TOMBSTONED --
 * every lane of every group ends up marked, so the page-level shortcut
 * applies the same as it would with any subset of lanes dead -- it is not a
 * separate, page-only mechanism from the per-lane one covered below. */
TEST(tombstone_all_flags_fastscan_page)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 9);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	uint32_t		nbuilt	 = 40; /* > 1 fastscan group */
	float		   *vecs	 = make_test_vectors(nbuilt, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, nbuilt, true);

	Page hp_before = vs_storage_read_page(&storage.base, head);
	ASSERT_TRUE(
			(prism_posting_opaque(hp_before)->flags &
			 PRISM_POSTING_PAGE_TOMBSTONED) == 0,
			"the built fastscan page starts untombstoned");
	ASSERT_EQ(
			nbuilt,
			prism_posting_head_live_count(hp_before),
			"with every entry live");
	vs_storage_release_page(&storage.base, head);
	ASSERT_EQ(
			nbuilt,
			scan_count(&storage, params, dim, centroid, head, 128, true),
			"and the scan reads them all");

	uint32_t *dead_vids = vs_alloc(nbuilt * sizeof(uint32_t));
	for (uint32_t i = 0; i < nbuilt; i++)
		dead_vids[i] = i;
	DeadSet dead = {.vids = dead_vids, .n = nbuilt};

	ASSERT_EQ(
			nbuilt,
			prism_posting_tombstone_chain(
					&storage.base, dim, head, vid_is_dead, &dead),
			"all fastscan entries accounted as tombstoned");

	Page hp = vs_storage_read_page(&storage.base, head);
	ASSERT_TRUE(
			(prism_posting_opaque(hp)->flags &
			 PRISM_POSTING_PAGE_TOMBSTONED) != 0,
			"fully-dead FASTSCAN page is flagged tombstoned");
	ASSERT_EQ(
			0,
			prism_posting_head_live_count(hp),
			"fastscan live_count is zero");
	vs_storage_release_page(&storage.base, head);

	ASSERT_EQ(
			0,
			scan_count(&storage, params, dim, centroid, head, 128, true),
			"fastscan scan skips the tombstoned page(s)");

	vs_rabitq_destroy(params);
}

/*
 * Deleting some, but not all, of the vectors on a fastscan page must only
 * remove those specific vectors from search results. It must not force
 * the rest of the page, or even the rest of their 32-vector group, to be
 * treated as dead too. This is the fastscan equivalent of flipping one
 * AoS entry's delete flag: the per-group tombstone_mask lets a single
 * vector be marked dead without disturbing its 31 group-mates.
 */
TEST(tombstone_marks_fastscan_lanes_and_scan_skips)
{
	log_fastscan_avx512_coverage();

	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 13);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	uint32_t		nbuilt	 = 40; /* 2 groups: a full one + an 8-entry one */
	float		   *vecs	 = make_test_vectors(nbuilt, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, nbuilt, true);

	ASSERT_EQ(
			nbuilt,
			chain_live_count(&storage, dim, head),
			"all 40 live before delete");

	/*
	 * This cluster holds 40 vectors, which the fastscan format packs into
	 * one full group of 32 (ids 0-31) and one partial group of 8 (ids
	 * 32-39). Delete one vector from each: id 31 from the full group, and
	 * ids 35 and 39 from the partial one.
	 *
	 * Id 31 is deliberately the very last lane of the full group. On a
	 * CPU with AVX-512, the code that excludes dead lanes processes a
	 * 32-lane group as two chunks of 16 lanes each (a "lower half" and an
	 * "upper half"); id 31 sits in the very last position of the upper
	 * half, which is exactly where an off-by-one bug in that split would
	 * most likely go unnoticed.
	 *
	 * None of these three deletions empties an entire group, so the
	 * page-level "everything here is dead" shortcut must not kick in --
	 * only the per-lane marking is being tested here.
	 */
	const uint32_t dead_vids[] = {31, 35, 39};
	DeadSet		   dead		   = {.vids = dead_vids, .n = 3};
	uint32_t	   marked	   = prism_posting_tombstone_chain(
			   &storage.base, dim, head, vid_is_dead, &dead);

	ASSERT_EQ(3, marked, "three fastscan lanes tombstoned");
	ASSERT_EQ(
			nbuilt - 3,
			chain_live_count(&storage, dim, head),
			"live count drops by exactly the tombstoned lanes");

	Page hp = vs_storage_read_page(&storage.base, head);
	ASSERT_EQ(
			nbuilt - 3,
			prism_posting_head_live_count(hp),
			"head live_count decremented");
	ASSERT_TRUE(
			(prism_posting_opaque(hp)->flags &
			 PRISM_POSTING_PAGE_TOMBSTONED) == 0,
			"page stays untombstoned: neither group went fully dead");
	vs_storage_release_page(&storage.base, head);

	ASSERT_EQ(
			nbuilt - 3,
			scan_count(&storage, params, dim, centroid, head, 128, true),
			"scan excludes exactly the tombstoned lanes");

	/* Re-tombstoning the same ids marks nothing new (idempotent). */
	ASSERT_EQ(
			0,
			prism_posting_tombstone_chain(
					&storage.base, dim, head, vid_is_dead, &dead),
			"already-tombstoned lanes are not re-counted");

	vs_rabitq_destroy(params);
}

/*
 * Reproduces, directly, the exact bug this whole change fixes: a stale
 * index entry winning a result slot after its row is gone and its heap
 * TID has been handed to a completely different row.
 *
 * Background: "lane" here means one vector's slot inside a fastscan
 * group. The fastscan format packs 32 vectors together so one CPU
 * instruction can score all of them at once, and "lane" is just which
 * one of those 32 positions a given vector occupies. A fastscan entry's
 * stored bits never change once written, except for the one tombstone
 * bit that marks it dead. So here's the scenario: PostgreSQL deletes a
 * row, and VACUUM later recycles that row's heap TID for a brand new,
 * unrelated row. If nothing marked the old lane dead, it still holds the
 * deleted vector's bits -- now sitting under the new row's TID.
 *
 * A query that happens to probe exactly where the old, deleted vector
 * used to be would then see that stale lane as a near-perfect match,
 * with no way to tell it apart from a real one -- unless that lane has
 * been marked dead, in which case it is skipped unconditionally, no
 * matter how good a match it would otherwise be.
 *
 * This test checks exactly that: is the id present in the results at
 * all, not how well it would have scored. That's deliberate. Scoring
 * depends on the fine details of RaBitQ's approximate distance, which
 * can vary; whether a dead lane gets excluded does not.
 */
TEST(tombstoned_fastscan_lane_excluded_despite_perfect_stale_match)
{
	log_fastscan_avx512_coverage();

	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = vs_rabitq_create(dim, 23);
	float		   *centroid = vs_alloc0(dim * sizeof(float));
	uint32_t		nbuilt	 = 40; /* 2 groups, so tombstoning one lane
									* cannot tombstone the whole page */
	float *vecs = make_test_vectors(nbuilt, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, nbuilt, true);

	/*
	 * Pick one vector from each half of the full 32-vector group: id 12
	 * from the lower half (lanes 0-15) and id 20 from the upper half
	 * (lanes 16-31). Checking both halves separately, by id, rules out a
	 * bug where the wrong lane gets excluded in just one half: that kind
	 * of bug could still leave the overall count of dead entries correct
	 * (one lane excluded is still one lane excluded) while silently
	 * excluding a different vector than the one that was actually deleted.
	 */
	uint32_t victim_lo = 12;
	uint32_t victim_hi = 20;

	/* Query using each victim's own vector, unchanged. The real distance
	 * from a vector to itself is zero, so if that vector's stale entry is
	 * scored at all, it will look like the best possible match -- there's
	 * no plausible score a real bug could produce that this probe would
	 * fail to catch. */
	float *pt_query_lo = vs_alloc(dim * sizeof(float));
	float *pt_query_hi = vs_alloc(dim * sizeof(float));
	vs_rabitq_rotate(params, vecs + (size_t)victim_lo * dim, pt_query_lo);
	vs_rabitq_rotate(params, vecs + (size_t)victim_hi * dim, pt_query_hi);

	RaBitQQueryState qstate_lo, qstate_hi;
	qstate_lo.transformed = vs_alloc_aligned(dim * sizeof(float), 64);
	qstate_lo.query_bits  = vs_alloc_aligned(VS_RABITQ_BYTES(dim), 64);
	vs_rabitq_init_query_constants(&qstate_lo, dim);
	vs_rabitq_init_query_state(
			&qstate_lo,
			pt_query_lo,
			centroid,
			dim,
			VS_DISTANCE_MODE_ASYMMETRIC);

	qstate_hi.transformed = vs_alloc_aligned(dim * sizeof(float), 64);
	qstate_hi.query_bits  = vs_alloc_aligned(VS_RABITQ_BYTES(dim), 64);
	vs_rabitq_init_query_constants(&qstate_hi, dim);
	vs_rabitq_init_query_state(
			&qstate_hi,
			pt_query_hi,
			centroid,
			dim,
			VS_DISTANCE_MODE_ASYMMETRIC);

	/* Sanity check before deleting anything: both vectors are still
	 * findable, each as its own exact match for its own query. */
	ASSERT_TRUE(
			scan_contains_vid(
					&storage, params, dim, &qstate_lo, head, 3, victim_lo),
			"lower-half victim is found before its line pointer is reused");
	ASSERT_TRUE(
			scan_contains_vid(
					&storage, params, dim, &qstate_hi, head, 3, victim_hi),
			"upper-half victim is found before its line pointer is reused");

	/* This stands in for VACUUM running after the heap has already
	 * recycled both victims' line pointers for unrelated new rows: mark
	 * both lanes dead in one pass. The other 38 vectors in the cluster
	 * are untouched. */
	const uint32_t dead_vids[] = {victim_lo, victim_hi};
	DeadSet		   dead		   = {.vids = dead_vids, .n = 2};
	ASSERT_EQ(
			2,
			prism_posting_tombstone_chain(
					&storage.base, dim, head, vid_is_dead, &dead),
			"both victims' lanes tombstoned");

	/* Run the same two queries again. Neither victim's id should come
	 * back, even though its stale bits are still physically present in
	 * the index and would otherwise be the best possible match. Checking
	 * both halves separately means a bug confined to just one half's
	 * exclusion logic shows up as exactly one of these two assertions
	 * failing, not both -- which points straight at where to look. */
	ASSERT_TRUE(
			!scan_contains_vid(
					&storage, params, dim, &qstate_lo, head, 3, victim_lo),
			"lower-half lane excluded despite its perfect stale score");
	ASSERT_TRUE(
			!scan_contains_vid(
					&storage, params, dim, &qstate_hi, head, 3, victim_hi),
			"upper-half lane excluded despite its perfect stale score");

	vs_free(pt_query_lo);
	vs_free(pt_query_hi);
	vs_rabitq_destroy(params);
}
