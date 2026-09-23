/*
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
 * Ground-truth walk of a cluster chain: count live (non-deleted) entries.
 * FASTSCAN dead entries aren't marked, so they count as live (matching how the
 * build stamps live_count). Used to validate the head's maintained live_count
 * against an independent walk — the production insert path never walks.
 */
typedef struct LiveCtx
{
	Dimension dim;
	uint32_t  live;
} LiveCtx;

/* A fastscan page has no per-entry flag, so all its entries count. */
static bool
count_live_page(PrismPostingChainPos *pos, void *state)
{
	LiveCtx						 *ctx = state;
	const PrismPostingPageOpaque *op  = prism_posting_opaque(pos->page);
	uint32_t					  n	  = op->entry_count;

	if (op->flags & PRISM_POSTING_PAGE_FASTSCAN)
	{
		ctx->live += n;
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

/* FASTSCAN entries can't be flagged individually, but a wholly-dead FASTSCAN
 * page is tombstoned at page granularity (and its entries leave live_count).
 */
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
