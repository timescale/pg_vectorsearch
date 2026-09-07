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
#include "mkt_test.h"
#include "quant/rabitq.h"
#include "standalone/pg_compat.h"

TEST_GROUP(PostingInsert);
TEST_MEMCTX_FIXTURE();

static inline ItemPointerData
vid_to_tid(uint32_t vid)
{
	ItemPointerData tid;
	mkt_posting_set_vector_id(&tid, vid);
	return tid;
}

/* ---- Array-backed test storage (multi-buffer; commit/release no-ops) ---- */

typedef struct TestPageStorage
{
	MktStorage base;
	char	  *pages;
	uint32_t   next_blkno;
	uint32_t   page_cap;
} TestPageStorage;

static Page
test_read_page(MktStorage *self, BlockNumber blkno)
{
	TestPageStorage *s = (TestPageStorage *)self;
	return s->pages + (size_t)blkno * BLCKSZ;
}

static void
test_release_page(MktStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static Page
test_write_page(MktStorage *self, BlockNumber blkno)
{
	TestPageStorage *s = (TestPageStorage *)self;
	return s->pages + (size_t)blkno * BLCKSZ;
}

static Page
test_new_page(MktStorage *self, BlockNumber *blkno_out)
{
	TestPageStorage *s = (TestPageStorage *)self;
	*blkno_out		   = s->next_blkno++;
	return s->pages + (size_t)*blkno_out * BLCKSZ;
}

static void
test_commit_page(MktStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static const MktStorageOps test_storage_ops = {
		.read_page	  = test_read_page,
		.release_page = test_release_page,
		.write_page	  = test_write_page,
		.new_page	  = test_new_page,
		.commit_page  = test_commit_page,
};

static TestPageStorage
make_test_storage(uint32_t num_pages)
{
	return (TestPageStorage){
			.base		= {.ops = &test_storage_ops},
			.pages		= mkt_alloc0((size_t)num_pages * BLCKSZ),
			.next_blkno = 0,
			.page_cap	= num_pages,
	};
}

static float *
make_test_vectors(uint32_t nvecs, Dimension dim)
{
	float *vecs = mkt_alloc(nvecs * dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs; i++)
		for (Dimension d = 0; d < dim; d++)
			vecs[(size_t)i * dim + d] = (float)((i * 13 + d * 7) % 100 - 50) /
										10.0f;
	return vecs;
}

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
	float *pt = mkt_alloc_aligned((size_t)dim * sizeof(float), 64);
	mkt_rabitq_rotate(params, vec, pt);
	mkt_posting_insert_one(
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

static void
setup_query_state(
		RaBitQQueryState *qstate,
		RaBitQParams	 *params,
		const float		 *centroid,
		Dimension		  dim)
{
	float *query	= mkt_alloc(dim * sizeof(float));
	float *pt_query = mkt_alloc(dim * sizeof(float));
	for (Dimension d = 0; d < dim; d++)
		query[d] = (float)(d % 10) / 5.0f;
	qstate->transformed = mkt_alloc_aligned(dim * sizeof(float), 64);
	qstate->query_bits	= mkt_alloc_aligned(MKT_RABITQ_BYTES(dim), 64);
	mkt_rabitq_init_query_constants(qstate, dim);
	mkt_rabitq_rotate(params, query, pt_query);
	mkt_rabitq_init_query_state(
			qstate, pt_query, centroid, dim, MKT_DISTANCE_MODE_ASYMMETRIC);
}

/* Build a single-cluster posting list of nbuilt vectors (AoS or fastscan). */
static BlockNumber
build_cluster(
		TestPageStorage *st,
		RaBitQParams	*params,
		Dimension		 dim,
		const float		*centroid,
		const float		*vecs,
		uint32_t		 nbuilt,
		bool			 fastscan)
{
	MktPostingBuilder builder;
	if (fastscan)
		mkt_posting_builder_init_fastscan(
				&builder, &st->base, params, dim, 0, centroid, centroid);
	else
		mkt_posting_builder_init(
				&builder, &st->base, params, dim, 0, centroid, centroid);
	for (uint32_t i = 0; i < nbuilt; i++)
		mkt_posting_builder_add(
				&builder, vid_to_tid(i), vecs + (size_t)i * dim);
	BlockNumber head = mkt_posting_builder_finish(&builder);
	mkt_posting_builder_cleanup(&builder);
	return head;
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

	MktPostingScan scan;
	mkt_posting_scan_init(
			&scan, &st->base, NULL, params, dim, mkt_posting_max_entries(dim));
	if (fastscan)
		mkt_posting_scan_enable_fastscan(&scan, 16);
	MktTopK topk;
	mkt_topk_init(&topk, k);

	mkt_posting_scan_begin_cluster(&scan, &qstate, head);
	if (fastscan)
		mkt_posting_scan_cluster_fastscan(&scan, &topk);
	else
		mkt_posting_scan_cluster(&scan, &topk);
	mkt_posting_scan_end_cluster(&scan);

	uint32_t n = topk.cand_count;
	mkt_topk_cleanup(&topk);
	mkt_posting_scan_cleanup(&scan);
	return n;
}

/*
 * Ground-truth walk of a cluster chain: count live (non-deleted) entries.
 * FASTSCAN dead entries aren't marked, so they count as live (matching how the
 * build stamps live_count). Used to validate the head's maintained live_count
 * against an independent walk — the production insert path never walks.
 */
static uint32_t
chain_live_count(TestPageStorage *st, Dimension dim, BlockNumber head)
{
	uint32_t	live = 0;
	BlockNumber blk	 = head;

	while (blk != InvalidBlockNumber)
	{
		Page						p  = mkt_storage_read_page(&st->base, blk);
		const MktPostingPageOpaque *op = mkt_posting_opaque(p);
		BlockNumber					next = op->next_blkno;
		uint32_t					n	 = op->entry_count;

		if (op->flags & MKT_POSTING_PAGE_FASTSCAN)
		{
			live += n;
		}
		else
		{
			char *content = (op->flags & MKT_POSTING_PAGE_FIRST)
								  ? mkt_posting_content_first(p, dim)
								  : mkt_posting_content(p);
			for (uint32_t i = 0; i < n; i++)
			{
				MktPostingEntryHeader *h =
						mkt_posting_entry_at(content, i, dim);
				if (!(h->meta.flags & MKT_POSTING_FLAG_DELETED))
					live++;
			}
		}

		mkt_storage_release_page(&st->base, blk);
		blk = next;
	}

	return live;
}

/* ---------------------------------------------------------------- */

TEST(insert_appends_and_counts)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = mkt_rabitq_create(dim, 42);
	float		   *centroid = mkt_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(8, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, 5, false);

	RaBitQScratch scratch;
	mkt_rabitq_scratch_init(&scratch, dim);
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

	Page hp = mkt_storage_read_page(&storage.base, head);
	ASSERT_NEQ(
			InvalidBlockNumber,
			mkt_posting_head_tail(hp),
			"tail_blkno should be filled after first insert");
	ASSERT_EQ(
			8,
			mkt_posting_head_live_count(hp),
			"live_count should track built + inserted");
	mkt_storage_release_page(&storage.base, head);

	mkt_rabitq_scratch_cleanup(&scratch);
	mkt_rabitq_destroy(params);
}

TEST(insert_then_scan_finds_all)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = mkt_rabitq_create(dim, 7);
	float		   *centroid = mkt_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(8, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, 5, false);
	RaBitQScratch scratch;
	mkt_rabitq_scratch_init(&scratch, dim);
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

	mkt_rabitq_scratch_cleanup(&scratch);
	mkt_rabitq_destroy(params);
}

TEST(insert_grows_overflow_page)
{
	Dimension		dim		  = 768;
	uint32_t		first_cap = mkt_posting_max_entries_first(dim);
	uint32_t		ninsert	  = first_cap + 5; /* force a 2nd page */
	TestPageStorage storage	  = make_test_storage(32);
	RaBitQParams   *params	  = mkt_rabitq_create(dim, 99);
	float		   *centroid  = mkt_alloc0(dim * sizeof(float));
	float		   *vecs	  = make_test_vectors(ninsert + 1, dim);

	/* Start from a 1-entry cluster, then insert past the first page's cap. */
	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, 1, false);
	RaBitQScratch scratch;
	mkt_rabitq_scratch_init(&scratch, dim);
	for (uint32_t i = 0; i < ninsert; i++)
		insert_vec(
				&storage,
				params,
				dim,
				head,
				100 + i,
				vecs + (size_t)(1 + i) * dim,
				&scratch);

	Page hp = mkt_storage_read_page(&storage.base, head);
	ASSERT_NEQ(
			InvalidBlockNumber,
			mkt_posting_opaque(hp)->next_blkno,
			"head should chain to an overflow page");
	mkt_storage_release_page(&storage.base, head);

	ASSERT_EQ(
			1 + ninsert,
			chain_live_count(&storage, dim, head),
			"all entries reachable across the chain");

	mkt_rabitq_scratch_cleanup(&scratch);
	mkt_rabitq_destroy(params);
}

TEST(insert_into_fastscan_cluster_mixed_chain)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = mkt_rabitq_create(dim, 3);
	float		   *centroid = mkt_alloc0(dim * sizeof(float));
	uint32_t		nbuilt	 = 40; /* > 1 fastscan group */
	float		   *vecs	 = make_test_vectors(nbuilt + 3, dim);

	/* FASTSCAN base + AoS overflow inserts in the same chain. */
	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, nbuilt, true);
	RaBitQScratch scratch;
	mkt_rabitq_scratch_init(&scratch, dim);
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

	mkt_rabitq_scratch_cleanup(&scratch);
	mkt_rabitq_destroy(params);
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
	uint32_t	   vid = mkt_posting_get_vector_id(&tid);
	for (uint32_t i = 0; i < d->n; i++)
		if (d->vids[i] == vid)
			return true;
	return false;
}

TEST(tombstone_marks_and_scan_skips)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = mkt_rabitq_create(dim, 11);
	float		   *centroid = mkt_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(8, dim);

	/* 5 built + 3 inserted = 8 live AoS entries. */
	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, 5, false);
	RaBitQScratch scratch;
	mkt_rabitq_scratch_init(&scratch, dim);
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
	uint32_t	   marked	   = mkt_posting_tombstone_chain(
			   &storage.base, dim, head, vid_is_dead, &dead);

	ASSERT_EQ(3, marked, "three entries tombstoned");
	ASSERT_EQ(
			5,
			chain_live_count(&storage, dim, head),
			"live count drops by the tombstoned entries");

	Page hp = mkt_storage_read_page(&storage.base, head);
	ASSERT_EQ(
			5, mkt_posting_head_live_count(hp), "head live_count decremented");
	mkt_storage_release_page(&storage.base, head);

	ASSERT_EQ(
			5,
			scan_count(&storage, params, dim, centroid, head, 64, false),
			"scan skips tombstoned entries");

	/* Re-tombstoning the same ids marks nothing new (idempotent). */
	ASSERT_EQ(
			0,
			mkt_posting_tombstone_chain(
					&storage.base, dim, head, vid_is_dead, &dead),
			"already-deleted entries are not re-counted");

	mkt_rabitq_scratch_cleanup(&scratch);
	mkt_rabitq_destroy(params);
}

/* Whole-page tombstone: when every entry on a page is dead, the page gets the
 * MKT_POSTING_PAGE_TOMBSTONED flag and the scan skips it. Covers AoS, where
 * entries are also individually flagged. */
TEST(tombstone_all_flags_aos_page)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = mkt_rabitq_create(dim, 5);
	float		   *centroid = mkt_alloc0(dim * sizeof(float));
	float		   *vecs	 = make_test_vectors(8, dim);

	/* 8 entries on a single AoS head page. */
	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, 8, false);

	Page hp_before = mkt_storage_read_page(&storage.base, head);
	ASSERT_TRUE(
			(mkt_posting_opaque(hp_before)->flags &
			 MKT_POSTING_PAGE_TOMBSTONED) == 0,
			"the built page starts untombstoned");
	ASSERT_EQ(8, mkt_posting_head_live_count(hp_before), "with 8 live");
	mkt_storage_release_page(&storage.base, head);
	ASSERT_EQ(
			8,
			scan_count(&storage, params, dim, centroid, head, 64, false),
			"and the scan reads all 8");

	const uint32_t dead_vids[] = {0, 1, 2, 3, 4, 5, 6, 7};
	DeadSet		   dead		   = {.vids = dead_vids, .n = 8};
	ASSERT_EQ(
			8,
			mkt_posting_tombstone_chain(
					&storage.base, dim, head, vid_is_dead, &dead),
			"all 8 entries tombstoned");

	Page hp = mkt_storage_read_page(&storage.base, head);
	ASSERT_TRUE(
			(mkt_posting_opaque(hp)->flags & MKT_POSTING_PAGE_TOMBSTONED) != 0,
			"fully-dead AoS page is flagged tombstoned");
	ASSERT_EQ(0, mkt_posting_head_live_count(hp), "live_count is zero");
	mkt_storage_release_page(&storage.base, head);

	ASSERT_EQ(
			0,
			scan_count(&storage, params, dim, centroid, head, 64, false),
			"scan skips the tombstoned page");

	mkt_rabitq_destroy(params);
}

/* FASTSCAN entries can't be flagged individually, but a wholly-dead FASTSCAN
 * page is tombstoned at page granularity (and its entries leave live_count).
 */
TEST(tombstone_all_flags_fastscan_page)
{
	Dimension		dim		 = 128;
	TestPageStorage storage	 = make_test_storage(32);
	RaBitQParams   *params	 = mkt_rabitq_create(dim, 9);
	float		   *centroid = mkt_alloc0(dim * sizeof(float));
	uint32_t		nbuilt	 = 40; /* > 1 fastscan group */
	float		   *vecs	 = make_test_vectors(nbuilt, dim);

	BlockNumber head =
			build_cluster(&storage, params, dim, centroid, vecs, nbuilt, true);

	Page hp_before = mkt_storage_read_page(&storage.base, head);
	ASSERT_TRUE(
			(mkt_posting_opaque(hp_before)->flags &
			 MKT_POSTING_PAGE_TOMBSTONED) == 0,
			"the built fastscan page starts untombstoned");
	ASSERT_EQ(
			nbuilt,
			mkt_posting_head_live_count(hp_before),
			"with every entry live");
	mkt_storage_release_page(&storage.base, head);
	ASSERT_EQ(
			nbuilt,
			scan_count(&storage, params, dim, centroid, head, 128, true),
			"and the scan reads them all");

	uint32_t *dead_vids = mkt_alloc(nbuilt * sizeof(uint32_t));
	for (uint32_t i = 0; i < nbuilt; i++)
		dead_vids[i] = i;
	DeadSet dead = {.vids = dead_vids, .n = nbuilt};

	ASSERT_EQ(
			nbuilt,
			mkt_posting_tombstone_chain(
					&storage.base, dim, head, vid_is_dead, &dead),
			"all fastscan entries accounted as tombstoned");

	Page hp = mkt_storage_read_page(&storage.base, head);
	ASSERT_TRUE(
			(mkt_posting_opaque(hp)->flags & MKT_POSTING_PAGE_TOMBSTONED) != 0,
			"fully-dead FASTSCAN page is flagged tombstoned");
	ASSERT_EQ(
			0, mkt_posting_head_live_count(hp), "fastscan live_count is zero");
	mkt_storage_release_page(&storage.base, head);

	ASSERT_EQ(
			0,
			scan_count(&storage, params, dim, centroid, head, 128, true),
			"fastscan scan skips the tombstoned page(s)");

	mkt_rabitq_destroy(params);
}
