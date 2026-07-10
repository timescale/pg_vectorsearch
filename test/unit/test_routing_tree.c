/*
 * test_routing_tree.c - Routing-tree build invariants
 *
 * The streaming build's contract: the PLAN pass clusters the sample once,
 * records each node's clustering in a blob store, and sizes the page
 * layout; the WRITE pass replays the records and must (a) consume exactly
 * the planned centroid pages, (b) fire on_leaf once per leaf with dense
 * ascending indices, (c) link every leaf entry to its formula-derived
 * posting head (first_posting + leaf, each exactly once), and (d) produce
 * byte-identical pages to a self-contained re-clustering write (the replay
 * carries the same clustering the plan computed). Also covers the blob
 * store's put/rewind/get contract, including growth well past its initial
 * buffer and zero-length records.
 */

#include <string.h>

#include "algo/kmeans.h"
#include "core/memory.h"
#include "index/centroid_page.h"
#include "index/index_build.h"
#include "index/parallel_build.h"
#include "mkt_test.h"
#include "quant/rabitq.h"
#include "standalone/pg_compat.h"

TEST_GROUP(RoutingTree);
TEST_MEMCTX_FIXTURE();

/* ----------------------------------------------------------------
 * Minimal in-memory page storage (mirrors test_posting_page.c)
 * ---------------------------------------------------------------- */

typedef struct TestPageStorage
{
	MktStorage base;
	char	  *pages;
	uint32_t   next_blkno;
	uint32_t   page_cap;
} TestPageStorage;

static Page
tps_read_page(MktStorage *self, BlockNumber blkno)
{
	TestPageStorage *s = (TestPageStorage *)self;
	return s->pages + (size_t)blkno * BLCKSZ;
}

static void
tps_release_page(MktStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static Page
tps_write_page(MktStorage *self, BlockNumber blkno)
{
	TestPageStorage *s = (TestPageStorage *)self;
	return s->pages + (size_t)blkno * BLCKSZ;
}

static Page
tps_new_page(MktStorage *self, BlockNumber *blkno_out)
{
	TestPageStorage *s = (TestPageStorage *)self;
	*blkno_out		   = s->next_blkno++;
	return s->pages + (size_t)*blkno_out * BLCKSZ;
}

static void
tps_commit_page(MktStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static const MktStorageOps tps_ops = {
		.read_page	  = tps_read_page,
		.release_page = tps_release_page,
		.write_page	  = tps_write_page,
		.new_page	  = tps_new_page,
		.commit_page  = tps_commit_page,
};

static void
tps_init(TestPageStorage *s, uint32_t page_cap, uint32_t next_blkno)
{
	memset(s, 0, sizeof(*s));
	s->base.ops	  = &tps_ops;
	s->pages	  = mkt_alloc0((size_t)page_cap * BLCKSZ);
	s->page_cap	  = page_cap;
	s->next_blkno = next_blkno;
}

/* Deterministic pseudo-random vectors (local copy of the standalone test
 * helper; rand() keeps runs reproducible under the fixed seed). */
static float *
make_vectors(uint32_t nvecs, uint32_t dim, uint32_t seed)
{
	srand(seed);
	float *data = mkt_alloc((size_t)nvecs * dim * sizeof(float));
	for (uint32_t i = 0; i < nvecs * dim; i++)
		data[i] = (float)(rand() % 10000 - 5000) / 5000.0f;
	return data;
}

/* ----------------------------------------------------------------
 * Blob store contract
 * ---------------------------------------------------------------- */

TEST(blobstore_roundtrip)
{
	MktBlobStore  *bs	  = mkt_pbuild_blobstore_begin();
	const uint32_t nblobs = 50;
	uint64_t	   sizes[50];

	/* ~4.7 MB total: several doublings past the store's initial buffer. */
	char *buf = mkt_alloc(200000);
	for (uint32_t i = 0; i < nblobs; i++)
	{
		sizes[i] = (i == 25) ? 0 : 50000 + (i * 7919) % 90000;
		for (uint64_t j = 0; j < sizes[i]; j++)
			buf[j] = (char)((i * 131 + j * 17) & 0xFF);
		mkt_pbuild_blobstore_put(bs, buf, sizes[i]);
	}

	mkt_pbuild_blobstore_rewind(bs);

	char *got = mkt_alloc(200000);
	for (uint32_t i = 0; i < nblobs; i++)
	{
		uint64_t sz = mkt_pbuild_blobstore_get(bs, got, 200000);
		ASSERT_EQ(sizes[i], sz, "blob size round-trips");
		for (uint64_t j = 0; j < sz; j++)
			if (got[j] != (char)((i * 131 + j * 17) & 0xFF))
			{
				ASSERT_TRUE(false, "blob payload byte mismatch");
				break;
			}
	}
	mkt_pbuild_blobstore_end(bs);
	mkt_free(buf);
	mkt_free(got);
}

/* ----------------------------------------------------------------
 * Plan/write round-trip invariants
 * ---------------------------------------------------------------- */

typedef struct LeafProbe
{
	uint32_t count;
	bool	 ascending;
} LeafProbe;

static void
leaf_probe_cb(void *arg, uint32_t leaf, const float *centroid)
{
	LeafProbe *p = (LeafProbe *)arg;
	(void)centroid;
	if (leaf != p->count)
		p->ascending = false;
	p->count++;
}

/*
 * Walk the centroid pages [first_centroid, first_posting) and verify the
 * leaf entries' formula links: the set of leaf children is exactly
 * {first_posting .. first_posting + nleaves - 1}, each once.
 */
static void
verify_leaf_links(
		MktTestResult	*result,
		TestPageStorage *st,
		Dimension		 dim,
		BlockNumber		 first_centroid,
		BlockNumber		 first_posting,
		uint32_t		 nleaves,
		uint32_t		 nlevels)
{
	bool *seen = mkt_alloc0((size_t)nleaves * sizeof(bool));

	for (BlockNumber blk = first_centroid; blk < first_posting; blk++)
	{
		Page				   page	  = st->pages + (size_t)blk * BLCKSZ;
		MktCentroidPageOpaque *opaque = MKT_CENTROID_OPAQUE(page);
		MktCentroidFormat	   fmt	  = mkt_centroid_page_format(page);
		/* FASTSCAN stores no leaf flag at all -- leaf-ness is positional
		 * (the deepest level); the other formats flag each entry. */
		bool fs_leaf_page = (opaque->level == nlevels - 1);

		for (uint16_t i = 0; i < opaque->entry_count; i++)
		{
			BlockNumber child;
			bool		is_leaf;
			if (fmt == MKT_CENTROID_FMT_FASTSCAN)
			{
				if (!fs_leaf_page)
					continue;
				char	*content = (char *)PageGetContents(page);
				uint32_t g		 = i / MKT_FASTSCAN_GROUP;
				uint32_t slot	 = i % MKT_FASTSCAN_GROUP;
				child			 = mkt_centroid_fastscan_group_child(
						   content, g, dim)[slot];
				is_leaf = true;
			}
			else
			{
				const MktCentroidEntryMeta *entry = mkt_centroid_meta(page, i);
				is_leaf = (entry->flags & MKT_CENTROID_FLAG_LEAF) != 0;
				child	= entry->child_blkno;
			}
			if (!is_leaf)
				continue;

			ASSERT_TRUE(
					child >= first_posting && child < first_posting + nleaves,
					"leaf child within the head region");
			uint32_t leaf = child - first_posting;
			ASSERT_TRUE(!seen[leaf], "each head linked exactly once");
			seen[leaf] = true;
		}
	}
	for (uint32_t l = 0; l < nleaves; l++)
		ASSERT_TRUE(seen[l], "every head linked from a leaf entry");
	mkt_free(seen);
}

static void
check_plan_write_roundtrip(
		MktTestResult	 *result,
		MktCentroidFormat fmt,
		uint32_t		  nvecs,
		Dimension		  dim,
		uint32_t		  nlist,
		uint32_t		  fan_out)
{
	float *vecs = make_vectors(nvecs, dim, 42);

	KMeansOptions opts = MKT_KMEANS_OPTIONS_DEFAULT;
	opts.algorithm	   = KMEANS_ALGO_LLOYD;

	MktBlobStore	 *store = mkt_pbuild_blobstore_begin();
	MktStreamTreePlan plan;
	ASSERT_TRUE(
			mkt_routing_tree_plan(
					vecs,
					nvecs,
					dim,
					nlist,
					fan_out,
					DISTANCE_L2,
					fmt,
					&opts,
					store,
					&plan),
			"plan pass succeeds");
	ASSERT_TRUE(plan.nleaves > 0, "plan reports leaves");
	ASSERT_TRUE(plan.centroid_pages > 0, "plan reports pages");

	BlockNumber first_centroid = 1;
	BlockNumber first_posting  = 1 + (BlockNumber)plan.centroid_pages;
	uint32_t	page_cap	   = first_posting + plan.nleaves + 8;

	RaBitQParams *rq = mkt_rabitq_create(dim, 42);

	/* Write pass A: replay from the blob store. */
	TestPageStorage sa;
	tps_init(&sa, page_cap, first_posting + plan.nleaves);
	mkt_pbuild_blobstore_rewind(store);
	LeafProbe	probe = {.count = 0, .ascending = true};
	BlockNumber root  = mkt_routing_tree_write(
			 &sa.base,
			 nvecs,
			 dim,
			 nlist,
			 fan_out,
			 fmt,
			 rq,
			 plan.leaf_mean,
			 store,
			 first_posting,
			 first_centroid,
			 leaf_probe_cb,
			 &probe);
	mkt_pbuild_blobstore_end(store);

	/* The DFS writes post-order: the root page lands last, inside the
	 * planned range (the parallel leader pins its root at block 1 instead;
	 * the serial contract is only that the layout fits the plan). */
	ASSERT_TRUE(
			root != InvalidBlockNumber && root >= first_centroid &&
					root < first_posting,
			"root lands inside the planned centroid range");
	ASSERT_EQ(plan.nleaves, probe.count, "on_leaf fires once per leaf");
	ASSERT_TRUE(probe.ascending, "on_leaf leaf indices are dense ascending");
	ASSERT_EQ(
			first_posting + plan.nleaves,
			sa.next_blkno,
			"write pass appends nothing past the reserved layout");

	verify_leaf_links(
			result,
			&sa,
			dim,
			first_centroid,
			first_posting,
			plan.nleaves,
			plan.nlevels);

	/* Write pass B: an independent plan pass re-clusters the same sample
	 * from scratch (the k-means seed is fixed), so its tape -- and the
	 * pages replayed from it -- must be byte-identical to pass A's. */
	MktBlobStore	 *store_b = mkt_pbuild_blobstore_begin();
	MktStreamTreePlan plan_b;
	ASSERT_TRUE(
			mkt_routing_tree_plan(
					vecs,
					nvecs,
					dim,
					nlist,
					fan_out,
					DISTANCE_L2,
					fmt,
					&opts,
					store_b,
					&plan_b),
			"independent plan pass succeeds");
	ASSERT_EQ(plan.nleaves, plan_b.nleaves, "plans agree on leaves");
	ASSERT_EQ(
			plan.centroid_pages,
			plan_b.centroid_pages,
			"plans agree on pages");
	TestPageStorage sb;
	tps_init(&sb, page_cap, first_posting + plan.nleaves);
	mkt_pbuild_blobstore_rewind(store_b);
	LeafProbe	probe_b = {.count = 0, .ascending = true};
	BlockNumber root_b	= mkt_routing_tree_write(
			 &sb.base,
			 nvecs,
			 dim,
			 nlist,
			 fan_out,
			 fmt,
			 rq,
			 plan_b.leaf_mean,
			 store_b,
			 first_posting,
			 first_centroid,
			 leaf_probe_cb,
			 &probe_b);
	mkt_pbuild_blobstore_end(store_b);
	ASSERT_EQ(root, root_b, "replay and re-cluster agree on the root");
	ASSERT_EQ(probe.count, probe_b.count, "same leaf count");
	ASSERT_TRUE(
			memcmp(sa.pages + (size_t)first_centroid * BLCKSZ,
				   sb.pages + (size_t)first_centroid * BLCKSZ,
				   (size_t)plan.centroid_pages * BLCKSZ) == 0,
			"replayed pages byte-identical to re-clustered pages");

	mkt_free(sa.pages);
	mkt_free(sb.pages);
	mkt_rabitq_destroy(rq);
	mkt_free(vecs);
}

TEST(plan_write_roundtrip_rabitq_depth3)
{
	check_plan_write_roundtrip(
			result, MKT_CENTROID_FMT_RABITQ, 600, 16, 40, 4);
}

TEST(plan_write_roundtrip_float_depth3)
{
	check_plan_write_roundtrip(result, MKT_CENTROID_FMT_FLOAT, 600, 16, 40, 4);
}

TEST(plan_write_roundtrip_fastscan_group_boundary)
{
	/* 33 leaves requested: one past the fastscan group size, the layout
	 * math's sharpest edge. */
	check_plan_write_roundtrip(
			result, MKT_CENTROID_FMT_FASTSCAN, 600, 16, 33, 8);
}

TEST(plan_write_roundtrip_flat_multipage_float)
{
	/* Flat tree whose single level spans several pages. */
	check_plan_write_roundtrip(
			result, MKT_CENTROID_FMT_FLOAT, 600, 64, 200, 255);
}

TEST(plan_write_roundtrip_flat_multipage_fastscan)
{
	check_plan_write_roundtrip(
			result, MKT_CENTROID_FMT_FASTSCAN, 600, 64, 200, 255);
}
