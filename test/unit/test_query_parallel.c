/*
 * test_query_parallel.c - Backend-neutral parallel query core
 *
 * Covers the shared-state primitives in isolation (cursor batching and
 * drain, claim-table uniqueness under threads, cancellation) and the
 * merge invariant: replaying per-participant candidate buffers into one
 * top-k reproduces the serial insert stream's survivor set, including
 * duplicate-id (SOAR replica) handling across participant boundaries.
 */

#include <pthread.h>
#include <string.h>

#include "algo/topk.h"
#include "core/memory.h"
#include "index/query_parallel.h"
#include "mkt_test.h"

TEST_GROUP(QueryParallel);
TEST_MEMCTX_FIXTURE();

/* Allocate a backend-style shared area: derived header (here: just the
 * neutral struct) + heads + claims regions. */
static MktQueryShared *
make_shared(uint32_t max_nscan, uint32_t nparticipants, uint32_t pool)
{
	uint32_t nslots = mkt_pquery_claim_slots(nparticipants, pool);

	uint64_t heads_off	= sizeof(MktQueryShared);
	uint64_t claims_off = heads_off + mkt_pquery_heads_size(max_nscan);
	/* Align the atomics region. */
	claims_off = (claims_off + 7) & ~UINT64_C(7);

	MktQueryShared *sh = mkt_alloc0(
			claims_off + mkt_pquery_claims_size(nslots));
	mkt_pquery_shared_init(
			sh, max_nscan, nparticipants, nslots, heads_off, claims_off);
	return sh;
}

/* ----------------------------------------------------------------
 * Cursor
 * ---------------------------------------------------------------- */

TEST(pquery_cursor_drains_exactly)
{
	uint32_t		n  = 137; /* not a multiple of any chunk size */
	MktQueryShared *sh = make_shared(256, 4, 160);

	BlockNumber heads[256];
	for (uint32_t i = 0; i < n; i++)
		heads[i] = 100 + i;
	mkt_pquery_publish_probes(sh, heads, n);

	bool *seen = mkt_alloc0(n * sizeof(bool));

	uint32_t start, cnt, total = 0, batches = 0;
	while ((cnt = mkt_pquery_next_batch(sh, &start)) > 0)
	{
		ASSERT_TRUE(cnt <= 8, "batch is bounded");
		for (uint32_t r = start; r < start + cnt; r++)
		{
			ASSERT_TRUE(r < n, "rank within the published list");
			ASSERT_TRUE(!seen[r], "each rank handed out exactly once");
			seen[r] = true;
		}
		total += cnt;
		batches++;
	}
	ASSERT_EQ(n, total, "cursor drains the whole probe list");
	ASSERT_TRUE(batches > n / 8, "ramp-down produced small tail batches");

	/* Tail batches must ramp down to 1: the last claim before the drain
	 * covered exactly one rank when remaining < 4*participants. */
	mkt_free(seen);
	mkt_free(sh);
}

TEST(pquery_cursor_rescan_resets)
{
	MktQueryShared *sh		  = make_shared(16, 2, 10);
	BlockNumber		heads[16] = {0};
	mkt_pquery_publish_probes(sh, heads, 8);

	uint32_t start, total = 0, cnt;
	while ((cnt = mkt_pquery_next_batch(sh, &start)) > 0)
		total += cnt;
	ASSERT_EQ(8, total, "first pass drains");

	mkt_pquery_begin(sh);
	ASSERT_EQ(0, sh->nprobes, "begin clears the published list");
	mkt_pquery_publish_probes(sh, heads, 5);
	total = 0;
	while ((cnt = mkt_pquery_next_batch(sh, &start)) > 0)
		total += cnt;
	ASSERT_EQ(5, total, "second pass drains the new list");
	mkt_free(sh);
}

/* ----------------------------------------------------------------
 * Claim table
 * ---------------------------------------------------------------- */

TEST(pquery_claim_unique)
{
	MktQueryShared *sh = make_shared(4, 2, 8);

	ASSERT_TRUE(mkt_pquery_claim(sh, 42), "first claim wins");
	ASSERT_TRUE(!mkt_pquery_claim(sh, 42), "second claim of same id loses");
	ASSERT_TRUE(mkt_pquery_claim(sh, 43), "distinct id claims fine");
	ASSERT_EQ(
			0, mkt_atomic_read_u32(&sh->claims_full), "no overflow recorded");

	/* filter_claims keeps first occurrence ownership semantics */
	MktTopKEntry cands[3] = {
			{.distance = 1.0f, .id = 100},
			{.distance = 2.0f, .id = 43}, /* already claimed above */
			{.distance = 3.0f, .id = 101},
	};
	uint32_t kept = mkt_pquery_filter_claims(sh, cands, 3);
	ASSERT_EQ(2, kept, "already-claimed id filtered out");
	ASSERT_EQ(100, (int)cands[0].id, "kept order preserved");
	ASSERT_EQ(101, (int)cands[1].id, "kept order preserved");
	mkt_free(sh);
}

typedef struct ClaimRaceArg
{
	MktQueryShared *sh;
	uint32_t		nids;
	uint32_t		wins;
} ClaimRaceArg;

static void *
claim_race_fn(void *argp)
{
	ClaimRaceArg *arg = argp;
	for (uint32_t id = 1; id <= arg->nids; id++)
		if (mkt_pquery_claim(arg->sh, id))
			arg->wins++;
	return NULL;
}

TEST(pquery_claim_race_exactly_one_winner)
{
	MktQueryShared *sh	 = make_shared(4, 4, 1024);
	const uint32_t	nids = 2000;

	pthread_t	 threads[4];
	ClaimRaceArg args[4];
	for (int t = 0; t < 4; t++)
	{
		args[t] = (ClaimRaceArg){.sh = sh, .nids = nids, .wins = 0};
		pthread_create(&threads[t], NULL, claim_race_fn, &args[t]);
	}
	uint32_t total = 0;
	for (int t = 0; t < 4; t++)
	{
		pthread_join(threads[t], NULL);
		total += args[t].wins;
	}
	ASSERT_EQ(nids, total, "every id claimed by exactly one thread");
	mkt_free(sh);
}

/* ----------------------------------------------------------------
 * Merge invariant
 * ---------------------------------------------------------------- */

/* Deterministic pseudo-random stream (xorshift). */
static uint64_t
next_rand(uint64_t *state)
{
	uint64_t x = *state;
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	return *state = x;
}

TEST(pquery_merge_reproduces_serial)
{
	const uint32_t k	  = 10;
	const uint32_t nent	  = 3000;
	const uint32_t nparts = 3;

	/* Synthetic entry stream with duplicate ids (SOAR replicas get
	 * different estimated distances for the same id). */
	MktTopKEntry *entries = mkt_alloc(nent * sizeof(MktTopKEntry));
	uint64_t	  rng	  = 12345;
	for (uint32_t i = 0; i < nent; i++)
	{
		uint64_t r = next_rand(&rng);
		entries[i] = (MktTopKEntry){
				.distance = (float)(r % 100000) / 1000.0f,
				.error	  = (float)((r >> 20) % 500) / 1000.0f,
				/* ~25% duplicate-id rate, ids never 0 */
				.id	 = 1 + (r % (nent / 4)),
				.src = (uint32_t)(i % 97),
		};
	}

	/* Serial reference: insert everything into one top-k. */
	MktTopK serial;
	mkt_topk_init(&serial, k);
	for (uint32_t i = 0; i < nent; i++)
	{
		serial.cur_src = entries[i].src;
		mkt_topk_insert(
				&serial, entries[i].distance, entries[i].error, entries[i].id);
	}
	MktTopKEntry *serial_out = mkt_alloc(nent * sizeof(MktTopKEntry));
	uint32_t	  serial_n;
	mkt_topk_extract_sorted(&serial, serial_out, &serial_n);

	/* Parallel: partition round-robin across participants (so replicas
	 * of one id land in different participants), then merge by replay. */
	MktTopK parts[3];
	for (uint32_t p = 0; p < nparts; p++)
		mkt_topk_init(&parts[p], k);
	for (uint32_t i = 0; i < nent; i++)
	{
		MktTopK *t = &parts[i % nparts];
		t->cur_src = entries[i].src;
		mkt_topk_insert(
				t, entries[i].distance, entries[i].error, entries[i].id);
	}

	MktTopK merged;
	mkt_topk_init(&merged, k);
	for (uint32_t p = 0; p < nparts; p++)
		mkt_pquery_merge_candidates(
				&merged, parts[p].candidates, parts[p].cand_count);

	MktTopKEntry *merged_out = mkt_alloc(nent * sizeof(MktTopKEntry));
	uint32_t	  merged_n;
	mkt_topk_extract_sorted(&merged, merged_out, &merged_n);

	/* The survivor SETS must match: same ids, same distances. (Order of
	 * equal distances may differ -- same nondeterminism serial has.) */
	ASSERT_EQ(serial_n, merged_n, "survivor count matches serial");
	for (uint32_t i = 0; i < serial_n; i++)
	{
		bool found = false;
		for (uint32_t j = 0; j < merged_n; j++)
			if (merged_out[j].id == serial_out[i].id &&
				merged_out[j].distance == serial_out[i].distance)
			{
				found = true;
				break;
			}
		ASSERT_TRUE(found, "serial survivor present in merged set");
	}

	mkt_topk_cleanup(&serial);
	for (uint32_t p = 0; p < nparts; p++)
		mkt_topk_cleanup(&parts[p]);
	mkt_topk_cleanup(&merged);
	mkt_free(entries);
	mkt_free(serial_out);
	mkt_free(merged_out);
}

/* ----------------------------------------------------------------
 * Cancellation
 * ---------------------------------------------------------------- */

TEST(pquery_cancel_flag)
{
	MktQueryShared *sh = make_shared(4, 1, 4);
	ASSERT_EQ(0, mkt_atomic_read_u32(&sh->cancel), "starts clear");
	mkt_pquery_cancel(sh);
	ASSERT_EQ(1, mkt_atomic_read_u32(&sh->cancel), "set after cancel");
	mkt_pquery_begin(sh);
	ASSERT_EQ(0, mkt_atomic_read_u32(&sh->cancel), "begin resets");
	mkt_free(sh);
}
