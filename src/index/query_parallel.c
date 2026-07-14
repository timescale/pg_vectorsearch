/*
 * query_parallel.c - Backend-neutral core for parallel query scans
 *
 * See query_parallel.h for the model. This file owns the semantics of
 * the shared work cursor, the SOAR claim table, the participant scan
 * loop, and the leader-merge; the backends own memory, publication
 * ordering, and participant lifecycles.
 */

#include "mkt_config.h"

#include <string.h>

#ifndef MKT_STANDALONE
#include <postgres.h>

#include <miscadmin.h>
#endif

#include "core/memory.h"
#include "index/query_parallel.h"

/* Ramp-down chunking bounds (see mkt_pquery_next_batch). */
#define MKT_PQUERY_BATCH_MAX 8

/* Claim-table floor: keeps small configurations collision-free and the
 * table trivially cheap (32 KB of zeroed slots). */
#define MKT_PQUERY_CLAIM_SLOTS_MIN 4096

static uint32_t
next_pow2(uint32_t v)
{
	uint32_t p = 1;
	while (p < v)
		p <<= 1;
	return p;
}

uint32_t
mkt_pquery_claim_slots(uint32_t nparticipants, uint32_t pool)
{
	uint64_t want = (uint64_t)2 * nparticipants * pool;
	if (want < MKT_PQUERY_CLAIM_SLOTS_MIN)
		want = MKT_PQUERY_CLAIM_SLOTS_MIN;
	if (want > UINT32_MAX / 2)
		want = UINT32_MAX / 2;
	return next_pow2((uint32_t)want);
}

void
mkt_pquery_shared_init(
		MktQueryShared *shared,
		uint32_t		max_nscan,
		uint32_t		nparticipants,
		uint32_t		nclaim_slots,
		uint64_t		heads_off,
		uint64_t		claims_off)
{
	shared->max_nscan	  = max_nscan;
	shared->nparticipants = nparticipants > 0 ? nparticipants : 1;
	shared->claim_mask	  = nclaim_slots - 1;
	shared->nprobes		  = 0;
	shared->heads_off	  = heads_off;
	shared->claims_off	  = claims_off;

	mkt_atomic_init_u32(&shared->cursor, 0);
	mkt_atomic_init_u32(&shared->cancel, 0);
	mkt_atomic_init_u32(&shared->claims_full, 0);

	mkt_atomic_uint64 *claims = mkt_pquery_claims(shared);
	for (uint32_t i = 0; i <= shared->claim_mask; i++)
		mkt_atomic_init_u64(&claims[i], 0);
}

void
mkt_pquery_begin(MktQueryShared *shared)
{
	shared->nprobes = 0;
	mkt_atomic_init_u32(&shared->cursor, 0);
	mkt_atomic_init_u32(&shared->cancel, 0);
	mkt_atomic_init_u32(&shared->claims_full, 0);

	mkt_atomic_uint64 *claims = mkt_pquery_claims(shared);
	for (uint32_t i = 0; i <= shared->claim_mask; i++)
		mkt_atomic_init_u64(&claims[i], 0);
}

void
mkt_pquery_publish_probes(
		MktQueryShared *shared, const BlockNumber *heads, uint32_t n)
{
	if (n > shared->max_nscan)
		n = shared->max_nscan;
	memcpy(mkt_pquery_heads(shared), heads, (size_t)n * sizeof(BlockNumber));
	shared->nprobes = n;
}

uint32_t
mkt_pquery_next_batch(MktQueryShared *shared, uint32_t *start)
{
	uint32_t nprobes = shared->nprobes;

	/* Size the batch from the CURRENT position: large while plenty of
	 * work remains, ramping to single clusters near the tail so the
	 * stragglers spread across participants (same idea as PostgreSQL's
	 * parallel block-range allocation). The read is racy against
	 * concurrent claims, which only makes a tail batch slightly larger
	 * than ideal -- never incorrect. */
	uint32_t pos	   = mkt_atomic_read_u32(&shared->cursor);
	uint32_t remaining = pos < nprobes ? nprobes - pos : 0;
	uint32_t chunk	   = remaining / (4 * shared->nparticipants);
	if (chunk < 1)
		chunk = 1;
	if (chunk > MKT_PQUERY_BATCH_MAX)
		chunk = MKT_PQUERY_BATCH_MAX;

	uint32_t first = mkt_atomic_fetch_add_u32(&shared->cursor, chunk);
	if (first >= nprobes)
		return 0;

	*start		 = first;
	uint32_t end = first + chunk;
	if (end > nprobes)
		end = nprobes;
	return end - first;
}

/*
 * Backend interrupt point between clusters. PostgreSQL: a pending
 * interrupt longjmps out of the scan (standard error propagation kills
 * the parallel query). Standalone: interrupts arrive only through the
 * cooperative cancel flag, checked by the caller.
 */
static inline void
pquery_check_interrupt(void)
{
#ifndef MKT_STANDALONE
	CHECK_FOR_INTERRUPTS();
#endif
}

bool
mkt_pquery_scan(
		MktQueryState  *qs,
		MktQueryShared *shared,
		MktDistanceMode mode,
		MktTopK		   *topk,
		MktQueryStats  *stats)
{
	const BlockNumber *heads = mkt_pquery_heads(shared);
	uint32_t		   start;
	uint32_t		   n;

	while ((n = mkt_pquery_next_batch(shared, &start)) > 0)
	{
		for (uint32_t r = start; r < start + n; r++)
		{
			if (mkt_atomic_read_u32(&shared->cancel) != 0)
				return false;
			pquery_check_interrupt();

			mkt_query_scan_cluster(qs, heads[r], r, mode, topk, stats);
		}
	}

	return mkt_atomic_read_u32(&shared->cancel) == 0;
}

bool
mkt_pquery_claim(MktQueryShared *shared, uint64_t id)
{
	mkt_atomic_uint64 *claims = mkt_pquery_claims(shared);
	uint32_t		   mask	  = shared->claim_mask;

	/* Open addressing, linear probe. The table is sized at 2x the
	 * worst-case claim volume, so probes are short. A full table means
	 * a pathological configuration drift (e.g. a cached plan launched
	 * with more participants than the table was sized for): prefer
	 * emitting a potential duplicate over dropping a result, and record
	 * the overflow so the backend can warn once. */
	uint32_t slot = (uint32_t)(id * UINT64_C(0x9E3779B97F4A7C15) >> 32) & mask;
	for (uint32_t i = 0; i <= mask; i++)
	{
		uint64_t seen = mkt_atomic_read_u64(&claims[slot]);
		if (seen == id)
			return false; /* someone (maybe us) already claimed it */
		if (seen == 0)
		{
			uint64_t expected = 0;
			if (mkt_atomic_cas_u64(&claims[slot], &expected, id))
				return true;
			if (expected == id)
				return false; /* lost the race to a claimer of the same id */
							  /* Slot taken by another id; keep probing. */
		}
		slot = (slot + 1) & mask;
	}

	mkt_atomic_write_u32(&shared->claims_full, 1);
	return true;
}

uint32_t
mkt_pquery_filter_claims(
		MktQueryShared *shared, MktTopKEntry *cands, uint32_t n)
{
	uint32_t kept = 0;

	for (uint32_t i = 0; i < n; i++)
	{
		if (mkt_pquery_claim(shared, cands[i].id))
			cands[kept++] = cands[i];
	}
	return kept;
}

void
mkt_pquery_cancel(MktQueryShared *shared)
{
	mkt_atomic_write_u32(&shared->cancel, 1);
}

void
mkt_pquery_merge_candidates(
		MktTopK *dst, const MktTopKEntry *cands, uint32_t n)
{
	for (uint32_t i = 0; i < n; i++)
		mkt_topk_insert_entry(dst, &cands[i]);
}
