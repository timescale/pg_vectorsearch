/*
 * topk.c - Top-K collection for PostgreSQL builds
 *
 * Uses PG's pairing heap for the threshold max-heap and palloc
 * for memory. Same API as the standalone implementation in
 * algo/topk_standalone.c but uses PG infrastructure.
 */

#include <postgres.h>

#include <lib/pairingheap.h>

#include "algo/topk.h"

#define MKT_TOPK_INITIAL_CAP_MIN 32

/* ----------------------------------------------------------------
 * Threshold heap node — embeds pairingheap_node
 * ---------------------------------------------------------------- */

typedef struct UBNode
{
	pairingheap_node ph_node;
	Distance		 ub;
} UBNode;

/* Max-heap comparator: largest upper bound at root */
static int
ub_max_cmp(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	Distance da = pairingheap_const_container(UBNode, ph_node, a)->ub;
	Distance db = pairingheap_const_container(UBNode, ph_node, b)->ub;

	(void)arg;
	if (da > db)
		return 1;
	if (da < db)
		return -1;
	return 0;
}

/* ----------------------------------------------------------------
 * Init / cleanup / create / destroy
 * ---------------------------------------------------------------- */

void
mkt_topk_init(MktTopK *topk, uint32_t k)
{
	topk->k		   = k;
	topk->ub_heap  = pairingheap_allocate(ub_max_cmp, NULL);
	topk->ub_nodes = palloc(k * sizeof(UBNode));
	topk->ub_count = 0;

	uint32_t cap = k * 2;
	if (cap < MKT_TOPK_INITIAL_CAP_MIN)
		cap = MKT_TOPK_INITIAL_CAP_MIN;
	topk->candidates	= palloc(cap * sizeof(MktTopKEntry));
	topk->cand_count	= 0;
	topk->cand_capacity = cap;
}

void
mkt_topk_cleanup(MktTopK *topk)
{
	if (topk == NULL)
		return;
	if (topk->ub_heap)
		pairingheap_free(topk->ub_heap);
	if (topk->ub_nodes)
		pfree(topk->ub_nodes);
	if (topk->candidates)
		pfree(topk->candidates);
	topk->ub_heap	 = NULL;
	topk->ub_nodes	 = NULL;
	topk->candidates = NULL;
}

MktTopK *
mkt_topk_create(uint32_t k)
{
	MktTopK *topk = palloc(sizeof(MktTopK));
	mkt_topk_init(topk, k);
	return topk;
}

void
mkt_topk_destroy(MktTopK *topk)
{
	if (topk == NULL)
		return;
	mkt_topk_cleanup(topk);
	pfree(topk);
}

void
mkt_topk_reset(MktTopK *topk)
{
	pairingheap_reset((pairingheap *)topk->ub_heap);
	topk->ub_count	 = 0;
	topk->cand_count = 0;
}

/* ----------------------------------------------------------------
 * Threshold
 * ---------------------------------------------------------------- */

Distance
mkt_topk_threshold(const MktTopK *topk)
{
	if (topk->ub_count < topk->k)
		return INFINITY;

	pairingheap_node *max = pairingheap_first((pairingheap *)topk->ub_heap);
	return pairingheap_container(UBNode, ph_node, max)->ub;
}

/* ----------------------------------------------------------------
 * Insert
 * ---------------------------------------------------------------- */

void
mkt_topk_insert(MktTopK *topk, Distance distance, Distance error, uint64_t id)
{
	Distance lb = distance - error;
	Distance ub = distance + error;

	/* Prune: lower bound exceeds threshold */
	if (lb >= mkt_topk_threshold(topk))
		return;

	/* Update threshold heap */
	pairingheap *ph	   = topk->ub_heap;
	UBNode		*nodes = topk->ub_nodes;

	if (topk->ub_count < topk->k)
	{
		UBNode *node = &nodes[topk->ub_count];
		node->ub	 = ub;
		pairingheap_add(ph, &node->ph_node);
		topk->ub_count++;
	}
	else
	{
		pairingheap_node *max_node = pairingheap_first(ph);
		UBNode *max = pairingheap_container(UBNode, ph_node, max_node);
		if (ub < max->ub)
		{
			pairingheap_remove_first(ph);
			max->ub = ub;
			pairingheap_add(ph, &max->ph_node);
		}
	}

	/* Grow candidate buffer if needed */
	if (topk->cand_count == topk->cand_capacity)
	{
		uint32_t new_cap = topk->cand_capacity * 2;
		topk->candidates =
				repalloc(topk->candidates, new_cap * sizeof(MktTopKEntry));
		topk->cand_capacity = new_cap;
	}

	/* Append to candidate buffer */
	topk->candidates[topk->cand_count++] = (MktTopKEntry){
			.distance = distance,
			.error	  = error,
			.id		  = id,
	};
}

/* ----------------------------------------------------------------
 * Extract
 * ---------------------------------------------------------------- */

static int
cmp_by_distance(const void *a, const void *b)
{
	const MktTopKEntry *ea = (const MktTopKEntry *)a;
	const MktTopKEntry *eb = (const MktTopKEntry *)b;
	if (ea->distance < eb->distance)
		return -1;
	if (ea->distance > eb->distance)
		return 1;
	return 0;
}

static void
topk_extract(
		MktTopK *topk, MktTopKEntry *results, uint32_t *count_out, bool sorted)
{
	Distance threshold = mkt_topk_threshold(topk);

	/* Filter stale candidates and copy survivors to results */
	uint32_t out = 0;
	for (uint32_t i = 0; i < topk->cand_count; i++)
	{
		Distance lb = topk->candidates[i].distance - topk->candidates[i].error;
		if (lb <= threshold)
			results[out++] = topk->candidates[i];
	}

	if (sorted && out > 1)
		qsort(results, out, sizeof(MktTopKEntry), cmp_by_distance);

	*count_out = out;
	mkt_topk_reset(topk);
}

void
mkt_topk_extract(MktTopK *topk, MktTopKEntry *results, uint32_t *count_out)
{
	topk_extract(topk, results, count_out, false);
}

void
mkt_topk_extract_sorted(
		MktTopK *topk, MktTopKEntry *results, uint32_t *count_out)
{
	topk_extract(topk, results, count_out, true);
}
