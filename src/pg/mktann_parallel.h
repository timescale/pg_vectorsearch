/*
 * mktann_parallel.h - Parallel index scan shared area (PostgreSQL)
 *
 * The DSM area behind ParallelIndexScanDesc.ps_offset_am. Embeds the
 * backend-neutral MktQueryShared first (probe list, work cursor, claim
 * table; see index/query_parallel.h) and adds the PG-only pieces: the
 * one-shot probe-list rendezvous (spinlock + condition variable, btree
 * seize-style) and the leader-published rotation params that spare each
 * fresh worker process the O(dim^3) matrix construction.
 */

#ifndef MKTANN_PARALLEL_H
#define MKTANN_PARALLEL_H

#include <postgres.h>

#include <access/relscan.h>
#include <storage/condition_variable.h>
#include <storage/spin.h>
#include <utils/rel.h>

#include "index/query_parallel.h"
#include "quant/rabitq.h"

/* Probe-list rendezvous states. Exactly one participant moves
 * PENDING -> DESCENDING, runs the descent, publishes the probe list and
 * broadcasts READY; on error it broadcasts FAILED and every other
 * participant errors out (standard parallel-query error propagation
 * would kill them anyway; FAILED only shortcuts the window). */
#define MKTANN_PSCAN_PENDING	0
#define MKTANN_PSCAN_DESCENDING 1
#define MKTANN_PSCAN_READY		2
#define MKTANN_PSCAN_FAILED		3

typedef struct MktannParallelScan
{
	MktQueryShared core; /* must be first */

	slock_t			  mutex; /* guards state */
	ConditionVariable cv;	 /* broadcast on READY/FAILED */
	uint32			  state;

	/* Frozen at init from the leader's executor-startup GUCs; every
	 * participant clamps to these (estimate and init run in the same
	 * startup with the same GUC values, so they cannot disagree; a
	 * cached generic plan re-executed under different GUCs is clamped
	 * rather than resized). */
	uint32 sized_pool; /* per-participant rerank/claim budget */

	/* Leader-published rotation params (flat RaBitQParams copy; dim and
	 * seed are embedded in the struct for worker-side validation). */
	bool   params_ready;
	uint64 params_off; /* from the start of this struct */
} MktannParallelScan;

static inline MktannParallelScan *
mktann_parallel_area(IndexScanDesc scan)
{
	return (MktannParallelScan *)((char *)scan->parallel_scan +
								  scan->parallel_scan->ps_offset_am);
}

static inline RaBitQParams *
mktann_parallel_params(MktannParallelScan *pscan)
{
	return (RaBitQParams *)((char *)pscan + pscan->params_off);
}

/* Index AM callbacks (amroutine). */
Size mktann_estimateparallelscan(Relation index, int nkeys, int norderbys);
void mktann_initparallelscan(void *target);
void mktann_parallelrescan(IndexScanDesc scan);

#endif /* MKTANN_PARALLEL_H */
