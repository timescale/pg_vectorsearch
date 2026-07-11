/*
 * mktann_storage.c - PG buffer cache MktStorage implementation
 *
 * Read path:  ReadBuffer + LockBuffer(SHARE) -> return page
 *             UnlockReleaseBuffer on release
 *
 * Write path: ReadBufferExtended(P_NEW) + exclusive lock
 *             GenericXLogStart/RegisterBuffer/Finish for WAL
 *             Commit releases buffer
 */

#include <postgres.h>

#include <access/generic_xlog.h>
#include <access/heapam.h>
#include <access/htup_details.h>
#include <access/tableam.h>
#include <catalog/pg_am_d.h>
#include <executor/tuptable.h>
#include <storage/bufmgr.h>
#include <storage/read_stream.h>
#include <utils/memutils.h>
#include <utils/snapmgr.h>

#include "algo/distance.h"
#include "algo/topk.h"
#include "algo/vecops.h"
#include "index/posting_page.h"
#include "mkt_pg.h"
#include "mktann_storage.h"

/* Downcast from base to concrete type */
#define PG_STORAGE(self) ((MktannStorage *)(self))

/* ----------------------------------------------------------------
 * Backend-local buffer-id cache (mkt.recent_buffers)
 *
 * ~22% of warm query CPU is BufTableLookup hash probes inside
 * ReadBuffer, for index pages that essentially never leave
 * shared_buffers. Remember the buffer id per block (per backend) and
 * re-pin it via ReadRecentBuffer, which validates the tag and pins
 * without touching the buffer mapping table. A stale id (page evicted
 * or buffer reused) just fails validation and falls back to
 * ReadBuffer, which refreshes the cached id — correctness never
 * depends on the cache.
 *
 * The cache stores 4-byte buffer ids, not page data, so it does not
 * duplicate shared_buffers (~11 MB per backend for a 21 GB index).
 * One relation is cached at a time per backend; switching indexes
 * swaps the cache.
 * ---------------------------------------------------------------- */
static bool			  g_recent_buffers = true;
static RelFileLocator g_bufcache_locator; /* zeroed = invalid */
static Buffer		 *g_bufcache	 = NULL;
static BlockNumber	  g_bufcache_len = 0;
static MemoryContext  g_bufcache_ctx = NULL;

void
mktann_storage_set_recent_buffers(bool enabled)
{
	g_recent_buffers = enabled;
}

static inline Buffer *
bufcache_slot(Relation index, BlockNumber blkno)
{
	const RelFileLocator *loc = &index->rd_locator;

	if (unlikely(
				g_bufcache == NULL ||
				!RelFileLocatorEquals(g_bufcache_locator, *loc)))
	{
		BlockNumber nblocks = RelationGetNumberOfBlocks(index);

		/* Headroom so post-build inserts don't invalidate the cache. */
		nblocks += nblocks / 8 + 1024;

		/* Backend-lifetime cache data belongs under CacheMemoryContext
		 * (as a named child, so it is attributed in
		 * pg_backend_memory_contexts rather than hiding in the top
		 * context). */
		if (g_bufcache_ctx == NULL)
			g_bufcache_ctx = AllocSetContextCreate(
					CacheMemoryContext,
					"mktann recent-buffers cache",
					ALLOCSET_START_SMALL_SIZES);

		if (g_bufcache != NULL)
			pfree(g_bufcache);
		g_bufcache = MemoryContextAlloc(
				g_bufcache_ctx, (Size)nblocks * sizeof(Buffer));
		for (BlockNumber i = 0; i < nblocks; i++)
			g_bufcache[i] = InvalidBuffer;
		g_bufcache_len	   = nblocks;
		g_bufcache_locator = *loc;
	}

	if (unlikely(blkno >= g_bufcache_len))
		return NULL;
	return &g_bufcache[blkno];
}

/* ----------------------------------------------------------------
 * Read path
 * ---------------------------------------------------------------- */

/* Buffer-id cache effectiveness counters (diagnostic; exposed via
 * mkt_routing_stats). */
uint64_t mkt_bufcache_hits;
uint64_t mkt_bufcache_cold;
uint64_t mkt_bufcache_stale;

static Page
pg_read_page(MktStorage *self, BlockNumber blkno)
{
	MktannStorage *s = PG_STORAGE(self);
	Buffer		   buf;

	if (g_recent_buffers)
	{
		Buffer *slot = bufcache_slot(s->index, blkno);

		if (slot != NULL && *slot != InvalidBuffer &&
			ReadRecentBuffer(s->index->rd_locator, MAIN_FORKNUM, blkno, *slot))
		{
			buf = *slot;
			mkt_bufcache_hits++;
		}
		else
		{
			/* Diagnostic: distinguish never-populated slots (first
			 * touch) from stale ids (eviction churn). */
			if (slot == NULL || *slot == InvalidBuffer)
				mkt_bufcache_cold++;
			else
				mkt_bufcache_stale++;
			buf = ReadBuffer(s->index, blkno);
			if (slot != NULL)
				*slot = buf;
		}
	}
	else
		buf = ReadBuffer(s->index, blkno);

	LockBuffer(buf, BUFFER_LOCK_SHARE);
	s->cur_buf = buf;
	s->read_count++;

	return BufferGetPage(buf);
}

static void
pg_release_page(MktStorage *self, BlockNumber blkno)
{
	MktannStorage *s = PG_STORAGE(self);

	(void)blkno;
	UnlockReleaseBuffer(s->cur_buf);
	s->cur_buf = InvalidBuffer;
}

/* ----------------------------------------------------------------
 * Write path
 * ---------------------------------------------------------------- */

static Page
pg_write_page(MktStorage *self, BlockNumber blkno)
{
	MktannStorage *s = PG_STORAGE(self);

	Buffer buf = ReadBuffer(s->index, blkno);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	s->cur_buf = buf;

	return BufferGetPage(buf);
}

static Page
pg_new_page(MktStorage *self, BlockNumber *blkno_out)
{
	MktannStorage *s = PG_STORAGE(self);

	Buffer buf = ReadBufferExtended(
			s->index, MAIN_FORKNUM, P_NEW, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

	*blkno_out = BufferGetBlockNumber(buf);
	s->cur_buf = buf;

	return BufferGetPage(buf);
}

static void
pg_commit_page(MktStorage *self, BlockNumber blkno)
{
	MktannStorage *s = PG_STORAGE(self);

	(void)blkno;

	if (s->build_mode)
	{
		/* During index build, skip per-page WAL logging.
		 * log_newpage_range() at the end covers all pages. */
		MarkBufferDirty(s->cur_buf);
	}
	else
	{
		/*
		 * meerkat pages store their data in the content area between pd_lower
		 * and pd_upper — the region PostgreSQL treats as the free "hole" and
		 * omits from standard-layout full-page images (GenericXLog assumes the
		 * standard layout). Cover the hole (pd_lower = pd_upper) before
		 * logging so the entire page is preserved; pd_lower is otherwise
		 * unused by meerkat (scans locate data via PageGetContents /
		 * pd_special).
		 */
		PageHeader ph = (PageHeader)BufferGetPage(s->cur_buf);
		ph->pd_lower  = ph->pd_upper;

		GenericXLogState *state = GenericXLogStart(s->index);
		GenericXLogRegisterBuffer(state, s->cur_buf, GENERIC_XLOG_FULL_IMAGE);
		GenericXLogFinish(state);
	}

	UnlockReleaseBuffer(s->cur_buf);
	s->cur_buf = InvalidBuffer;
}

/* ----------------------------------------------------------------
 * Bulk extend: pre-allocate contiguous pages
 * ---------------------------------------------------------------- */

static BlockNumber
pg_extend(MktStorage *self, uint32_t npages)
{
	MktannStorage *s = PG_STORAGE(self);

	BlockNumber start	  = InvalidBlockNumber;
	uint32_t	remaining = npages;

	while (remaining > 0)
	{
		uint32_t batch	 = Min(remaining, 512);
		Buffer	*buffers = palloc(batch * sizeof(Buffer));
		uint32_t got	 = 0;

		BlockNumber batch_start = ExtendBufferedRelBy(
				BMR_REL(s->index),
				MAIN_FORKNUM,
				NULL,
				0,
				batch,
				buffers,
				&got);

		if (start == InvalidBlockNumber)
			start = batch_start;

		for (uint32_t i = 0; i < got; i++)
			ReleaseBuffer(buffers[i]);

		pfree(buffers);
		remaining -= got;
	}

	return start;
}

/* ----------------------------------------------------------------
 * Rerank: fetch vectors from heap, compute exact L2
 * ---------------------------------------------------------------- */

/* Comparator for sorting candidate indices by TID block order */
static int
cmp_tid_order(const void *a, const void *b, void *arg)
{
	const MktTopKEntry *cands = arg;

	/* Encoded ids are (block << 16) | offset — strictly monotonic in
	 * (block, offset), so raw id comparison IS TID order; no need to
	 * decode both TIDs on every comparison. */
	uint64_t id_a = cands[*(const uint32_t *)a].id;
	uint64_t id_b = cands[*(const uint32_t *)b].id;

	if (id_a < id_b)
		return -1;
	if (id_a > id_b)
		return 1;
	return 0;
}

/*
 * pg_rerank - Rerank candidates with exact L2 distances.
 *
 * Fetches full-precision vectors from the heap table and computes
 * exact L2 distance for candidates with nonzero error. Candidates
 * are visited in TID block-number order to minimize buffer cache
 * thrash — sequential block access avoids re-pinning the same
 * buffer multiple times.
 *
 * SnapshotAny: medoid vectors are structural index data that happen
 * to live in the heap. They must be readable regardless of MVCC
 * state — an MVCC snapshot could fail if the row was DELETEd but
 * not yet VACUUMed, breaking centroid reranking. If the tuple is
 * physically reclaimed by VACUUM, the index is stale and needs
 * REINDEX.
 */
static uint32_t
pg_rerank(
		MktStorage		   *self,
		const float		   *query,
		Dimension			dim,
		const MktTopKEntry *candidates,
		uint32_t			count,
		uint32_t			keep,
		uint32_t		   *out_indices,
		Distance		   *out_distances)
{
	MktannStorage *s = PG_STORAGE(self);

	if (s->rel == NULL || count == 0)
		return 0;

	AttrNumber vec_attnum = s->index->rd_index->indkey.values[0];

	/* Sort by TID block order for sequential I/O */
	uint32_t *order = palloc(count * sizeof(uint32_t));
	for (uint32_t i = 0; i < count; i++)
		order[i] = i;
	qsort_arg(
			order, count, sizeof(uint32_t), cmp_tid_order, (void *)candidates);

	MktTopK topk;
	mkt_topk_init(&topk, keep);

	TupleTableSlot *slot = table_slot_create(s->rel, NULL);

	for (uint32_t i = 0; i < count; i++)
	{
		uint32_t idx = order[i];

		Distance d;
		if (candidates[idx].error == 0.0f)
		{
			d = candidates[idx].distance;
		}
		else
		{
			ItemPointerData tid = mkt_posting_decode_tid(candidates[idx].id);
			if (table_tuple_fetch_row_version(s->rel, &tid, SnapshotAny, slot))
			{
				bool  isnull;
				Datum val = slot_getattr(slot, vec_attnum, &isnull);
				if (!isnull)
				{
					MktVector *vec	= DatumGetMktVector(val);
					VectorRef  qref = {.data = query, .dim = dim};
					VectorRef  vref = {.data = vec->x, .dim = dim};
					d				= mkt_distance(qref, vref, s->metric);
				}
				else
				{
					d = candidates[idx].distance;
				}
				ExecClearTuple(slot);
			}
			else
			{
				d = candidates[idx].distance;
			}
		}

		mkt_topk_insert_unique(&topk, d, 0.0f, (uint64_t)idx);
	}

	ExecDropSingleTupleTableSlot(slot);

	MktTopKEntry *entries = palloc(topk.cand_count * sizeof(MktTopKEntry));
	uint32_t	  nresults;
	mkt_topk_extract_sorted_unique(&topk, entries, &nresults);

	for (uint32_t i = 0; i < nresults; i++)
	{
		out_indices[i]	 = (uint32_t)entries[i].id;
		out_distances[i] = entries[i].distance;
	}

	pfree(entries);
	mkt_topk_cleanup(&topk);
	pfree(order);

	return nresults;
}

/* ----------------------------------------------------------------
 * Heap AM rerank with read_stream
 *
 * Uses PostgreSQL's read_stream API for batched async I/O when
 * fetching heap tuples for reranking. The stream callback yields
 * block numbers in candidate order, and each buffer is processed
 * directly as it arrives.
 * ---------------------------------------------------------------- */

typedef struct RerankStreamState
{
	const MktTopKEntry *candidates;
	const uint32_t	   *order;
	MktTopK			   *topk;
	MktannStorage	   *storage;
	uint32_t			count;
	uint32_t			pos;
} RerankStreamState;

static BlockNumber
rerank_stream_cb(
		ReadStream *stream, void *callback_private_data, void *per_buffer_data)
{
	RerankStreamState *st = callback_private_data;

	while (st->pos < st->count)
	{
		uint32_t idx = st->order[st->pos++];

		/* Already exact — insert directly, no heap fetch */
		if (st->candidates[idx].error == 0.0f)
		{
			mkt_topk_insert_unique(
					st->topk,
					st->candidates[idx].distance,
					0.0f,
					(uint64_t)idx);
			continue;
		}

		*(uint32_t *)per_buffer_data = idx;
		ItemPointerData tid = mkt_posting_decode_tid(st->candidates[idx].id);
		return ItemPointerGetBlockNumber(&tid);
	}

	return InvalidBlockNumber;
}

static uint32_t
pg_rerank_readstream(
		MktStorage		   *self,
		const float		   *query,
		Dimension			dim,
		const MktTopKEntry *candidates,
		uint32_t			count,
		uint32_t			keep,
		uint32_t		   *out_indices,
		Distance		   *out_distances)
{
	MktannStorage *s = PG_STORAGE(self);

	if (s->rel == NULL || count == 0)
		return 0;

	AttrNumber vec_attnum = s->index->rd_index->indkey.values[0];

	uint32_t *order = palloc(count * sizeof(uint32_t));
	for (uint32_t i = 0; i < count; i++)
		order[i] = i;
	qsort_arg(
			order, count, sizeof(uint32_t), cmp_tid_order, (void *)candidates);

	MktTopK topk;
	mkt_topk_init(&topk, keep);

	RerankStreamState state = {
			.candidates = candidates,
			.order		= order,
			.topk		= &topk,
			.storage	= s,
			.count		= count,
			.pos		= 0,
	};

	TupleTableSlot *slot = MakeSingleTupleTableSlot(
			RelationGetDescr(s->rel), &TTSOpsBufferHeapTuple);

	ReadStream *stream = read_stream_begin_relation(
			READ_STREAM_DEFAULT,
			NULL,
			s->rel,
			MAIN_FORKNUM,
			rerank_stream_cb,
			&state,
			sizeof(uint32_t));

	void  *per_buffer_data;
	Buffer buf;

	while (BufferIsValid(
			buf = read_stream_next_buffer(stream, &per_buffer_data)))
	{
		uint32_t		idx = *(uint32_t *)per_buffer_data;
		ItemPointerData tid = mkt_posting_decode_tid(candidates[idx].id);

		Page		 page = BufferGetPage(buf);
		OffsetNumber off  = ItemPointerGetOffsetNumber(&tid);
		ItemId		 lp	  = PageGetItemId(page, off);

		Distance d = candidates[idx].distance;
		if (ItemIdIsNormal(lp))
		{
			HeapTupleData tuple;
			tuple.t_tableOid = RelationGetRelid(s->rel);
			tuple.t_data	 = (HeapTupleHeader)PageGetItem(page, lp);
			tuple.t_len		 = ItemIdGetLength(lp);
			ItemPointerCopy(&tid, &tuple.t_self);

			ExecStoreBufferHeapTuple(&tuple, slot, buf);

			bool  isnull;
			Datum val = slot_getattr(slot, vec_attnum, &isnull);
			if (!isnull)
			{
				MktVector *vec	= DatumGetMktVector(val);
				VectorRef  qref = {.data = query, .dim = dim};
				VectorRef  vref = {.data = vec->x, .dim = dim};
				d				= mkt_distance(qref, vref, s->metric);
			}
			ExecClearTuple(slot);
		}

		mkt_topk_insert_unique(&topk, d, 0.0f, (uint64_t)idx);
		ReleaseBuffer(buf);
	}

	read_stream_end(stream);
	ExecDropSingleTupleTableSlot(slot);

	MktTopKEntry *entries = palloc(topk.cand_count * sizeof(MktTopKEntry));
	uint32_t	  nresults;
	mkt_topk_extract_sorted_unique(&topk, entries, &nresults);

	for (uint32_t i = 0; i < nresults; i++)
	{
		out_indices[i]	 = (uint32_t)entries[i].id;
		out_distances[i] = entries[i].distance;
	}

	pfree(entries);
	mkt_topk_cleanup(&topk);
	pfree(order);

	return nresults;
}

/* ----------------------------------------------------------------
 * Static vtables
 * ---------------------------------------------------------------- */

static const MktStorageOps pg_storage_ops = {
		.read_page	  = pg_read_page,
		.release_page = pg_release_page,
		.write_page	  = pg_write_page,
		.new_page	  = pg_new_page,
		.commit_page  = pg_commit_page,
		.extend		  = pg_extend,
		.rerank		  = pg_rerank,
};

static const MktStorageOps pg_storage_readstream_ops = {
		.read_page	  = pg_read_page,
		.release_page = pg_release_page,
		.write_page	  = pg_write_page,
		.new_page	  = pg_new_page,
		.commit_page  = pg_commit_page,
		.extend		  = pg_extend,
		.rerank		  = pg_rerank_readstream,
};

/* ----------------------------------------------------------------
 * Initialization
 * ---------------------------------------------------------------- */

void
mktann_storage_init(
		MktannStorage *s, Relation index, Relation rel, DistanceMetric metric)
{
	if (rel != NULL && RelationGetForm(rel)->relam == HEAP_TABLE_AM_OID)
		s->base.ops = &pg_storage_readstream_ops;
	else
		s->base.ops = &pg_storage_ops;
	s->index	  = index;
	s->rel		  = rel;
	s->build_mode = false;
	s->cur_buf	  = InvalidBuffer;
	s->metric	  = metric;
	s->read_count = 0;
}

void
mktann_storage_set_rel(MktannStorage *s, Relation rel)
{
	s->rel = rel;
	if (rel != NULL && RelationGetForm(rel)->relam == HEAP_TABLE_AM_OID)
		s->base.ops = &pg_storage_readstream_ops;
}
