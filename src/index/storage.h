/*
 * storage.h - Unified I/O abstraction for index page access
 *
 * MktStorageOps is a vtable for page and vector I/O. MktStorage is a
 * base struct containing an ops pointer that implementations embed as
 * their first member, adding implementation-specific context fields.
 *
 * WAL logging and durability are internal to each implementation —
 * callers just see read/release/write/commit.
 *
 * Standalone: embed in a struct with an array of malloc'd 8KB buffers
 * PG mode:    embed in MktannStorage with Relation + buffer cache
 * Cloud:      embed with remote block store handle (future)
 *
 * Caller pattern:
 *
 *   // Search (read-only hot path)
 *   Page page = mkt_storage_read_page(s, blkno);
 *   // ... scan page entries ...
 *   mkt_storage_release_page(s, blkno);
 *
 *   // Build (write path)
 *   Page page = mkt_storage_new_page(s, &blkno);
 *   mkt_centroid_page_init(page, level);
 *   mkt_centroid_page_add(page, dim, ...);
 *   mkt_storage_commit_page(s, blkno);
 */

#ifndef MKT_STORAGE_H
#define MKT_STORAGE_H

#include "algo/topk.h"
#include "core/types.h"

#ifdef MKT_STANDALONE
#include "standalone/pg_compat.h"
#else
#include <postgres.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <storage/bufpage.h>
#pragma GCC diagnostic pop
#include <storage/itemptr.h>
#endif

typedef struct MktStorage MktStorage;

typedef struct MktStorageOps
{
	/* Read path */
	Page (*read_page)(MktStorage *self, BlockNumber blkno);
	void (*release_page)(MktStorage *self, BlockNumber blkno);

	/*
	 * Advisory async prefetch: hint that blkno will be read soon so the
	 * implementation can start the I/O (e.g. posix_fadvise via
	 * PrefetchBuffer). Best-effort -- a later read_page still performs the
	 * actual read. NULL = unsupported; callers reach it through
	 * mkt_storage_prefetch(), which tolerates a NULL op and an invalid block.
	 */
	void (*prefetch)(MktStorage *self, BlockNumber blkno);

	/* Write path (durability/WAL is internal to implementation) */
	Page (*write_page)(MktStorage *self, BlockNumber blkno);
	Page (*new_page)(MktStorage *self, BlockNumber *blkno_out);
	void (*commit_page)(MktStorage *self, BlockNumber blkno);

	/*
	 * Bulk extend: pre-allocate npages contiguous pages.
	 * Returns the first block number. NULL = not supported.
	 */
	BlockNumber (*extend)(MktStorage *self, uint32_t npages);

	/*
	 * Rerank candidates with exact distances.
	 *
	 * Fetches full-precision vectors and computes exact L2 for
	 * candidates that can't be pruned by error bounds. The
	 * implementation owns I/O strategy, vector extraction,
	 * and distance computation.
	 *
	 * query and stored vectors are opaque Datums — the storage
	 * knows the vector type and handles extraction + dispatch.
	 *
	 * Input:
	 *   query              — query vector (float array)
	 *   dim                — vector dimension
	 *   candidates[count]  — approximate results (id encodes TID)
	 *
	 * Output:
	 *   out_indices[keep]   — indices into candidates
	 *   out_distances[keep] — exact distances
	 *
	 * Returns: result count (<= keep), sorted by distance asc.
	 * NULL pointer means reranking is not supported.
	 */
	uint32_t (*rerank)(
			MktStorage		   *self,
			const float		   *query,
			Dimension			dim,
			const MktTopKEntry *candidates,
			uint32_t			count,
			uint32_t			keep,
			uint32_t		   *out_indices,
			Distance		   *out_distances);
} MktStorageOps;

struct MktStorage
{
	const MktStorageOps *ops;
};

/* ----------------------------------------------------------------
 * Inline wrappers — hide s->ops->fn(s, ...) behind clean calls
 * ---------------------------------------------------------------- */

static inline Page
mkt_storage_read_page(MktStorage *s, BlockNumber blkno)
{
	return s->ops->read_page(s, blkno);
}

static inline void
mkt_storage_release_page(MktStorage *s, BlockNumber blkno)
{
	s->ops->release_page(s, blkno);
}

static inline void
mkt_storage_prefetch(MktStorage *s, BlockNumber blkno)
{
	if (s != NULL && s->ops->prefetch != NULL && blkno != InvalidBlockNumber)
		s->ops->prefetch(s, blkno);
}

static inline Page
mkt_storage_write_page(MktStorage *s, BlockNumber blkno)
{
	return s->ops->write_page(s, blkno);
}

static inline Page
mkt_storage_new_page(MktStorage *s, BlockNumber *blkno_out)
{
	return s->ops->new_page(s, blkno_out);
}

static inline void
mkt_storage_commit_page(MktStorage *s, BlockNumber blkno)
{
	s->ops->commit_page(s, blkno);
}

static inline BlockNumber
mkt_storage_extend(MktStorage *s, uint32_t npages)
{
	if (s->ops->extend != NULL)
		return s->ops->extend(s, npages);
	return InvalidBlockNumber;
}

static inline uint32_t
mkt_storage_rerank(
		MktStorage		   *s,
		const float		   *query,
		Dimension			dim,
		const MktTopKEntry *candidates,
		uint32_t			count,
		uint32_t			keep,
		uint32_t		   *out_indices,
		Distance		   *out_distances)
{
	return s->ops->rerank(
			s,
			query,
			dim,
			candidates,
			count,
			keep,
			out_indices,
			out_distances);
}

#endif /* MKT_STORAGE_H */
