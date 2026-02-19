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

#include "mkt_types.h"

#ifdef MKT_STANDALONE
#include "core/pg_compat.h"
#else
#include <postgres.h>

#include <storage/bufpage.h>
#include <storage/itemptr.h>
#endif

typedef struct MktStorage MktStorage;

typedef struct MktStorageOps
{
	/* Read path */
	Page (*read_page)(MktStorage *self, BlockNumber blkno);
	void (*release_page)(MktStorage *self, BlockNumber blkno);

	/* Write path (durability/WAL is internal to implementation) */
	Page (*write_page)(MktStorage *self, BlockNumber blkno);
	Page (*new_page)(MktStorage *self, BlockNumber *blkno_out);
	void (*commit_page)(MktStorage *self, BlockNumber blkno);

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
	 *   query           — opaque query vector (Datum)
	 *   dim             — vector dimension
	 *   tids[count]     — candidate TIDs
	 *   distances[count] — approximate distances
	 *   errors[count]   — error bounds (0 = already exact)
	 *
	 * Output:
	 *   out_indices[keep]   — indices into input arrays
	 *   out_distances[keep] — exact distances
	 *
	 * Returns: result count (<= keep), sorted by distance asc.
	 * NULL pointer means reranking is not supported.
	 */
	uint32_t (*rerank)(
			MktStorage			  *self,
			Datum				   query,
			Dimension			   dim,
			const ItemPointerData *tids,
			const Distance		  *distances,
			const Distance		  *errors,
			uint32_t			   count,
			uint32_t			   keep,
			uint32_t			  *out_indices,
			Distance			  *out_distances);
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

static inline uint32_t
mkt_storage_rerank(
		MktStorage			  *s,
		Datum				   query,
		Dimension			   dim,
		const ItemPointerData *tids,
		const Distance		  *distances,
		const Distance		  *errors,
		uint32_t			   count,
		uint32_t			   keep,
		uint32_t			  *out_indices,
		Distance			  *out_distances)
{
	return s->ops->rerank(
			s,
			query,
			dim,
			tids,
			distances,
			errors,
			count,
			keep,
			out_indices,
			out_distances);
}

#endif /* MKT_STORAGE_H */
