/*
 * mkt_shm_toc.h - Keyed shared-region table
 *
 * The parallel build publishes its shared regions (the build header, the
 * sample/centroid/assignment slots, the page queues, ...) into a table of
 * contents keyed by small integers; the leader allocates and inserts, the
 * workers look up. In a PostgreSQL build that table lives in a DSM segment and
 * is PostgreSQL's shm_toc. In a standalone (thread-based) build there is no
 * DSM — every thread already shares the heap — so this provides the same API
 * over a plain arena plus a key->pointer map, letting the driver allocate,
 * insert, and look up regions with identical code in both builds.
 */

#ifndef MKT_SHM_TOC_H
#define MKT_SHM_TOC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct shm_toc shm_toc;

/*
 * Accumulates the arena size and key count before the table is created,
 * mirroring PostgreSQL's estimator so the driver's sizing code is unchanged.
 */
typedef struct shm_toc_estimator
{
	size_t space_for_chunks;
	size_t number_of_keys;
} shm_toc_estimator;

#define MKT_TOC_ALIGN(sz) (((size_t)(sz) + 7) & ~(size_t)7)

#define shm_toc_initialize_estimator(e) \
	((e)->space_for_chunks = 0, (e)->number_of_keys = 0)
#define shm_toc_estimate_chunk(e, sz) \
	((e)->space_for_chunks += MKT_TOC_ALIGN(sz))
#define shm_toc_estimate_keys(e, cnt) ((e)->number_of_keys += (size_t)(cnt))

/* Arena bytes needed for the estimated chunks (keys are tracked separately).
 */
extern size_t shm_toc_estimate(shm_toc_estimator *e);

/*
 * Create a table over the caller-provided arena [address, address+nbytes).
 * The arena holds the inserted regions; the key map is kept separately on the
 * heap. Free with mkt_shm_toc_free. The magic is retained for parity and
 * otherwise unused.
 */
extern shm_toc *shm_toc_create(uint64_t magic, void *address, size_t nbytes);

/* Bump-allocate nbytes (8-byte aligned) from the arena. */
extern void *shm_toc_allocate(shm_toc *toc, size_t nbytes);

/* Publish a region under a key. */
extern void shm_toc_insert(shm_toc *toc, uint64_t key, void *address);

/*
 * Look up a region by key. Returns NULL for a missing key when noError is
 * true; otherwise a missing key is a fatal error.
 */
extern void *shm_toc_lookup(shm_toc *toc, uint64_t key, bool noError);

/* Standalone-only: release the table's heap bookkeeping (not the arena). */
extern void mkt_shm_toc_free(shm_toc *toc);

#endif /* MKT_SHM_TOC_H */
