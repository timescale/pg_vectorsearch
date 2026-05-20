/*
 * atomics.h - Cross-platform atomic operations
 *
 * Standalone builds use C11 stdatomic.h.
 * PostgreSQL builds use pg_atomic.h.
 * Both compile to the same GCC __atomic_* intrinsics underneath,
 * but the types are incompatible (C11 _Atomic qualifier vs PG
 * wrapper struct).
 */

#ifndef MKT_ATOMICS_H
#define MKT_ATOMICS_H

#ifdef MKT_STANDALONE

#include <stdatomic.h>

typedef _Atomic(uint32_t) mkt_atomic_uint32;

static inline void
mkt_atomic_init_u32(mkt_atomic_uint32 *a, uint32_t val)
{
	atomic_store(a, val);
}

static inline uint32_t
mkt_atomic_read_u32(mkt_atomic_uint32 *a)
{
	return atomic_load(a);
}

static inline uint32_t
mkt_atomic_fetch_add_u32(mkt_atomic_uint32 *a, uint32_t val)
{
	return atomic_fetch_add(a, val);
}

#else /* PostgreSQL */

#include <postgres.h>

#include <port/atomics.h>

typedef pg_atomic_uint32 mkt_atomic_uint32;

static inline void
mkt_atomic_init_u32(mkt_atomic_uint32 *a, uint32_t val)
{
	pg_atomic_init_u32(a, val);
}

static inline uint32_t
mkt_atomic_read_u32(mkt_atomic_uint32 *a)
{
	return pg_atomic_read_u32(a);
}

static inline uint32_t
mkt_atomic_fetch_add_u32(mkt_atomic_uint32 *a, uint32_t val)
{
	return pg_atomic_fetch_add_u32(a, val);
}

#endif /* MKT_STANDALONE */

#endif /* MKT_ATOMICS_H */
