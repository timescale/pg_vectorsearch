/*
 * mkt_shm_mq.h - Single-reader/single-writer message queue
 *
 * Phase 3 of the build streams completed posting pages from each worker to the
 * leader, which writes them. The transport is a shm_mq: the worker sends pages
 * (blocking when the ring is full, which bounds its memory to ~one page per
 * cluster), and the leader drains every queue, waking on its latch.
 *
 * In a PostgreSQL build this is PostgreSQL's shm_mq over a DSM segment. In a
 * standalone (thread-based) build we provide the same API over a byte ring in
 * the toc arena, woken via the Latch shim. Standalone deliberately uses the
 * SAME streaming path (not a direct write) so the drain/finalize code is the
 * code its tests exercise.
 *
 * The "proc" a sender/receiver registers is just the thread's latch (MyLatch);
 * sends and detaches set the peer's latch so a blocked counterpart wakes.
 */

#ifndef MKT_SHM_MQ_H
#define MKT_SHM_MQ_H

#ifdef MKT_STANDALONE

#include <stddef.h>
#include <stdint.h>

#include "core/mkt_latch.h"

/* Opaque DSM segment; unused in standalone (threads share the heap). */
typedef struct dsm_segment dsm_segment;

/* A participant is identified by its latch, the only thing the queue wakes. */
#define MyProc MyLatch

typedef enum shm_mq_result
{
	SHM_MQ_SUCCESS,
	SHM_MQ_WOULD_BLOCK,
	SHM_MQ_DETACHED,
} shm_mq_result;

typedef struct shm_mq		 shm_mq;
typedef struct shm_mq_handle shm_mq_handle;

/* Lay a queue over [address, address+size); the ring follows the header. */
extern shm_mq *shm_mq_create(void *address, size_t size);

extern void shm_mq_set_sender(shm_mq *mq, Latch *proc);
extern void shm_mq_set_receiver(shm_mq *mq, Latch *proc);

/* Bind the calling thread (sender or receiver, per the set_* above). */
extern shm_mq_handle *
shm_mq_attach(shm_mq *mq, dsm_segment *seg, void *handle);

/*
 * Send nbytes. With nowait=false, blocks (backpressure) until the message
 * fits; with nowait=true, returns SHM_MQ_WOULD_BLOCK instead. Returns
 * SHM_MQ_DETACHED if the receiver has gone. force_flush is accepted for
 * parity.
 */
extern shm_mq_result shm_mq_send(
		shm_mq_handle *mqh,
		size_t		   nbytes,
		const void	  *data,
		bool		   nowait,
		bool		   force_flush);

/*
 * Receive the next message. *datap points at a buffer valid until the next
 * receive on this handle. With nowait=true, returns SHM_MQ_WOULD_BLOCK when
 * empty; returns SHM_MQ_DETACHED once the sender has detached and the ring is
 * drained.
 */
extern shm_mq_result
shm_mq_receive(shm_mq_handle *mqh, size_t *nbytesp, void **datap, bool nowait);

/* Detach this end, waking the peer. */
extern void shm_mq_detach(shm_mq_handle *mqh);

#else /* !MKT_STANDALONE */

#include <storage/shm_mq.h>

#endif /* MKT_STANDALONE */

#endif /* MKT_SHM_MQ_H */
