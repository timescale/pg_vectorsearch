/*
 * shm_mq.c - Single-reader/single-writer message queue over a byte ring
 *
 * Standalone implementation of the shm_mq API the posting drain uses (see
 * mkt_shm_mq.h). PG builds use PostgreSQL's shm_mq instead, so this file is
 * compiled only for standalone.
 *
 * One sender, one receiver. The ring holds length-prefixed messages. Blocking
 * waits use the participant latches: a send blocks on the sender's latch until
 * the receiver frees space, and the receiver blocks on its own latch until the
 * sender adds data; each side sets the other's latch on progress and on
 * detach.
 */

#ifdef MKT_STANDALONE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/shm_mq.h"

struct shm_mq
{
	Latch		   *sender_latch;
	Latch		   *receiver_latch;
	pthread_mutex_t mutex;
	bool			sender_detached;
	bool			receiver_detached;
	size_t			ring_size;
	size_t			head; /* write offset */
	size_t			tail; /* read offset */
	size_t			used; /* bytes occupied */
};

struct shm_mq_handle
{
	shm_mq *mq;
	bool	is_sender;
	char   *bounce; /* receiver's copy-out buffer */
	size_t	bounce_cap;
};

static inline char *
mq_ring(shm_mq *mq)
{
	return (char *)mq + sizeof(shm_mq);
}

/* Circular write of n bytes from src at the head; advances head + used. */
static void
ring_write(shm_mq *mq, const void *src, size_t n)
{
	char  *ring	 = mq_ring(mq);
	size_t first = mq->ring_size - mq->head;

	if (first >= n)
	{
		memcpy(ring + mq->head, src, n);
	}
	else
	{
		memcpy(ring + mq->head, src, first);
		memcpy(ring, (const char *)src + first, n - first);
	}
	mq->head = (mq->head + n) % mq->ring_size;
	mq->used += n;
}

/* Circular read of n bytes into dst from the tail; advances tail + used. */
static void
ring_read(shm_mq *mq, void *dst, size_t n)
{
	char  *ring	 = mq_ring(mq);
	size_t first = mq->ring_size - mq->tail;

	if (first >= n)
	{
		memcpy(dst, ring + mq->tail, n);
	}
	else
	{
		memcpy(dst, ring + mq->tail, first);
		memcpy((char *)dst + first, ring, n - first);
	}
	mq->tail = (mq->tail + n) % mq->ring_size;
	mq->used -= n;
}

shm_mq *
shm_mq_create(void *address, size_t size)
{
	shm_mq *mq = (shm_mq *)address;

	if (size <= sizeof(shm_mq) + sizeof(size_t))
	{
		fprintf(stderr, "shm_mq_create: queue too small (%zu)\n", size);
		abort();
	}

	mq->sender_latch   = NULL;
	mq->receiver_latch = NULL;
	pthread_mutex_init(&mq->mutex, NULL);
	mq->sender_detached	  = false;
	mq->receiver_detached = false;
	mq->ring_size		  = size - sizeof(shm_mq);
	mq->head			  = 0;
	mq->tail			  = 0;
	mq->used			  = 0;
	return mq;
}

void
shm_mq_set_sender(shm_mq *mq, Latch *proc)
{
	mq->sender_latch = proc;
}

void
shm_mq_set_receiver(shm_mq *mq, Latch *proc)
{
	mq->receiver_latch = proc;
}

shm_mq_handle *
shm_mq_attach(shm_mq *mq, dsm_segment *seg, void *handle)
{
	shm_mq_handle *h = malloc(sizeof(shm_mq_handle));

	(void)seg;
	(void)handle;

	if (h == NULL)
	{
		fprintf(stderr, "shm_mq_attach: out of memory\n");
		abort();
	}
	h->mq		  = mq;
	h->is_sender  = (mq->sender_latch == MyLatch);
	h->bounce	  = NULL;
	h->bounce_cap = 0;
	return h;
}

shm_mq_result
shm_mq_send(
		shm_mq_handle *mqh,
		size_t		   nbytes,
		const void	  *data,
		bool		   nowait,
		bool		   force_flush)
{
	shm_mq *mq	 = mqh->mq;
	size_t	need = sizeof(size_t) + nbytes;

	(void)force_flush;

	if (need > mq->ring_size)
	{
		fprintf(stderr,
				"shm_mq_send: message %zu exceeds ring %zu\n",
				nbytes,
				mq->ring_size);
		abort();
	}

	pthread_mutex_lock(&mq->mutex);

	while (mq->ring_size - mq->used < need)
	{
		if (mq->receiver_detached)
		{
			pthread_mutex_unlock(&mq->mutex);
			return SHM_MQ_DETACHED;
		}
		if (nowait)
		{
			pthread_mutex_unlock(&mq->mutex);
			return SHM_MQ_WOULD_BLOCK;
		}
		/* Wait on our own latch; the receiver sets it when it frees space. */
		pthread_mutex_unlock(&mq->mutex);
		WaitLatch(MyLatch, WL_LATCH_SET, -1, 0);
		ResetLatch(MyLatch);
		pthread_mutex_lock(&mq->mutex);
	}

	ring_write(mq, &nbytes, sizeof(size_t));
	ring_write(mq, data, nbytes);

	pthread_mutex_unlock(&mq->mutex);

	if (mq->receiver_latch != NULL)
		SetLatch(mq->receiver_latch);
	return SHM_MQ_SUCCESS;
}

shm_mq_result
shm_mq_receive(shm_mq_handle *mqh, size_t *nbytesp, void **datap, bool nowait)
{
	shm_mq *mq = mqh->mq;
	size_t	len;

	pthread_mutex_lock(&mq->mutex);

	while (mq->used == 0)
	{
		if (mq->sender_detached)
		{
			pthread_mutex_unlock(&mq->mutex);
			return SHM_MQ_DETACHED;
		}
		if (nowait)
		{
			pthread_mutex_unlock(&mq->mutex);
			return SHM_MQ_WOULD_BLOCK;
		}
		pthread_mutex_unlock(&mq->mutex);
		WaitLatch(MyLatch, WL_LATCH_SET, -1, 0);
		ResetLatch(MyLatch);
		pthread_mutex_lock(&mq->mutex);
	}

	ring_read(mq, &len, sizeof(size_t));

	if (len > mqh->bounce_cap)
	{
		mqh->bounce = realloc(mqh->bounce, len);
		if (mqh->bounce == NULL)
		{
			fprintf(stderr, "shm_mq_receive: out of memory\n");
			abort();
		}
		mqh->bounce_cap = len;
	}
	ring_read(mq, mqh->bounce, len);

	pthread_mutex_unlock(&mq->mutex);

	/* Freed ring space — wake a sender blocked on backpressure. */
	if (mq->sender_latch != NULL)
		SetLatch(mq->sender_latch);

	*nbytesp = len;
	*datap	 = mqh->bounce;
	return SHM_MQ_SUCCESS;
}

void
shm_mq_detach(shm_mq_handle *mqh)
{
	shm_mq *mq = mqh->mq;

	pthread_mutex_lock(&mq->mutex);
	if (mqh->is_sender)
		mq->sender_detached = true;
	else
		mq->receiver_detached = true;
	pthread_mutex_unlock(&mq->mutex);

	/* Wake the peer so it observes the detach. */
	if (mqh->is_sender)
	{
		if (mq->receiver_latch != NULL)
			SetLatch(mq->receiver_latch);
	}
	else if (mq->sender_latch != NULL)
	{
		SetLatch(mq->sender_latch);
	}

	free(mqh->bounce);
	free(mqh);
}

#endif /* MKT_STANDALONE */
