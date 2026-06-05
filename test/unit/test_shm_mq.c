/*
 * test_shm_mq.c - Single-reader/single-writer message queue (standalone) tests
 *
 * Covers the streaming semantics the posting drain relies on: a sender and
 * receiver on separate threads exchange ordered messages through a ring small
 * enough to force backpressure (blocking sends woken by the draining
 * receiver), the receiver observes SHM_MQ_DETACHED after the sender leaves and
 * the ring is empty, and a nowait receive on an empty queue would-blocks.
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#include "mkt_test.h"
#include "standalone/shm_mq.h"

TEST_GROUP(ShmMq);

#define MQ_NMSG 2000

typedef struct
{
	shm_mq		 *mq;
	uint32_t	  nmsg;
	_Atomic(int) *err;
	Latch		 *latch; /* owned by the caller; outlives this thread */
} MqSendArg;

static void *
mq_sender(void *arg)
{
	MqSendArg *a = (MqSendArg *)arg;

	/*
	 * The sender's latch must outlive the thread: after it detaches, the
	 * receiver still sets it while draining the buffered messages. So the
	 * caller owns it (a thread-stack latch would dangle). In the real build
	 * the worker latches live in stable shared memory for the same reason.
	 */
	mkt_latch_attach_self(a->latch);
	shm_mq_set_sender(a->mq, MyProc);

	shm_mq_handle *h = shm_mq_attach(a->mq, NULL, NULL);

	for (uint32_t i = 0; i < a->nmsg; i++)
	{
		shm_mq_result r = shm_mq_send(h, sizeof(i), &i, false, true);
		if (r != SHM_MQ_SUCCESS)
		{
			atomic_store(a->err, 1);
			break;
		}
	}
	shm_mq_detach(h);
	return NULL;
}

TEST(stream_ordered_with_backpressure_and_detach)
{
	/* A small arena: the ring holds only a handful of 4-byte messages, so the
	 * sender must block and be woken by the draining receiver. */
	void *arena = malloc(512);

	shm_mq *mq = shm_mq_create(arena, 512);

	/* Receiver = this thread. */
	Latch recv_latch;
	InitLatch(&recv_latch);
	mkt_latch_attach_self(&recv_latch);
	shm_mq_set_receiver(mq, MyProc);
	shm_mq_handle *rh = shm_mq_attach(mq, NULL, NULL);

	_Atomic(int) send_err = 0;
	Latch		 send_latch; /* owned here so it outlives the sender thread */
	InitLatch(&send_latch);
	MqSendArg sarg = {
			.mq = mq, .nmsg = MQ_NMSG, .err = &send_err, .latch = &send_latch};
	pthread_t sth;
	pthread_create(&sth, NULL, mq_sender, &sarg);

	uint32_t expected = 0;
	bool	 ok		  = true;
	for (;;)
	{
		size_t		  len;
		void		 *data;
		shm_mq_result r = shm_mq_receive(rh, &len, &data, false);

		if (r == SHM_MQ_DETACHED)
			break;
		if (r != SHM_MQ_SUCCESS || len != sizeof(uint32_t) ||
			*(uint32_t *)data != expected)
		{
			ok = false;
			break;
		}
		expected++;
	}

	pthread_join(sth, NULL);
	shm_mq_detach(rh);
	free(arena);

	ASSERT_TRUE(ok, "messages arrive in order, intact");
	ASSERT_EQ(0, atomic_load(&send_err), "every send succeeded");
	ASSERT_EQ(
			MQ_NMSG, (int)expected, "every message was received then detach");
}

TEST(nowait_receive_on_empty_would_block)
{
	void   *arena = malloc(256);
	shm_mq *mq	  = shm_mq_create(arena, 256);

	Latch recv_latch;
	InitLatch(&recv_latch);
	mkt_latch_attach_self(&recv_latch);
	shm_mq_set_receiver(mq, MyProc);
	shm_mq_handle *rh = shm_mq_attach(mq, NULL, NULL);

	size_t		  len;
	void		 *data;
	shm_mq_result r = shm_mq_receive(rh, &len, &data, true);

	ASSERT_EQ(SHM_MQ_WOULD_BLOCK, r, "empty nowait receive would-blocks");

	shm_mq_detach(rh);
	free(arena);
}
