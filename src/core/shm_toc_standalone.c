/*
 * shm_toc_standalone.c - Keyed shared-region table over a heap arena
 *
 * Standalone implementation of the shm_toc API the parallel build uses. See
 * mkt_shm_toc.h. PG builds use PostgreSQL's shm_toc (backed by a DSM segment)
 * instead, so this file is compiled only for standalone.
 */

#ifdef MKT_STANDALONE

#include <stdio.h>
#include <stdlib.h>

#include "core/mkt_shm_toc.h"

struct shm_toc
{
	uint64_t magic;
	char	*arena;		/* caller-provided chunk arena */
	size_t	 nbytes;	/* arena capacity */
	size_t	 allocated; /* bump offset into the arena */

	/* Key -> region map (grown on insert; separate from the arena). */
	uint64_t *keys;
	void	**ptrs;
	size_t	  nkeys;
	size_t	  cap;
};

size_t
shm_toc_estimate(shm_toc_estimator *e)
{
	return e->space_for_chunks;
}

shm_toc *
shm_toc_create(uint64_t magic, void *address, size_t nbytes)
{
	shm_toc *toc = malloc(sizeof(shm_toc));

	if (toc == NULL)
	{
		fprintf(stderr, "shm_toc_create: out of memory\n");
		abort();
	}

	toc->magic	   = magic;
	toc->arena	   = (char *)address;
	toc->nbytes	   = nbytes;
	toc->allocated = 0;
	toc->keys	   = NULL;
	toc->ptrs	   = NULL;
	toc->nkeys	   = 0;
	toc->cap	   = 0;
	return toc;
}

void *
shm_toc_allocate(shm_toc *toc, size_t nbytes)
{
	size_t offset = MKT_TOC_ALIGN(toc->allocated);
	size_t end	  = offset + MKT_TOC_ALIGN(nbytes);

	if (end > toc->nbytes)
	{
		fprintf(stderr,
				"shm_toc_allocate: arena exhausted (need %zu, have %zu)\n",
				end,
				toc->nbytes);
		abort();
	}

	toc->allocated = end;
	return toc->arena + offset;
}

void
shm_toc_insert(shm_toc *toc, uint64_t key, void *address)
{
	if (toc->nkeys == toc->cap)
	{
		size_t	  newcap = toc->cap == 0 ? 8 : toc->cap * 2;
		uint64_t *nk	 = realloc(toc->keys, newcap * sizeof(uint64_t));
		void	**np	 = realloc(toc->ptrs, newcap * sizeof(void *));

		if (nk == NULL || np == NULL)
		{
			fprintf(stderr, "shm_toc_insert: out of memory\n");
			abort();
		}
		toc->keys = nk;
		toc->ptrs = np;
		toc->cap  = newcap;
	}

	toc->keys[toc->nkeys] = key;
	toc->ptrs[toc->nkeys] = address;
	toc->nkeys++;
}

void *
shm_toc_lookup(shm_toc *toc, uint64_t key, bool noError)
{
	for (size_t i = 0; i < toc->nkeys; i++)
		if (toc->keys[i] == key)
			return toc->ptrs[i];

	if (!noError)
	{
		fprintf(stderr,
				"shm_toc_lookup: key %llu not found\n",
				(unsigned long long)key);
		abort();
	}
	return NULL;
}

void
mkt_shm_toc_free(shm_toc *toc)
{
	if (toc == NULL)
		return;
	free(toc->keys);
	free(toc->ptrs);
	free(toc);
}

#endif /* MKT_STANDALONE */
