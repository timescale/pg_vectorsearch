/*
 * vec32_source.h - Iterator interface for streaming vector data
 *
 * Abstracts the data source so index builds can stream vectors from
 * memory, HDF5 files, or any other source without loading the full
 * dataset at once.
 *
 * A build performs two passes over the source:
 *   1. Sample vectors for clustering (strided reads)
 *   2. Reset, then assign all vectors to clusters (sequential reads)
 */

#ifndef VEC32_SOURCE_H
#define VEC32_SOURCE_H

#include <stdbool.h>
#include <stdint.h>

typedef struct Vec32Source Vec32Source;

struct Vec32Source
{
	/*
	 * Get the next vector, advancing by stride positions.
	 *
	 * stride=1 reads sequentially. stride>1 skips ahead, enabling
	 * uniform sampling without random access. Implementations may
	 * optimize internally (e.g., pointer arithmetic for arrays,
	 * hyperslab offset for HDF5).
	 *
	 * Returns true if a vector was produced.
	 * vec_out points to dim floats (valid until the next call).
	 * id_out receives the vector ID (position in the dataset).
	 */
	bool (*next)(
			Vec32Source	 *src,
			uint32_t	  stride,
			const float **vec_out,
			uint32_t	 *id_out);

	/* Reset to the beginning for another pass. */
	void (*reset)(Vec32Source *src);

	/*
	 * Bulk-read all vectors into a contiguous buffer.
	 * dest must hold nvecs * dim floats.
	 * Optional — NULL means not supported (use next() loop).
	 */
	bool (*read_all)(Vec32Source *src, float *dest);

	uint32_t nvecs; /* total number of vectors */
	uint32_t dim;	/* vector dimension */
};

/*
 * Array-backed vector source (wraps a flat float array).
 *
 * The source borrows the data pointer — the caller must keep
 * the array alive until the build completes.
 */
typedef struct MktArraySource
{
	Vec32Source	 base;
	const float *data;
	uint32_t	 pos;
} MktArraySource;

void mkt_array_source_init(
		MktArraySource *src, const float *data, uint32_t nvecs, uint32_t dim);

#endif /* VEC32_SOURCE_H */
