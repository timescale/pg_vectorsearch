/*
 * hdf5_source.h - HDF5 streaming vector source
 *
 * Reads vectors one at a time from an HDF5 dataset using hyperslab
 * selection. Only one row is in memory at a time (plus a small
 * read buffer), enabling index builds on datasets larger than RAM.
 */

#ifndef MKT_HDF5_SOURCE_H
#define MKT_HDF5_SOURCE_H

#include "mkt_config.h"

#ifdef MKT_HAVE_HDF5

#include <hdf5.h>

#include "standalone/vec32_source.h"

typedef struct MktHdf5Source
{
	Vec32Source base;
	hid_t		file;	/* HDF5 file handle */
	hid_t		dset;	/* dataset handle */
	hid_t		fspace; /* file dataspace */
	hid_t		mspace; /* memory dataspace (one row) */
	float	   *buf;	/* read buffer [dim] */
	uint32_t	pos;	/* current row */
} MktHdf5Source;

/*
 * Open an HDF5 dataset as a streaming vector source.
 *
 * path:    HDF5 file path
 * dataset: dataset name (e.g., "train", "test")
 *
 * Returns 0 on success, -1 on failure.
 * Call mkt_hdf5_source_close() when done.
 */
int mkt_hdf5_source_open(
		MktHdf5Source *src, const char *path, const char *dataset);

/*
 * Close the HDF5 source and release all handles.
 */
void mkt_hdf5_source_close(MktHdf5Source *src);

#endif /* MKT_HAVE_HDF5 */
#endif /* MKT_HDF5_SOURCE_H */
