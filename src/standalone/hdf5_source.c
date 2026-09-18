/*
 * hdf5_source.c - HDF5 streaming vector source
 *
 * Reads one row at a time via H5Sselect_hyperslab. The file and
 * dataset remain open between calls, with a single-row memory
 * dataspace reused for each read.
 */

#include "mkt_config.h"

#ifdef MKT_HAVE_HDF5

#include <stdlib.h>

#include "standalone/hdf5_source.h"

static bool
hdf5_source_next(
		Vec32Source	 *src,
		uint32_t	  stride,
		const float **vec_out,
		uint32_t	 *id_out)
{
	MktHdf5Source *hs = (MktHdf5Source *)src;
	if (hs->pos >= src->nvecs)
		return false;

	/* Select one row from the file dataspace */
	hsize_t offset[2] = {hs->pos, 0};
	hsize_t count[2]  = {1, src->dim};
	H5Sselect_hyperslab(hs->fspace, H5S_SELECT_SET, offset, NULL, count, NULL);

	/* Read into buffer */
	H5Dread(hs->dset,
			H5T_NATIVE_FLOAT,
			hs->mspace,
			hs->fspace,
			H5P_DEFAULT,
			hs->buf);

	*id_out	 = hs->pos;
	*vec_out = hs->buf;
	hs->pos += stride;
	return true;
}

static void
hdf5_source_reset(Vec32Source *src)
{
	MktHdf5Source *hs = (MktHdf5Source *)src;
	hs->pos			  = 0;
}

static bool
hdf5_source_read_all(Vec32Source *src, float *dest)
{
	MktHdf5Source *hs	   = (MktHdf5Source *)src;
	hsize_t		   dims[2] = {src->nvecs, src->dim};
	hid_t		   mspace  = H5Screate_simple(2, dims, NULL);
	if (mspace < 0)
		return false;

	H5Sselect_all(hs->fspace);
	herr_t err = H5Dread(
			hs->dset, H5T_NATIVE_FLOAT, mspace, hs->fspace, H5P_DEFAULT, dest);
	H5Sclose(mspace);
	return err >= 0;
}

int
mkt_hdf5_source_open(MktHdf5Source *src, const char *path, const char *dataset)
{
	src->file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (src->file < 0)
		return -1;

	src->dset = H5Dopen2(src->file, dataset, H5P_DEFAULT);
	if (src->dset < 0)
	{
		H5Fclose(src->file);
		return -1;
	}

	/* Get dataset dimensions */
	src->fspace = H5Dget_space(src->dset);
	hsize_t dims[2];
	H5Sget_simple_extent_dims(src->fspace, dims, NULL);

	src->base.nvecs	   = (uint32_t)dims[0];
	src->base.dim	   = (uint32_t)dims[1];
	src->base.next	   = hdf5_source_next;
	src->base.reset	   = hdf5_source_reset;
	src->base.read_all = hdf5_source_read_all;
	src->pos		   = 0;

	/* Create memory dataspace for one row */
	hsize_t mdims[1] = {dims[1]};
	src->mspace		 = H5Screate_simple(1, mdims, NULL);

	/* Allocate read buffer */
	src->buf = malloc(dims[1] * sizeof(float));

	return 0;
}

void
mkt_hdf5_source_close(MktHdf5Source *src)
{
	if (src == NULL)
		return;

	free(src->buf);
	src->buf = NULL;

	if (src->mspace >= 0)
		H5Sclose(src->mspace);
	if (src->fspace >= 0)
		H5Sclose(src->fspace);
	if (src->dset >= 0)
		H5Dclose(src->dset);
	if (src->file >= 0)
		H5Fclose(src->file);
}

#endif /* MKT_HAVE_HDF5 */
